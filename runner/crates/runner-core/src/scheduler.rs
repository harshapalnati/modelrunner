use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{self, Duration};
use runner_backend::InferenceBackend;
use crate::kv::{PagedKvManager, Reservation, PrefixCache};

pub struct Request {
    pub prompt: String,
    pub respond: oneshot::Sender<String>,
    pub max_tokens: usize,
    pub reservation: Option<Reservation>,
}

#[derive(Clone)]
pub struct Handle {
    pub(crate) tx: mpsc::Sender<Request>,
    pub queue_depth: Arc<AtomicUsize>,
    pub last_batch_size: Arc<AtomicUsize>,
    pub kv: Arc<PagedKvManager>,
    pub prefix: Arc<PrefixCache>,
}

pub struct SchedulerV1;

impl SchedulerV1 {
    pub fn start(backend: Arc<dyn InferenceBackend>, kv: Arc<PagedKvManager>, prefix: Arc<PrefixCache>) -> Handle {
        let (tx, mut rx) = mpsc::channel::<Request>(1024);
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let last_batch_size = Arc::new(AtomicUsize::new(0));
        let qd = queue_depth.clone();
        let lbs = last_batch_size.clone();
        let kv_bg = kv.clone();
        tokio::spawn(async move {
            let mut ticker = time::interval(Duration::from_millis(2));
            loop {
                ticker.tick().await;
                let mut batch: Vec<Request> = Vec::with_capacity(32);
                while let Ok(req) = rx.try_recv() { batch.push(req); if batch.len() >= 32 { break; } }
                qd.store(rx.len(), Ordering::Relaxed);
                if batch.is_empty() { continue; }
                lbs.store(batch.len(), Ordering::Relaxed);
                for req in batch {
                    let backend_ref = backend.clone();
                    let _kv = kv_bg.clone();
                    tokio::spawn(async move {
                        let text = super::decode::generate_once(backend_ref.as_ref(), &req.prompt, req.max_tokens);
                        let _ = req.respond.send(text.unwrap_or_default());
                        drop(req.reservation);
                    });
                }
            }
        });
        Handle { tx, queue_depth, last_batch_size, kv, prefix }
    }

    pub async fn enqueue(handle: &Handle, prompt: String, max_tokens: usize) -> String {
        let est_prompt_tokens = std::cmp::max(1, prompt.len() / 4);
        let prefix_hash = handle.prefix.hash_prefix(&prompt);
        handle.prefix.note(prefix_hash);
        let mut total_tokens = est_prompt_tokens + max_tokens;
        if handle.prefix.is_common(prefix_hash) { total_tokens = total_tokens.saturating_sub(32); }
        let predicted_blocks = handle.kv.tokens_to_blocks(total_tokens);
        let reservation = handle.kv.try_reserve(predicted_blocks);
        if reservation.is_none() {
            return String::from("SERVER_BUSY: insufficient KV capacity");
        }
        let (tx, rx) = oneshot::channel();
        let _ = handle.tx.send(Request { prompt, respond: tx, max_tokens, reservation }).await;
        rx.await.unwrap_or_default()
    }
}

