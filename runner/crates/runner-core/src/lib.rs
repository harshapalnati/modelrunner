//! Core scheduling, KV cache, and sampling (skeleton)

pub struct Scheduler;

impl Scheduler {
    pub fn new() -> Self { Self }
}

pub mod decode {
    use runner_backend::InferenceBackend;
    use runner_common::Result;
    use super::sampler::{sample_top_k_top_p};

    pub fn generate_once(
        backend: &dyn InferenceBackend,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let _ = max_tokens; // unused in mock path
        // Try to use backend tokenize/detokenize; if not available, fall back to echo.
        let tokens = backend.tokenize(prompt).unwrap_or_default();
        // For now, return detokenized prompt; sampling comes once logits are exposed
        let _ = sample_top_k_top_p::<rand::rngs::StdRng>(&[0.0_f32; 1], 0, 1.0, 1.0, None);
        let text = backend.detokenize(&tokens).unwrap_or_else(|_| prompt.to_string());
        Ok(text)
    }
}

// Simple scheduler v1 (skeleton)
pub mod scheduler {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use tokio::sync::{mpsc, oneshot};
    use tokio::time::{self, Duration};
    use runner_backend::InferenceBackend;
    use super::kv::{PagedKvManager, Reservation, PrefixCache};

    pub struct Request {
        pub prompt: String,
        pub respond: oneshot::Sender<String>,
        pub max_tokens: usize,
        pub reservation: Option<Reservation>,
    }

    #[derive(Clone)]
    pub struct Handle {
        tx: mpsc::Sender<Request>,
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
                    // drain a batch
                    let mut batch: Vec<Request> = Vec::with_capacity(32);
                    while let Ok(req) = rx.try_recv() { batch.push(req); if batch.len() >= 32 { break; } }
                    qd.store(rx.len(), Ordering::Relaxed);
                    if batch.is_empty() { continue; }
                    lbs.store(batch.len(), Ordering::Relaxed);
                    // naive parallel handling within a tick
                    for req in batch {
                        let backend_ref = backend.clone();
                        let _kv = kv_bg.clone();
                        tokio::spawn(async move {
                            let text = runner_core_generate_once(backend_ref.as_ref(), &req.prompt, req.max_tokens);
                            let _ = req.respond.send(text.unwrap_or_default());
                            drop(req.reservation); // release KV on completion
                        });
                    }
                }
            });
            Handle { tx, queue_depth, last_batch_size, kv, prefix }
        }

        pub async fn enqueue(handle: &Handle, prompt: String, max_tokens: usize) -> String {
            // Admission control via paged KV prediction
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

    fn runner_core_generate_once(backend: &dyn InferenceBackend, prompt: &str, max_tokens: usize) -> runner_common::Result<String> {
        crate::decode::generate_once(backend, prompt, max_tokens)
    }
}

// Naive KV cache placeholder
pub mod kv {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

    pub struct NaiveKvCache { pub capacity_bytes: usize }
    impl NaiveKvCache { pub fn new(capacity_bytes: usize) -> Self { Self { capacity_bytes } } }

    pub struct PagedKvManager {
        capacity_blocks: usize,
        used_blocks: AtomicUsize,
        enable_spill: bool,
        free_list: Mutex<Vec<usize>>, // simplistic free list for defrag demo
    }

    impl PagedKvManager {
        pub const TOKENS_PER_BLOCK: usize = 32;
        pub fn new(capacity_bytes: usize) -> Arc<Self> {
            let capacity_blocks = capacity_bytes / 4096;
            Arc::new(Self { capacity_blocks, used_blocks: AtomicUsize::new(0), enable_spill: false, free_list: Mutex::new(Vec::new()) })
        }
        pub fn tokens_to_blocks(&self, tokens: usize) -> usize {
            (tokens + Self::TOKENS_PER_BLOCK - 1) / Self::TOKENS_PER_BLOCK
        }
        pub fn try_reserve(self: &Arc<Self>, blocks: usize) -> Option<Reservation> {
            loop {
                let used = self.used_blocks.load(Ordering::Relaxed);
                if used + blocks > self.capacity_blocks { return None; }
                if self.used_blocks.compare_exchange(used, used + blocks, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                    return Some(Reservation { manager: self.clone(), blocks });
                }
            }
        }
        pub fn used_blocks(&self) -> usize { self.used_blocks.load(Ordering::Relaxed) }
        pub fn capacity_blocks(&self) -> usize { self.capacity_blocks }
        fn release(&self, blocks: usize) {
            self.used_blocks.fetch_sub(blocks, Ordering::SeqCst);
            let mut fl = self.free_list.lock().unwrap();
            fl.push(blocks);
        }
        pub fn defragment(&self) { let mut fl = self.free_list.lock().unwrap(); fl.clear(); }
        pub fn enable_spill_to_host(&mut self, enable: bool) { self.enable_spill = enable; }
    }

    pub struct Reservation { pub(crate) manager: Arc<PagedKvManager>, pub(crate) blocks: usize }
    impl Drop for Reservation { fn drop(&mut self) { self.manager.release(self.blocks) } }

    #[derive(Default)]
    pub struct PrefixCache { counts: Mutex<HashMap<u64, usize>>, tokens: Mutex<HashMap<u64, Vec<u32>>> }
    impl PrefixCache {
        pub fn new() -> Arc<Self> { Arc::new(Self::default()) }
        pub fn hash_prefix(&self, text: &str) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            let slice = if text.len() > 256 { &text[..256] } else { text };
            slice.hash(&mut hasher);
            hasher.finish()
        }
        pub fn note(&self, h: u64) { let mut g = self.counts.lock().unwrap(); *g.entry(h).or_insert(0) += 1; }
        pub fn is_common(&self, h: u64) -> bool { let g = self.counts.lock().unwrap(); g.get(&h).copied().unwrap_or(0) >= 2 }
        pub fn put_tokens(&self, h: u64, toks: Vec<u32>) { let mut t = self.tokens.lock().unwrap(); t.insert(h, toks); }
        pub fn get_tokens(&self, h: u64) -> Option<Vec<u32>> { let t = self.tokens.lock().unwrap(); t.get(&h).cloned() }
    }
}

pub mod sampler {
    use rand::prelude::*;

    pub fn sample_top_k_top_p<R: Rng + ?Sized>(
        logits: &[f32],
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: Option<u64>,
    ) -> usize {
        let mut rng: StdRng = match seed { Some(s) => SeedableRng::seed_from_u64(s), None => StdRng::from_entropy() };
        if logits.is_empty() { return 0; }
        let mut pairs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l / temperature.max(1e-4))).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cutoff = pairs.len();
        if top_k > 0 { cutoff = cutoff.min(top_k); }
        let mut sum = 0.0_f32;
        let mut probs: Vec<(usize, f32)> = Vec::with_capacity(cutoff);
        for &(i, l) in &pairs[..cutoff] {
            let p = (l).exp();
            probs.push((i, p));
            sum += p;
        }
        probs.iter_mut().for_each(|p| p.1 /= sum.max(1e-9));
        if top_p < 1.0 {
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut acc = 0.0_f32;
            let mut keep = 0;
            for &(_, p) in &probs { acc += p; keep += 1; if acc >= top_p { break; } }
            probs.truncate(keep);
            let z: f32 = probs.iter().map(|p| p.1).sum();
            for p in &mut probs { p.1 /= z.max(1e-9); }
        }
        let r: f32 = rng.gen();
        let mut acc = 0.0_f32;
        for (i, p) in probs { acc += p; if r <= acc { return i; } }
        pairs[0].0
    }
}

