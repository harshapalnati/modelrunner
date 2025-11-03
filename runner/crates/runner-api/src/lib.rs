//! HTTP API (skeleton -> minimal JSON + SSE)

use std::sync::Arc;

use axum::{
    extract::State,
    response::{sse::{Event, Sse}, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use axum::extract::ws::{WebSocketUpgrade, Message};
use once_cell::sync::Lazy;
use prometheus::{Encoder, IntCounter, IntCounterVec, Histogram, HistogramOpts, TextEncoder};
use runner_backend::{mock::MockBackend, InferenceBackend};
use runner_backend_llamacpp::LlamaCppBackend;
use runner_core::decode::generate_once;
use runner_core::scheduler::{SchedulerV1, Handle};
use runner_core::kv::{PagedKvManager, PrefixCache};
use runner_common::Result;
use tokio_stream::{wrappers::ReceiverStream, StreamExt as _};
use runner_obs::{init as obs_init, spawn_gpu_polling};

#[derive(Clone)]
pub struct AppState {
    backend: Arc<dyn InferenceBackend>,
    requests_total: IntCounter,
    tokens_generated_total: IntCounter,
    ttft_seconds: Histogram,
    scheduler: Handle,
    queue_depth_gauge: prometheus::IntGauge,
    batch_size_gauge: prometheus::IntGauge,
    kv_used_blocks: prometheus::IntGauge,
    kv_capacity_blocks: prometheus::IntGauge,
    limiter: RateLimiter,
    budgets: TokenBudgets,
    model_path: tokio::sync::RwLock<Option<String>>,
}

static ENCODER: Lazy<TextEncoder> = Lazy::new(|| TextEncoder::new());

pub fn app() -> Router {
    let backend: Arc<dyn InferenceBackend> = select_backend();
    obs_init();
    spawn_gpu_polling();
    let kv = PagedKvManager::new(512 * 1024 * 1024); // 512MB placeholder
    let prefix = PrefixCache::new();
    let scheduler = SchedulerV1::start(backend.clone(), kv.clone(), prefix.clone());
    let queue_depth_gauge = prometheus::register_int_gauge!("runner_queue_depth", "Scheduler queue depth").expect("gauge");
    let batch_size_gauge = prometheus::register_int_gauge!("runner_batch_size", "Last batch size").expect("gauge");
    let kv_used_blocks = prometheus::register_int_gauge!("runner_kv_used_blocks", "KV used blocks").expect("gauge");
    let kv_capacity_blocks = prometheus::register_int_gauge!("runner_kv_capacity_blocks", "KV capacity blocks").expect("gauge");
    let state = AppState {
        backend,
        requests_total: prometheus::register_int_counter!(
            "runner_requests_total",
            "Total number of /generate requests"
        )
        .expect("counter"),
        tokens_generated_total: prometheus::register_int_counter!(
            "runner_tokens_generated_total",
            "Total output tokens (approx)"
        )
        .expect("counter"),
        ttft_seconds: prometheus::register_histogram!(
            "runner_ttft_seconds",
            "Time to first token (approx for mock)"
        )
        .expect("histogram"),
        scheduler,
        queue_depth_gauge,
        batch_size_gauge,
        kv_used_blocks,
        kv_capacity_blocks,
    };

    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/sse/generate", get(generate_sse))
        .route("/ws/generate", get(ws_generate))
        .route("/admin/set_model", post(admin_set_model))
        .route("/openapi.json", get(openapi))
        .with_state(state)
}

fn select_backend() -> Arc<dyn InferenceBackend> {
    // Try llama backend first if model path is provided
    if let Ok(model_path) = std::env::var("RUNNER_MODEL") {
        let llama = LlamaCppBackend::new();
        // ignore params for now
        if llama.load_model(&model_path, runner_backend::LoadParams).is_ok() {
            tracing::info!(target: "api", "using llama.cpp backend with model {}", model_path);
            return Arc::new(llama);
        } else {
            tracing::warn!(target: "api", "failed to init llama backend, falling back to mock");
        }
    }
    Arc::new(MockBackend::new())
}

async fn metrics() -> impl IntoResponse {
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    ENCODER.encode(&metric_families, &mut buffer).unwrap();
    ([("content-type", ENCODER.format_type().to_string())], buffer)
}

async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    // ready if scheduler is running and (if a model was requested) a path is set
    let has_model = state.model_path.read().await.is_some();
    let running = state.scheduler.queue_depth.load(std::sync::atomic::Ordering::Relaxed) >= 0;
    if running { ([("content-type", "text/plain")], if has_model { "ready" } else { "ready-no-model" }) }
    else { ([("content-type", "text/plain")], "not-ready") }
}

#[derive(serde::Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[allow(dead_code)]
    max_tokens: Option<usize>,
}

#[derive(serde::Serialize)]
struct GenerateResponse { text: String }

async fn generate(State(state): State<AppState>, Json(req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    state.requests_total.inc();
    if !state.limiter.check_allow(&tenant_id()).await { return Json(GenerateResponse { text: String::from("RATE_LIMITED") }); }
    tracing::info!(target: "api", "generate request");
    let start = std::time::Instant::now();
    // update gauges from scheduler atomics
    state.queue_depth_gauge.set(state.scheduler.queue_depth.load(std::sync::atomic::Ordering::Relaxed) as i64);
    state.batch_size_gauge.set(state.scheduler.last_batch_size.load(std::sync::atomic::Ordering::Relaxed) as i64);
    // KV metrics (approx)
    state.kv_used_blocks.set(state.scheduler.kv.used_blocks() as i64);
    state.kv_capacity_blocks.set(state.scheduler.kv.capacity_blocks() as i64);

    let text = if let Some(model_path) = state.model_path.read().await.clone() {
        // Try llama backend path with real decode if available
        let llama = LlamaCppBackend::new();
        if llama.load_model(&model_path, runner_backend::LoadParams).is_ok() {
            #[cfg(llama_ffi)]
            {
                let _ = model_path; // silence unused in cfg
                // no streaming here; collect
                llama.generate_with_callback(&req.prompt, req.max_tokens.unwrap_or(64), |_piece| {}).unwrap_or_default()
            }
            #[cfg(not(llama_ffi))]
            { generate_once(state.backend.as_ref(), &req.prompt, req.max_tokens.unwrap_or(128)).unwrap_or_default() }
        } else {
            // Use scheduler (mock or other backend)
            runner_core::scheduler::SchedulerV1::enqueue(&state.scheduler, req.prompt.clone(), req.max_tokens.unwrap_or(128)).await
        }
    } else {
        runner_core::scheduler::SchedulerV1::enqueue(&state.scheduler, req.prompt.clone(), req.max_tokens.unwrap_or(128)).await
    };
    state.ttft_seconds.observe(start.elapsed().as_secs_f64());
    // very rough tokenization proxy for mock: bytes → tokens
    state.tokens_generated_total.inc_by(text.len() as u64);
    state.budgets.record(&tenant_id(), text.len() as u64).await;
    Json(GenerateResponse { text })
}

async fn generate_sse(State(state): State<AppState>) -> Sse<impl axum::response::sse::Stream<Item = Result<Event>>> {
    state.requests_total.inc();
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    let start = std::time::Instant::now();
    tokio::spawn(async move {
        if let Ok(model_path) = std::env::var("RUNNER_MODEL") {
            let llama = LlamaCppBackend::new();
            if llama.load_model(&model_path, runner_backend::LoadParams).is_ok() {
                #[cfg(llama_ffi)]
                {
                    let mut emit = |piece: String| {
                        let _ = tx.blocking_send(Ok(Event::default().data(piece)));
                    };
                    let _ = llama.generate_with_callback("", 0, |_| {}); // ensure symbols
                    // Generate from a default prompt for SSE test
                    let _ = llama.generate_with_callback("Hello", 64, &mut emit);
                }
                #[cfg(not(llama_ffi))]
                {
                    let _ = tx.send(Ok(Event::default().data("ffi disabled"))).await;
                }
            } else {
                let _ = tx.send(Ok(Event::default().data("model load failed"))).await;
            }
        } else {
            // fallback demo
            let tokens = ["hello", " ", "world", "!\n"];
            for t in tokens { let _ = tx.send(Ok(Event::default().data(t))).await; }
        }
    });
    let stream = ReceiverStream::new(rx).map(|e| e);
    state.ttft_seconds.observe(start.elapsed().as_secs_f64());
    Sse::new(stream)
}

async fn ws_generate(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(|mut socket| async move {
        let _ = socket.send(Message::Text("hello".into())).await;
        let _ = socket.send(Message::Text(" ".into())).await;
        let _ = socket.send(Message::Text("world".into())).await;
        let _ = socket.send(Message::Text("!".into())).await;
        let _ = socket.close().await;
    })
}

async fn openapi() -> impl IntoResponse {
    let spec = serde_json::json!({
        "openapi": "3.0.0",
        "info": {"title": "Next Inference API", "version": "0.1.0"},
        "paths": {
            "/generate": {"post": {"summary": "Generate text"}},
            "/v1/chat/completions": {"post": {"summary": "OpenAI chat subset"}},
            "/sse/generate": {"get": {"summary": "SSE stream demo"}},
            "/ws/generate": {"get": {"summary": "WebSocket stream demo"}},
            "/metrics": {"get": {"summary": "Prometheus metrics"}},
            "/healthz": {"get": {"summary": "health"}},
            "/readyz": {"get": {"summary": "readiness"}},
            "/admin/set_model": {"post": {"summary": "Hot load model"}}
        }
    });
    Json(spec)
}

#[derive(serde::Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct ChatRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[allow(dead_code)]
    stream: Option<bool>,
    #[allow(dead_code)]
    max_tokens: Option<usize>,
}

#[derive(serde::Serialize)]
struct ChatChoiceMessage { role: String, content: String }

#[derive(serde::Serialize)]
struct ChatChoice { index: u32, message: ChatChoiceMessage, finish_reason: String }

#[derive(serde::Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    choices: Vec<ChatChoice>,
}

async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    state.requests_total.inc();
    if !state.limiter.check_allow(&tenant_id()).await { return Json(ChatResponse { id: "rate-limited".into(), object: "chat.completion".into(), choices: vec![ChatChoice { index: 0, message: ChatChoiceMessage { role: "assistant".into(), content: String::from("RATE_LIMITED") }, finish_reason: "stop".into() }] }); }
    tracing::info!(target: "api", "chat request: {} messages", req.messages.len());
    let mut prompt = String::new();
    for m in &req.messages { if m.role == "system" || m.role == "user" { prompt.push_str(&m.content); prompt.push('\n'); } }
    let text = generate_once(state.backend.as_ref(), &prompt, req.max_tokens.unwrap_or(128))
        .unwrap_or_else(|_| String::new());
    let resp = ChatResponse {
        id: "chatcmpl-1".into(),
        object: "chat.completion".into(),
        choices: vec![ChatChoice { index: 0, message: ChatChoiceMessage { role: "assistant".into(), content: text }, finish_reason: "stop".into() }],
    };
    Json(resp)
}

#[derive(serde::Deserialize)]
struct SetModel { path: String }

async fn admin_set_model(State(state): State<AppState>, Json(req): Json<SetModel>) -> impl IntoResponse {
    state.model_path.write().await.replace(req.path);
    ([("content-type", "text/plain")], "ok")
}

fn tenant_id() -> String {
    // For now, a single-tenant placeholder. Extend with headers/ip as needed.
    "default".into()
}

use std::collections::HashMap;
use tokio::sync::Mutex as AsyncMutex;

#[derive(Clone)]
struct RateLimiter { inner: Arc<AsyncMutex<HashMap<String, (u64, std::time::Instant)>>> }
impl RateLimiter {
    fn new() -> Self { Self { inner: Arc::new(AsyncMutex::new(HashMap::new())) } }
    async fn check_allow(&self, key: &str) -> bool {
        let mut g = self.inner.lock().await;
        let entry = g.entry(key.to_string()).or_insert((0, std::time::Instant::now()));
        if entry.1.elapsed() > std::time::Duration::from_secs(60) { *entry = (0, std::time::Instant::now()); }
        let limit: u64 = std::env::var("RUNNER_RATE_LIMIT_PER_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
        if entry.0 >= limit { return false; }
        entry.0 += 1; true
    }
}

#[derive(Clone)]
struct TokenBudgets { inner: Arc<AsyncMutex<HashMap<String, u64>>> }
impl TokenBudgets {
    fn new() -> Self { Self { inner: Arc::new(AsyncMutex::new(HashMap::new())) } }
    async fn record(&self, key: &str, tokens: u64) {
        let mut g = self.inner.lock().await;
        let v = g.entry(key.to_string()).or_insert(0);
        *v += tokens;
    }
    async fn allowed(&self, key: &str, new_tokens: u64) -> bool {
        let budget: u64 = std::env::var("RUNNER_TOKEN_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
        let g = self.inner.lock().await;
        let used = *g.get(key).unwrap_or(&0);
        used + new_tokens <= budget
    }
}

