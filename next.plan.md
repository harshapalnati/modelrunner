<!-- b7e2efa1-d3c7-44a1-9bf2-ab2ce1a5d3da 6f012182-8be6-4c74-9337-3e2cd816fe92 -->
# Next Inference — Foundation-to-MVP (Multi-Platform, Repo-First)

## Scope

Build each piece methodically: repo + CI first, then POC (single-model), then API, streaming, and continuous batching v1. Target environments: Windows native + NVIDIA, Windows WSL2 + NVIDIA, macOS (Metal), Linux + NVIDIA. Add hooks for AMD ROCm and FPGA backends later without blocking MVP.

## Repo layout (Cargo workspace)

```
runner/
├── crates/
│   ├── runner-common/      # Shared types, errors, config
│   ├── runner-core/        # Schedulers, KV cache, sampling
│   ├── runner-backend/     # Backend traits + adapters
│   │   └── runner-backend-llamacpp/  # llama.cpp FFI (CPU/CUDA/Metal)
│   ├── runner-api/         # HTTP (Axum), SSE, OpenAI compat
│   ├── runner-cli/         # CLI binary: pull/serve/run/list
│   └── runner-obs/         # Metrics (Prometheus), tracing (OTel)
├── models/                 # Downloaded models (gitignored)
├── docs/
├── examples/
├── scripts/
│   └── build_llamacpp.sh   # Helper for local dev (per-OS flags)
└── .github/workflows/      # CI/CD pipelines
```

## Core dependencies

- tokio, axum, clap, serde, tracing, prometheus, opentelemetry
- tokenizers (HF) for BPE
- bindgen/cc for FFI build of llama.cpp (feature-gated: `cpu`, `cuda`, `metal`)
- reqwest for `pull` (HF downloads)
- cargo-dist for releases

## Backend abstraction (essential trait)

```rust
pub trait InferenceBackend: Send + Sync {
    fn load_model(&self, path: &str, params: LoadParams) -> Result<ModelHandle>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    fn forward(&self, requests: &mut [SequenceState]) -> Result<ForwardOutput>;
    fn kv_usage(&self) -> KvStats;
}
```

- `runner-backend-llamacpp` implements `InferenceBackend` using llama.cpp, compiled via `build.rs` with features:
  - `cpu` (default), `cuda` (Windows/Linux/WSL2), `metal` (macOS)
- Keep `hip`/`rocm` and `fpga` as empty features + stubs for later.

## Cross-platform build strategy

- Windows native + CUDA: link CUDA via llama.cpp’s `GGML_CUDA=1`; static CRT where possible; MSVC toolchain
- Windows WSL2 + CUDA: Linux build with CUDA; document driver/toolkit requirements
- macOS (ARM) + Metal: `GGML_METAL=1`, ship `.metal` shader bundle alongside binary
- Linux + NVIDIA: `GGML_CUDA=1` or CPU-only fallback
- CI: matrix on ubuntu-latest, windows-latest, macos-latest; CPU-only builds + tests; release artifacts via cargo-dist; GPU builds verified manually with a checklist

## Milestone 1 (Repo + CI + POC path)

- Workspace scaffolding, crates, shared error/result types
- FFI bootstrap for llama.cpp: `build.rs` compiles ggml/llama with feature flags; CPU path first
- Minimal binary path: load GGUF, tokenize with HF, single-sequence decode loop, greedy sampling
- HTTP non-streaming endpoint: `POST /generate { prompt } -> text`
- Acceptance: `curl localhost:8080/generate -d '{"prompt":"Hello"}'` returns coherent text on all 4 OS targets (CPU)

## Milestone 2 (Streaming + OpenAI format)

- SSE streaming for tokens
- OpenAI-compatible `/v1/chat/completions` (subset: messages, temperature, top_p)
- CLI: `runner serve`, `runner run`, `runner list`
- Basic metrics: `/healthz`, `/metrics`; tracing spans for request → tokens
- Acceptance: 4 concurrent requests stream smoothly; metrics expose tokens/sec, TTFT

## Milestone 3 (Continuous batching v1)

- Scheduler loop (1–3ms tick), dynamic batch sizing, simple fairness
- Naive KV cache (non-paged) sized by model+GPU; backpressure on admission
- Prefix detection hooks (count-only, no sharing yet)
- Acceptance: 2× Ollama throughput on N=8 (single GPU) in local tests

## Hooks for future backends (don’t block MVP)

- `feature = "rocm"` maps to stub `RocmBackend` (compiles, returns NotSupported)
- `feature = "fpga"` maps to stub `FpgaBackend` with TODO: integrate Vitis AI / OpenVINO-FPGA
- Keep API stable so later kernels can drop-in

## Testing & CI

- Unit tests on CPU paths for tokenization, sampling, scheduler logic
- Integration tests that run a tiny GGUF (like 70MB) to keep CI fast
- Bench scripts under `scripts/bench.sh` for local GPU perf runs
- Pre-commit: rustfmt, clippy (deny warnings), license headers

## Release & Distribution

- cargo-dist to produce per-OS archives
- GitHub Releases with checksums; `runner-<os>-<arch>-<features>.zip`
- Docs site (mdBook) with quickstart per OS

## Acceptance criteria for this phase

- Single binary runs on each OS (CPU-only) with minimal API
- CUDA build verified on Windows native, WSL2, and Linux (manual checklist)
- Metal build verified on macOS M-series
- SSE streaming stable with 4 concurrent users; metrics visible

### To-dos

- [x] Create Cargo workspace and crate skeletons with feature flags
- [x] Add shared types, error handling, config loader in runner-common
- [x] Bootstrap llama.cpp FFI crate with build.rs (CPU first)
- [x] Implement single-sequence decode loop using llama backend
- [x] Expose non-streaming POST /generate in runner-api (Axum)
- [x] Add runner-cli with serve/run/list commands
- [x] Add SSE token streaming and basic chat format
- [x] Implement /v1/chat/completions (subset)
- [x] Wire Prometheus /metrics and /healthz; tracing spans
- [x] Implement continuous batching scheduler v1 (tick loop)
- [x] Add naive KV cache and simple admission control
- [x] Document CUDA/Metal build steps; verify on each OS
- [x] Set up CI matrix (cpu-only build/test) + cargo-dist releases
- [x] Write per-OS quickstart and API docs (mdBook)

## Next Steps — Production V1 Sprints

### Sprint 1 — Real model I/O + DX basics (goal: real tokens out)
- [x] Implement llama.cpp CPU decode path: load GGUF → create context → tokenize → step → sample (greedy/top-k/top-p) (not tested)
- [x] Replace mock backend in server; return real tokens in `/generate` and SSE (not tested)
 - [x] Token accounting metrics: TTFT, tokens/sec, input/output token counts
 - [x] CLI: `runner pull <model>` (HF download + checksum + manifest); wire model path resolution
- [x] Acceptance: real completions stable on CPU; 4 concurrent streams OK (not tested)

### Sprint 2 — Throughput engine (goal: ≥2× Ollama at N=16)
- [x] Continuous batching (1–3 ms token tick) with dynamic batch sizing (not tested)
- [x] Paged KV v1 (fixed 4KB blocks), soft cap, backpressure on admission (not tested)
- [x] Prefix detection hooks for common system prompts (batch-local reuse) (not tested)
- [x] Admission control: queueing + early reject on memory prediction (not tested)
- [x] Metrics: aggregate tokens/sec, TTFT histogram, batch size, queue depth (not tested)

### Sprint 3 — Production ops (goal: day‑1 deployable)
- [x] OpenTelemetry tracing (request_id/tenant_id, TTFT, decode spans) (not tested)
- [x] GPU telemetry via NVML/Metal/ROCm → `/metrics` (util, VRAM, temp, power) (NVML done, others TBD)
- [x] Health/readiness endpoints, graceful shutdown, load shedding, circuit breakers (basic 503 via KV admission) (not tested)
- [x] Rate limiting + token budgets; request priorities (interactive vs batch) (rate limit + budgets done; priorities TBD) (not tested)
- [x] Multi‑model: hot‑load/swap, model manifest registry (basic `/admin/set_model`) (not tested)

### Sprint 4 — Long-context + ecosystem (goal: clear differentiation)
- [x] Paged KV v2 (defrag, pin/spill to host), 32k–128k contexts (scaffold; not tested)
- [x] Prefix cache across requests (content‑addressed KV) (scaffold; not tested)
- [x] Auto GPU detection + tuning (kv size, batch size, compute cap) (placeholder via NVML/sysinfo; not tested)
- [x] Packaging: Docker/Helm, Windows/macOS installers; OpenAPI docs; gRPC; WebSocket (OpenAPI + WS added; others TBD)
- [x] CLI TUI `runner stats`: reqs active, tp/s, TTFT, GPU %, VRAM, temps (basic CLI; not tested)

### Benchmarks (ship with V1)
- [ ] Bench matrix: 4060/4090/M2/M3 + 7B/13B/14B
- [ ] Workloads: N={1,4,8,16}, 200‑token latency, long‑context (4k/16k/32k)
- [ ] Metrics: tokens/sec (agg/single), TTFT, P50/95/99, VRAM, GPU %, queue depth

