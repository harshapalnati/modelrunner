# Quickstart

## Build & Run (CPU-only)

```bash
cd runner
cargo run -p runner-cli -- serve
# then in another shell
curl -X POST localhost:8080/generate -H "content-type: application/json" -d '{"prompt":"Hello"}'
```

## OpenAI-compatible (subset)

```bash
curl -X POST localhost:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Metrics

- GET /metrics Prometheus text format
- GET /healthz basic health check

## SSE Demo

- GET /sse/generate streams example tokens

## GPU builds (notes)

- CUDA: build llama.cpp with `GGML_CUDA=1`, set `LLAMA_CPP_DIR` to its lib folder
- Metal: build llama.cpp with `GGML_METAL=1`, set `LLAMA_CPP_DIR`

