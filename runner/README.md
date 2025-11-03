# Next Inference (skeleton)

- Workspace crates created
- CLI: `runner-cli` with `serve` starting an Axum server on :8080
- Backend traits defined; llama.cpp backend stub with feature flags

Build (CPU-only):

```bash
cargo build -p runner-cli
```

Run:

```bash
cargo run -p runner-cli -- serve
```

## llama.cpp FFI bootstrap (CPU prebuilt lib)

Set `LLAMA_CPP_DIR` to a directory containing a prebuilt llama library (e.g., `libllama.a` or platform equivalent). The build will enable FFI if set:

```bash
# Example
export LLAMA_CPP_DIR=/path/to/llama/lib
cargo build -p runner-backend-llamacpp --features cpu
```

If not set, the llama backend remains a stub (compiles, returns NotImplemented). CUDA/Metal wiring will be added in later milestones.

## CUDA (Linux/Windows) — manual notes (to be expanded)

- Build llama.cpp with `GGML_CUDA=1` and ensure `libllama` is produced
- Set `LLAMA_CPP_DIR` to the folder containing the built library
- Ensure CUDA toolkit and driver versions are compatible

## Metal (macOS) — manual notes

- Build llama.cpp with `GGML_METAL=1`; ensure `.metal` shaders are packaged
- Set `LLAMA_CPP_DIR` accordingly
- Codesign may be required for release bundles

## Vendoring llama.cpp (all systems)

You can vendor llama.cpp as a submodule and build it locally:

```bash
# POSIX
./runner/scripts/vendor_llama.sh
# Windows PowerShell
./runner/scripts/vendor_llama.ps1
```

Then build llama.cpp (CPU or with CUDA/Metal) inside `third_party/llama.cpp` and set:

```bash
export LLAMA_CPP_DIR=third_party/llama.cpp
# or on Windows PowerShell
$env:LLAMA_CPP_DIR="third_party/llama.cpp"
```

## Docker images

CPU build:

```bash
docker build -f runner/docker/Dockerfile.cpu -t next-runner:cpu .
docker run --rm -p 8080:8080 -e RUNNER_MODEL=/models/model.gguf -v $PWD/runner/models:/models next-runner:cpu
```

CUDA build (requires NVIDIA container runtime):

```bash
docker build -f runner/docker/Dockerfile.cuda -t next-runner:cuda .
docker run --rm --gpus all -p 8080:8080 -e RUNNER_MODEL=/models/model.gguf -v $PWD/runner/models:/models next-runner:cuda
```

Note: Metal (Apple) is not supported in Docker; ship native macOS binaries instead.

