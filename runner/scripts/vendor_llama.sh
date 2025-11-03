#!/usr/bin/env bash
set -euo pipefail

# Vendor llama.cpp as a git submodule and optionally build CPU library

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -d "$ROOT_DIR/third_party/llama.cpp" ]; then
  git submodule update --init --recursive third_party/llama.cpp
fi

echo "Vendored llama.cpp at third_party/llama.cpp"
echo "To build CPU library (example):"
cat <<'EOT'
  cd third_party/llama.cpp
  make -j
  # then set LLAMA_CPP_DIR to this directory when building runner
EOT

