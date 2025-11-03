Param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Join-Path -ChildPath ".." | Resolve-Path
$llama = Join-Path $root "third_party\llama.cpp"

if (-not (Test-Path $llama)) {
  git submodule update --init --recursive third_party/llama.cpp
}

Write-Host "Vendored llama.cpp at third_party/llama.cpp"
Write-Host "To build CPU library (example using make in MSYS/WSL), or use CMake on Windows."
Write-Host "Set LLAMA_CPP_DIR to the built library directory when building runner."

