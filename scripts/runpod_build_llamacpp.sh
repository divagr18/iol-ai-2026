#!/bin/bash
# Build llama.cpp from source on RunPod H100 (CUDA available)

set -e

echo "=== Building llama.cpp from source ==="

# 1. Clone llama.cpp
cd /workspace
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# 2. Build with CUDA support
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) llama-server

echo ""
echo "Build complete: /workspace/llama.cpp/build/bin/llama-server"
/workspace/llama.cpp/build/bin/llama-server --version
