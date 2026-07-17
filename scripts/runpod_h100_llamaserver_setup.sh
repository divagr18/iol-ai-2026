#!/bin/bash
# RunPod H100 llama-server Setup for IOL-AI 2026
# Paste this into your RunPod H100 terminal after starting a pod

set -e

echo "=== IOL-AI 2026 H100 llama-server Setup ==="

# 1. Clone/update repo
REPO_URL="${REPO_URL:-https://github.com/divagr18/iol-ai-2026}"
if [ ! -d "/workspace/iol" ]; then
    git clone "$REPO_URL" /workspace/iol
fi
cd /workspace/iol

# 2. Install dependencies (minimal — no torch/transformers/autoawq, llama.cpp handles inference)
pip install -q -r requirements_server.txt

# 3. Build llama-server from source (no prebuilt Linux CUDA binary available)
LLAMA_BIN="/workspace/llama.cpp/build/bin/llama-server"

if [ ! -f "$LLAMA_BIN" ]; then
    echo "llama-server not found. Building llama.cpp from source with CUDA..."
    
    if ! command -v cmake &> /dev/null; then
        echo "Installing build dependencies (cmake, build-essential)..."
        apt-get update -qq && apt-get install -y -qq build-essential cmake
    fi
    
    cd /workspace
    if [ ! -d "llama.cpp" ]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    fi
    cd llama.cpp
    git pull --depth 1 || true
    mkdir -p build && cd build
    cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc) llama-server
    cd /workspace/iol
fi

echo "llama-server: $LLAMA_BIN"

# 4. Download Gemma-4 12B GGUF (Q4_K_M, ~7.5GB)
MODEL_DIR="/workspace/iol/models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/gemma-4-12b-it-q4_k_m.gguf" ]; then
    echo "Downloading Gemma-4-12B Q4_K_M GGUF (~7.5GB)..."
    cd "$MODEL_DIR"
    
    wget -q --show-progress "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-q4_k_m.gguf" -O gemma-4-12b-it-q4_k_m.gguf || \
    wget -q --show-progress "https://huggingface.co/bartowski/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf" -O gemma-4-12b-it-q4_k_m.gguf || \
    echo "WARNING: Could not auto-download. Please download manually to $MODEL_DIR/"
    
    cd /workspace/iol
fi

if [ ! -f "$MODEL_DIR/gemma-4-12b-it-q4_k_m.gguf" ]; then
    echo "ERROR: Model file not found. Please download it manually:"
    echo "  wget -O $MODEL_DIR/gemma-4-12b-it-q4_k_m.gguf https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-q4_k_m.gguf"
    exit 1
fi

# 5. Run test
MODEL="$MODEL_DIR/gemma-4-12b-it-q4_k_m.gguf"
echo ""
echo "=== Setup complete ==="
echo "Model: $MODEL"
echo "VRAM needed: ~8GB for Q4_K_M + overhead"
echo "H100 VRAM: 80GB ✓"
echo ""
echo "Run this to test on 5 problems:"
echo "  cd /workspace/iol"
echo "  python runpod_h100_llamaserver.py --model $MODEL --limit 5 --score"
echo ""
echo "Run full bake-off with analyzers:"
echo "  cd /workspace/iol"
echo "  python runpod_h100_llamaserver.py --model $MODEL --limit 40 --score --use_analysis"
