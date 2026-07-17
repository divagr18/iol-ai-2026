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

# 2. Install dependencies
pip install -q -r requirements.txt

# 3. Download llama-server prebuilt binary (Linux CUDA)
LLAMA_DIR="/workspace/llama-bin"
mkdir -p "$LLAMA_DIR"

if [ ! -f "$LLAMA_DIR/llama-server" ]; then
    echo "Downloading llama.cpp Linux CUDA binaries..."
    # Try latest release first
    LATEST_URL=$(curl -s https://api.github.com/repos/ggerganov/llama.cpp/releases/latest | grep "browser_download_url.*llama-.*-bin-ubuntu.*-cuda.*x64.tar.gz" | head -1 | cut -d '"' -f 4)
    
    if [ -z "$LATEST_URL" ]; then
        # Fallback to a known good release
        LATEST_URL="https://github.com/ggerganov/llama.cpp/releases/download/b4614/llama-b4614-bin-ubuntu-22.04-cuda-x64.tar.gz"
    fi
    
    cd /tmp
    wget -q "$LATEST_URL" -O llama-cuda.tar.gz
    tar -xzf llama-cuda.tar.gz -C "$LLAMA_DIR" --strip-components=1 2>/dev/null || tar -xzf llama-cuda.tar.gz -C "$LLAMA_DIR"
    rm llama-cuda.tar.gz
    chmod +x "$LLAMA_DIR/llama-server"
    cd /workspace/iol
fi

echo "llama-server: $LLAMA_DIR/llama-server"

# 4. Download Gemma-4 12B GGUF (Q4_K_M, ~7.5GB)
MODEL_DIR="/workspace/iol/models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/gemma-4-12b-it-q4_k_m.gguf" ]; then
    echo "Downloading Gemma-4-12B Q4_K_M GGUF (~7.5GB)..."
    cd "$MODEL_DIR"
    
    # Primary source: unsloth
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
