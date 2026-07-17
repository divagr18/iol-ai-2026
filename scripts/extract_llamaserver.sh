#!/bin/bash
# Extract llama-server binary from H100 build into repo for eval sandbox

cd /workspace/iol

# Copy the built binary
if [ -f "/workspace/llama.cpp/build/bin/llama-server" ]; then
    cp /workspace/llama.cpp/build/bin/llama-server ./llama-server
    chmod +x ./llama-server
    echo "Copied llama-server binary"
else
    echo "ERROR: llama-server not found at /workspace/llama.cpp/build/bin/llama-server"
    echo "Run: bash scripts/runpod_h100_llamaserver_setup.sh"
    exit 1
fi

# Verify it works
./llama-server --version
