# IOL-AI 2026 Submission

## Current Status: Core System Built (D1)

- [x] Project scaffold (dirs, .gitignore, requirements.txt)
- [x] Scorer replicates competition metric (weighted EM x chrF geometric mean)
- [x] Harness downloads Linguini, formats as competition schema, runs script.py locally
- [x] Baseline script.py v0.1 (organizers' pattern, configurable)
- [x] Deterministic linguistic analyzers (freq tables, morpheme mining, alignment, paradigm matrix, number system)
- [x] Neuro-symbolic verifiers (count, number system, matching bijection, fill-blank consistency, translation heuristics)
- [x] Jury-track explanation generator
- [x] script.py v1.0 integrates analyzers + verifiers + repair loop + explanation
- [x] T4 rehearsal Dockerfile (mirrors eval sandbox)
- [x] H100 bake-off harness (batch model comparison)
- [x] Unit tests pass for all modules
- [x] Scorer validated: perfect=92.4, corrupted=64.2 (discriminates correctly)

## Quick Commands

### Local 4060 (llama.cpp SERVER — Recommended, Model Stays Hot in VRAM)
Uses `llama-server.exe` with **OpenAI-compatible** `/v1/chat/completions`. The server applies the model's built-in chat template automatically — works for Gemma, Qwen, Llama, etc. No manual prompt formatting.

**Recommended model:** Gemma-4-E4B (~4B params, instruction-tuned, no reasoning overhead, fast on 4060).

```bash
# Download Unsloth Gemma-4-E4B Q4_K_M GGUF (~2.5GB) via browser:
# https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF → click gemma-4-e4b-it-q4_k_m.gguf → download
# Save to D:\IOL\models\

# Run on 5 problems (server starts automatically, stays hot)
python local_4060_llamaserver.py --model models\gemma-4-e4b-it-q4_k_m.gguf --limit 5 --score

# Tune performance
python local_4060_llamaserver.py --model models\gemma-4-e4b-it-q4_k_m.gguf --limit 10 \
  --threads 4 --ngl 999 --ctx 2048 --batch 256

# Enable deterministic linguistic analyzers (test if it helps for a specific model)
python local_4060_llamaserver.py --model models\gemma-4-e4b-it-q4_k_m.gguf --limit 5 --score --use_analysis
```

**Qwen3.5 local test (has reasoning/thinking blocks, ~60s/problem on 4060):**
```bash
python local_4060_llamaserver.py --model models\qwen3.5-4b-q4_k_m.gguf --limit 5 --score --max_tokens 1024
```
The server automatically passes `--chat-template-kwargs '{"enable_thinking":false}'` to suppress Qwen3.5 reasoning.

### Local 4060 (llama.cpp CLI — Fallback, Spawns Per Problem)
Slower because it reloads the model every problem. Only use if server mode fails.
```bash
python local_4060_llamacpp.py --model models\gemma-4-e4b-it-q4_k_m.gguf --limit 5 --score --verbose
```

### Local 4060 (Transformers AWQ — Eval-Compatible)
Same code that runs in the T4 sandbox:
```bash
pip install autoawq autoawq-kernels pandas
# Download AWQ
huggingface-cli download cyankiwi/gemma-4-E4B-it-AWQ-4bit --local-dir models/gemma4-e4b-awq
# Test harness
MODEL_ID=./models/gemma4-e4b-awq QUANT=awq python -m src.harness --limit 5 --output data/run_output
```

### RunPod H100 (One-Command Setup)
```bash
# Paste this into your RunPod H100 terminal
git clone https://github.com/divagr18/iol-ai-2026.git /workspace/iol
cd /workspace/iol
bash scripts/runpod_h100_setup.sh
```

### RunPod T4 (Dress Rehearsal)
```bash
git clone https://github.com/divagr18/iol-ai-2026.git /workspace/iol
cd /workspace/iol
bash scripts/runpod_t4_rehearsal.sh
```

### Dev Commands
```bash
# Unit tests (no GPU needed)
python tests/test_pipeline.py

# Score a submission
python -m src.scorer submission.csv gold.csv --verbose

# Full bake-off (H100 only)
python -m src.bakeoff --config configs/bakeoff_models.json --limit 40
```

## Next Steps

### 1. Local 4060 — Prompt Iteration NOW
Use `local_4060_gguf.py` with Unsloth Q4_K_M for instant feedback on your 4060. Tune TASK_PROMPTS in `script.py`, test analyzer output, iterate fast. This is LOCAL ONLY — the eval sandbox doesn't support GGUF.

### 2. HF Account & Repo Setup
```bash
# Create public repo and upload weights + script
python scripts/setup_hf_repo.py \
  --repo_id <your-username>/iol-ai-2026-submission \
  --model_id cyankiwi/Qwen3.5-4B-AWQ-4bit \
  --token $HF_TOKEN
```

### 3. Smoke Submission (D1 Evening)
Submit your repo to the competition Space to validate the pipeline end-to-end. Uses 1/3 daily submissions.

### 4. T4 Dress Rehearsal (D1/D2)
Rent RunPod T4, run `scripts/runpod_t4_rehearsal.sh`. Verify `submission.csv` completeness and runtime < 25 min inside the exact eval Docker image.

### 5. H100 Bake-off (D1 Night)
Rent RunPod H100, run `scripts/runpod_h100_setup.sh`. Compare Qwen3.5-4B/9B AWQ vs Gemma-4-12B/26B on 40 Linguini problems. Pick primary + fallback.

### 6. Iterate & Submit Daily (D2–D8)
- Tune prompts, analyzers, verifiers
- Submit best variant daily (max 3/day)
- Pick 2 finals before July 26, 23:59 UTC

## Project Structure

```
D:\IOL\
├── script.py                 # Main submission (reads /tmp/data/test.csv)
├── src/
│   ├── scorer.py             # Local scoring metric
│   ├── harness.py            # Local test harness
│   ├── analyzers.py          # Linguistic feature extraction
│   ├── verifiers.py          # Neuro-symbolic validators
│   ├── explanation.py        # Jury-track explanation generator
│   └── bakeoff.py            # Batch model comparison
├── tests/
│   └── test_pipeline.py      # Unit tests (no GPU needed)
├── scripts/
│   └── setup_hf_repo.py      # HF repo + weight uploader
├── configs/
│   └── bakeoff_models.json   # Model candidates for bake-off
├── Dockerfile.t4             # T4 rehearsal environment
├── requirements.txt
└── README.md
```

## Important Notes

- **Eval sandbox:** No internet, T4 16GB, 30 min limit, fp16 only
- **Weights:** Must ship inside the public repo (load from `.`)
- **License:** Use redistributable models (Apache 2.0, MIT, or Gemma Terms)
- **Repo must stay public** through July 26, 2026
- **Daily quota:** 3 submissions/day, pick 2 for final private leaderboard

## License

Your submission code — whatever license you choose. Model weights subject to their original licenses."` (repo root).
