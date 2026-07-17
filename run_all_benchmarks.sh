#!/bin/bash
# Run all benchmarks sequentially.
#
# Standardized methodology (for fair cross-model comparison):
#   * Thinking / chain-of-thought DISABLED on every model:
#       - gpt-4o        : not a thinking model (nothing to disable)
#       - Gemini 3 Flash: thinking_budget=0        (backend)
#       - NVIDIA models : reasoning_effort='none'  (backend)
#   * --max-new-tokens 2048 (headroom; the model never sees this as a target)
#   * Same prompt for all models (see PROMPT_TEMPLATE in evaluate.py)
#
# NOTE: Gemini 3.x Pro is intentionally EXCLUDED — it rejects thinking_budget=0
#       ("only works in thinking mode"), so it cannot be run no-thinking.
#
# API keys are read from the environment — DO NOT hardcode them here. Export
# whichever you need before running, e.g.:
#   export OPENAI_API_KEY=...
#   export GEMINI_API_KEY=...
#   export NVIDIA_BUILD_API_KEYS=key1,key2,...   # pool; rotates past 429s
cd "$(dirname "$0")"

MAX_TOKENS=2048

echo "=========================================="
echo "Starting benchmark suite at $(date)"
echo "=========================================="

# OpenAI gpt-4o
echo ""
echo ">>> [1/5] OpenAI gpt-4o"
python3 evaluate.py --model openai/gpt-4o --max-new-tokens $MAX_TOKENS 2>&1 | tee /tmp/bench_openai.log

# Gemini 3 Flash (thinking disabled)
echo ""
echo ">>> [2/5] Gemini 3 Flash"
python3 evaluate.py --model gemini/gemini-3-flash-preview --max-new-tokens $MAX_TOKENS 2>&1 | tee /tmp/bench_gemini.log

# DeepSeek v4 Flash (NVIDIA Build, thinking disabled)
echo ""
echo ">>> [3/5] DeepSeek v4 Flash (NVIDIA Build)"
python3 evaluate.py --model nvidia/deepseek-ai/deepseek-v4-flash --max-new-tokens $MAX_TOKENS 2>&1 | tee /tmp/bench_deepseek.log

# MiniMax M3 (NVIDIA Build, thinking disabled)
echo ""
echo ">>> [4/5] MiniMax M3 (NVIDIA Build)"
python3 evaluate.py --model nvidia/minimaxai/minimax-m3 --max-new-tokens $MAX_TOKENS 2>&1 | tee /tmp/bench_minimax.log

# GLM-5.2 (NVIDIA Build, thinking disabled)
echo ""
echo ">>> [5/5] GLM-5.2 (NVIDIA Build)"
python3 evaluate.py --model nvidia/z-ai/glm-5.2 --max-new-tokens $MAX_TOKENS 2>&1 | tee /tmp/bench_glm.log

echo ""
echo "=========================================="
echo "All benchmarks finished at $(date)"
echo "=========================================="
