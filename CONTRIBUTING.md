# Contributing to the Ghanaian MMLU Benchmark

Thank you for helping broaden this benchmark! The most valuable contributions are
**new benchmark results** — running the evaluation on models that are not yet on
the leaderboard.

There are two kinds of result contributions:

1. **Hosted API models** not already benchmarked (e.g. a new OpenAI / Gemini /
   Anthropic / NVIDIA-Build model, or any OpenAI-compatible endpoint). This is
   how we widen coverage across the field.
2. **Custom or fine-tuned models** — models you have built or fine-tuned on one
   or more of the [12 supported languages](README.md). Run them locally through
   the `vllm/` or transformers backends and submit the results.

Code, documentation and dataset improvements are welcome too — open an issue or
PR as usual. The rest of this guide is about **result submissions**.

---

## Please run the *unmodified* code

For results to be comparable, please run the evaluation **without editing** the
scoring logic, the prompt, the label parser, the dataset, or the
thinking/decoding settings. The standardized methodology (all baked into the
code — please don't change it):

| Setting | Value |
|---|---|
| Prompt | `PROMPT_TEMPLATE` in `evaluate.py` (number first, ≤3 sentence explanations) |
| Thinking / reasoning | **disabled** for all models (`thinking_budget=0` for Gemini, `reasoning_effort='none'` for NVIDIA-Build, native for non-thinking models) |
| `--max-new-tokens` | `2048` |
| `--temperature` | `0.0` |
| Dataset | `data/ghanaian_mmlu.jsonl` (all 2,194 questions, all languages) |

> **Why no thinking?** Chain-of-thought support is inconsistent across models and
> providers (some can't disable it, some can't cap it), so enabling it would make
> scores non-comparable. Models that *cannot* run with thinking disabled — e.g.
> Gemini 3.x **Pro**, which rejects `thinking_budget=0` — cannot be added to this
> standardized leaderboard.

---

## How to run

Install dependencies and add whichever provider API key you need:

```bash
pip install -r requirements.txt
```

**Hosted API model** (example — a new model):

```bash
OPENAI_API_KEY=...   python evaluate.py --model openai/<model>       --max-new-tokens 2048
GEMINI_API_KEY=...   python evaluate.py --model gemini/<model>       --max-new-tokens 2048
NVIDIA_BUILD_API_KEY=... python evaluate.py --model nvidia/<org>/<model> --max-new-tokens 2048
```

**Custom / fine-tuned local model** (vLLM or transformers):

```bash
python evaluate.py --model vllm/<path-or-hf-id> --max-new-tokens 2048
python evaluate.py --model <hf-id>              --max-new-tokens 2048   # transformers
```

Each run writes to `results/<model_slug>/`:
`results.json`, `predictions.jsonl`, and `checkpoint.json`.

---

## Submitting

1. Fork the repo and create a branch.
2. Run the evaluation (unmodified).
3. Commit **only** the new `results/<model_slug>/` directory
   (`results.json` + `predictions.jsonl`).
4. Open a PR titled `Add benchmark: <provider>/<model>`. In the description note
   whether it's a hosted API model or a custom/fine-tuned model, and for custom
   models say what data it was trained/fine-tuned on.

The leaderboard is regenerated from the `results/` directories.
