# AGENTS.md — Instructions for Coding Agents

This file tells coding assistants (Claude Code, Cursor, Copilot, etc.) how to
run the Ghanaian MMLU benchmark on this repo.

## Quick Start

```bash
cd ghanaian_mmlu
python evaluate.py --dry-run          # verify pipeline works (no model needed)
python evaluate.py --model <MODEL>    # run actual evaluation
```

## Project Structure

```
ghanaian_mmlu/
├── data/
│   ├── ghanaian_mmlu.jsonl           # full dataset (2,194 questions)
│   └── ghanaian_mmlu_<lang>.jsonl    # per-language splits
├── backends/
│   ├── base.py                       # BaseModelBackend ABC
│   ├── factory.py                    # create_backend() from model spec string
│   ├── transformers_backend.py       # HuggingFace transformers (local)
│   ├── vllm_backend.py              # vLLM (local, batched)
│   └── api_backends.py              # OpenAI, Anthropic, Google Gemini
├── evaluate.py                       # main eval script — run this
├── prepare_dataset.py                # raw NTC HTML → JSONL (already run)
├── upload_results.py                 # push results.json to HF dataset repo
├── results/                          # local results (gitignored)
└── requirements.txt
```

## How to Run the Benchmark

### API Models

```bash
# OpenAI
OPENAI_API_KEY=sk-... python evaluate.py --model openai/gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... python evaluate.py --model anthropic/claude-sonnet-4-20250514

# Google Gemini
GOOGLE_API_KEY=... python evaluate.py --model gemini/gemini-2.0-flash

# Filter to specific languages
python evaluate.py --model openai/gpt-4o --languages ewe dagbani

# Limit question count (for testing)
python evaluate.py --model openai/gpt-4o --max-questions 10
```

### Local Models (transformers)

```bash
pip install torch transformers accelerate
python evaluate.py --model meta-llama/Llama-3-8B-Instruct --device cuda:0
python evaluate.py --model mistralai/Mistral-7B-Instruct-v0.3 --dtype bfloat16
```

### Local Models (vLLM — fast, batched)

```bash
pip install vllm
python evaluate.py --model vllm/meta-llama/Llama-3-8B-Instruct
```

### Resume Interrupted Runs

```bash
python evaluate.py --model openai/gpt-4o --resume
```

## How the Prompting Works

Each question is formatted as:

```
Answer the following multiple-choice question in {language}.
Reply with ONLY a single number (1-N) — nothing else.

Example:
Question: Mɛnfa nea ɛyɛ akɛse paa no ka ho?
1. Ɔde
2. Ɛne
3. Ɛfa
Answer: 2

Question: {actual question}
1. {option}
2. {option}
3. {option}
4. {option}
Answer:
```

The model's response is parsed for a digit 1-N. If found, it's scored.
If the model returns nothing parseable, it's counted as wrong.

## How to Contribute Results

```bash
# 1. Run the benchmark
python evaluate.py --model <your-model>

# 2. Results are saved to results/<model_slug>/results.json
ls results/

# 3. Upload to the leaderboard dataset
pip install huggingface_hub
HUGGINGFACE_TOKEN=hf_... python upload_results.py results/<model_slug>/results.json

# 4. Or submit a PR to the GitHub repo
git add results/
git commit -m "Add <model> results"
git push && gh pr create
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--model` | Model spec: `openai/gpt-4o`, `anthropic/claude-...`, `gemini/...`, `vllm/...`, or bare HF name |
| `--languages` | Filter: `--languages ewe dagbani nzema` |
| `--tests` | Filter: `--tests past1 past2` |
| `--max-questions` | Limit N questions (for quick tests) |
| `--resume` | Resume from checkpoint |
| `--dry-run` | Test pipeline without model |
| `--device` | Device for local models (default: auto) |
| `--dtype` | dtype for local models (default: auto) |

## Dataset Details

- **2,194** total questions across **12** Ghanaian languages
- Source: [NTC Licensing Practice Tests](https://ntc.gov.gh/practice_test/ghanaian_languages/)
- Format: JSONL with fields `question_id`, `language`, `question`, `options`, `answer`
- Most questions have 4 options; a few have 2, 3, or 5
- Languages: Akuapem Twi, Asante Twi, Dagaare, Dagbani, Dangme, Ewe, Fante, Ga, Gonja, Gurene, Kasem, Nzema
