# Ghanaian MMLU Benchmark

A multiple-choice benchmark for evaluating LLMs on Ghanaian language exams from the
National Teaching Council (NTC) licensing practice tests.

## Dataset

| Language     | ISO 639-3 | Family | Questions | Tests         |
|-------------|-----------|--------|-----------|---------------|
| Akuapem Twi | twi       | Kwa    | 200       | past1, past2  |
| Asante Twi  | twi       | Kwa    | 200       | past1, past2  |
| Dagaare     | dga       | Gur    | 200       | past1, past2  |
| Dagbani     | dag       | Gur    | 200       | past1, past2  |
| Dangme      | ada       | Kwa    | 200       | past1, past2  |
| Ewe         | ewe       | Gbe    | 199       | past1, past2  |
| Fante       | fat       | Kwa    | 100       | past2 only    |
| Ga          | gaa       | Kwa    | 100       | past2 only    |
| Gonja       | gjn       | Kwa    | 200       | past1, past2  |
| Gurene      | gur       | Gur    | 200       | past1, past2  |
| Kasem       | xsm       | Gur    | 196       | past1, past2  |
| Nzema       | nzi       | Kwa    | 199       | past1, past2  |
| **Total**   |           |        | **2,194** |               |

Source: [NTC Licensing Practice Tests](https://ntc.gov.gh/practice_test/ghanaian_languages/)

## Quick Start

```bash
# 1. Prepare the dataset (already done if data/ exists)
python prepare_dataset.py

# 2. Dry run (no model needed)
python evaluate.py --dry-run

# 3. Evaluate an API model
export OPENAI_API_KEY=sk-...
python evaluate.py --model openai/gpt-4o

# 4. Evaluate a local model
pip install torch transformers accelerate
python evaluate.py --model meta-llama/Llama-3-8B-Instruct --device cuda:0
```

## Supported Backends

### API Models

```bash
# OpenAI
python evaluate.py --model openai/gpt-4o
python evaluate.py --model openai/gpt-4o-mini

# Anthropic Claude
python evaluate.py --model anthropic/claude-sonnet-4-20250514
python evaluate.py --model anthropic/claude-3-5-haiku-20241022

# Google Gemini
python evaluate.py --model gemini/gemini-2.0-flash
python evaluate.py --model gemini/gemini-1.5-pro
```

### Local Models (HuggingFace Transformers)

```bash
pip install torch transformers accelerate

python evaluate.py --model meta-llama/Llama-3-8B-Instruct --device cuda:0
python evaluate.py --model mistralai/Mistral-7B-Instruct-v0.3 --dtype bfloat16
python evaluate.py --model google/gemma-2-9b-it --device cuda:0
```

### Local Models (vLLM — fast batched inference)

```bash
pip install vllm

python evaluate.py --model vllm/meta-llama/Llama-3-8B-Instruct
python evaluate.py --model vllm/mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.85
```

## Options

```
--model, -m          Model spec (required unless --dry-run)
--data, -d           Dataset path (default: data/ghanaian_mmlu.jsonl)
--languages, -l      Filter languages (e.g. ewe dagbani)
--tests, -t          Filter tests (e.g. past1 past2)
--output, -o         Results dir (default: results/)
--max-questions      Limit question count
--resume             Resume from checkpoint
--dry-run            Test without model
--device             Device (default: auto)
--dtype              Data type (default: auto)
--max-new-tokens     Generation length (default: 256)
--temperature        Sampling temperature (default: 0.0)
--api-key            API key (or set env var)
```

## Results Format

Results are saved to `results/<model_slug>/`:

- `results.json` — aggregated scores
- `predictions.jsonl` — per-question predictions
- `checkpoint.json` — resumable checkpoint

### Compare models with leaderboard.py

```bash
python leaderboard.py results/
python leaderboard.py results/ --languages ewe dagbani
python leaderboard.py results/ --format csv
```

## Re-extracting from NTC Website

```bash
python extract_v2.py   # fetches all pages, outputs to raw/all_questions.json
python prepare_dataset.py  # converts to MMLU JSONL format
```

## Directory Structure

```
ghanaian_mmlu/
├── data/
│   ├── ghanaian_mmlu.jsonl          # full dataset
│   └── ghanaian_mmlu_<lang>.jsonl   # per-language splits
├── backends/
│   ├── base.py                      # abstract interface
│   ├── factory.py                   # creates backend from model spec
│   ├── transformers_backend.py      # HuggingFace local
│   ├── vllm_backend.py              # vLLM local
│   └── api_backends.py              # OpenAI, Claude, Gemini
├── evaluate.py                      # main evaluation script
├── leaderboard.py                   # compare results across models
├── prepare_dataset.py               # raw → MMLU JSONL
└── requirements.txt
```
