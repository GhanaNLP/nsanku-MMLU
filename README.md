# nsanku-MMLU

A multiple-choice benchmark for evaluating LLMs on Ghanaian language exams from the
National Teaching Council (NTC) licensing practice tests.

**Leaderboard:** [Hugging Face Spaces](https://huggingface.co/spaces/ghananlpcommunity/nsanku-mmlu-leaderboard)

## Dataset

| Language     | ISO 639-3 | Family | Questions |
|-------------|-----------|--------|-----------|
| Akuapem Twi | twi       | Kwa    | 200       |
| Asante Twi  | twi       | Kwa    | 200       |
| Dagaare     | dga       | Gur    | 200       |
| Dagbani     | dag       | Gur    | 200       |
| Dangme      | ada       | Kwa    | 200       |
| Ewe         | ewe       | Gbe    | 199       |
| Fante       | fat       | Kwa    | 100       |
| Ga          | gaa       | Kwa    | 100       |
| Gonja       | gjn       | Kwa    | 200       |
| Gurene      | gur       | Gur    | 200       |
| Kasem       | xsm       | Gur    | 196       |
| Nzema       | nzi       | Kwa    | 199       |
| **Total**   |           |        | **2,194** |

Source: [NTC Licensing Practice Tests](https://ntc.gov.gh/practice_test/ghanaian_languages/)

## Quick Start

```bash
pip install -r requirements.txt

# Dry run (no model needed)
python evaluate.py --dry-run

# Evaluate an API model
export OPENAI_API_KEY=sk-...
python evaluate.py --model openai/gpt-4o

# Evaluate a local model
python evaluate.py --model meta-llama/Llama-3-8B-Instruct --device cuda:0
```

## Supported Backends

### API Models

```bash
python evaluate.py --model openai/gpt-4o
python evaluate.py --model anthropic/claude-sonnet-4-20250514
python evaluate.py --model gemini/gemini-2.0-flash
```

### Local Models (HuggingFace Transformers)

```bash
pip install torch transformers accelerate
python evaluate.py --model meta-llama/Llama-3-8B-Instruct --device cuda:0
python evaluate.py --model mistralai/Mistral-7B-Instruct-v0.3 --dtype bfloat16
```

### Local Models (vLLM)

```bash
pip install vllm
python evaluate.py --model vllm/meta-llama/Llama-3-8B-Instruct
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

### Compare models

```bash
python leaderboard.py results/
python leaderboard.py results/ --languages ewe dagbani
python leaderboard.py results/ --format csv
```

## Directory Structure

```
nsanku-MMLU/
├── data/
│   ├── ghanaian_mmlu.jsonl          # full dataset (2,194 questions)
│   └── ghanaian_mmlu_<lang>.jsonl   # per-language splits
├── backends/
│   ├── base.py                      # abstract interface
│   ├── factory.py                   # creates backend from model spec
│   ├── transformers_backend.py      # HuggingFace local
│   ├── vllm_backend.py              # vLLM local
│   └── api_backends.py              # OpenAI, Claude, Gemini
├── hf-space/                        # static HTML leaderboard (reads results/ from GitHub)
├── raw/                             # raw extracted questions from NTC
├── evaluate.py                      # main evaluation script
├── leaderboard.py                   # compare results across models
├── prepare_dataset.py               # raw → MMLU JSONL
├── CONTRIBUTING.md                  # how to submit new benchmark results
└── requirements.txt
```

## Contributing

The most valuable contributions are **new benchmark results** — running the
evaluation on models not yet on the leaderboard (new hosted-API models, or your
own custom / fine-tuned models for the supported languages).

To keep every number comparable, please run the **unmodified** code with the
standardized configuration (thinking disabled for all models,
`--max-new-tokens 2048`, `temperature 0.0`).

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide.

## License

Data sourced from [NTC Practice Tests](https://ntc.gov.gh/practice_test/ghanaian_languages/).
