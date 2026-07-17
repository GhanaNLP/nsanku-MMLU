#!/usr/bin/env python3
"""
Ghanaian MMLU Benchmark — Evaluate LLMs on Ghanaian language exams.

Usage:
    # API model (OpenAI)
    python evaluate.py --model openai/gpt-4o --languages dagaare dagbani

    # API model (Claude)
    python evaluate.py --model anthropic/claude-sonnet-4-20250514 --languages ewe

    # API model (Gemini)
    python evaluate.py --model gemini/gemini-2.0-flash --languages gonja

    # Local model (transformers)
    python evaluate.py --model meta-llama/Llama-3-8B-Instruct --device cuda:0

    # Local model (vLLM — batched, fast)
    python evaluate.py --model vllm/meta-llama/Llama-3-8B-Instruct

    # Dry run (no model, tests pipeline)
    python evaluate.py --dry-run

    # Resume from checkpoint
    python evaluate.py --model openai/gpt-4o --resume results/openai_gpt-4o/
"""

import argparse
import json
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# All prompts are concatenated into a single string (no system role) because
# many local models (Llama, Mistral, Gemma, Qwen fine-tunes) don't have a
# dedicated system token — they'd just see it as part of the user text anyway.

PROMPT_TEMPLATE = """\
Answer the following multiple-choice question in {language_name}.
Start your reply with a single number (1-{n_options}) — the number must come first.
If you add any explanation, keep it to no more than 3 sentences.

Example:
Question: Mɛnfa nea ɛyɛ akɛse paa no ka ho?
1. Ɔde
2. Ɛne
3. Ɛfa
Answer: 2

Question: {question}
{choices_block}
Answer:"""


def format_choices(entry: dict) -> str:
    lines = []
    for i, opt in enumerate(entry["options"], 1):
        lines.append(f"{i}. {opt}")
    return "\n".join(lines)


def build_prompt(entry: dict) -> str:
    n_options = len(entry["options"])
    return PROMPT_TEMPLATE.format(
        language_name=entry["language_name"],
        n_options=n_options,
        question=entry["question"],
        choices_block=format_choices(entry),
    )


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

import re


def parse_label(text: str, n_options: int = 4) -> Optional[str]:
    """
    Extract a single digit 1-n_options from model output.

    Priority:
      1. digit at start of response (after whitespace)
      2. any digit 1-n_options in the response
    """
    text = text.strip()

    # 1. digit at start
    m = re.match(rf"\s*([1-{n_options}])", text)
    if m:
        return m.group(1)

    # 2. any valid digit in the response
    m = re.search(rf"\b([1-{n_options}])\b", text)
    if m:
        return m.group(1)

    return None


def get_correct_answer(entry: dict) -> Optional[str]:
    """Get the correct answer number (1-based) for an entry."""
    answer = entry["answer"]
    options = entry["options"]
    if answer in options:
        return str(options.index(answer) + 1)
    return None


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint(results_dir: Path) -> dict:
    ckpt_path = results_dir / "checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return {"completed_ids": [], "predictions": []}


def save_checkpoint(results_dir: Path, ckpt: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_spec: str,
    data_path: str,
    languages: Optional[list[str]] = None,
    tests: Optional[list[str]] = None,
    output_dir: str = "results",
    batch_size: int = 1,
    max_questions: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
    **backend_kwargs,
):
    # Load dataset
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"ERROR: Dataset not found at {data_file}", file=sys.stderr)
        print("Run prepare_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Filter by language / test
    if languages:
        lang_set = {l.lower() for l in languages}
        entries = [e for e in entries if e["language"].lower() in lang_set]
    if tests:
        test_set = {t.lower() for t in tests}
        entries = [e for e in entries if e["test"].lower() in test_set]

    if max_questions:
        entries = entries[:max_questions]

    total = len(entries)
    print(f"\n{'='*60}")
    print(f"  Ghanaian MMLU Benchmark")
    print(f"  Model:   {model_spec}")
    print(f"  Questions: {total}")
    if languages:
        print(f"  Languages: {', '.join(languages)}")
    if tests:
        print(f"  Tests:    {', '.join(tests)}")
    print(f"{'='*60}\n")

    # Setup results directory
    model_slug = model_spec.replace("/", "__")
    results_dir = Path(output_dir) / model_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint for resumption
    ckpt = {"completed_ids": [], "predictions": []}
    if resume:
        ckpt = load_checkpoint(results_dir)
        print(f"Resuming: {len(ckpt['completed_ids'])} questions already done")

    completed_set = set(ckpt["completed_ids"])
    predictions = ckpt["predictions"]
    remaining = [e for e in entries if e["question_id"] not in completed_set]

    if dry_run:
        print("[DRY RUN] Testing pipeline without model inference...\n")
        for entry in remaining[:5]:
            correct_answer = get_correct_answer(entry)
            prompt = build_prompt(entry)
            print(f"  {entry['question_id']}: correct={correct_answer}")
            print(f"    Q: {entry['question'][:80]}...")
            print(f"    Prompt length: {len(prompt)} chars")
            print()
        print(f"[DRY RUN] Would evaluate {len(remaining)} questions total.")
        return

    # Create backend and run
    from backends import create_backend

    backend = create_backend(model_spec, **backend_kwargs)
    backend.load()

    eval_start = time.time()
    n_correct = sum(1 for p in predictions if p.get("correct"))

    try:
        for i, entry in enumerate(remaining):
            prompt = build_prompt(entry)
            correct_answer = get_correct_answer(entry)
            n_options = len(entry["options"])

            response = backend.predict(prompt)
            predicted = parse_label(response.raw_text, n_options)
            is_correct = predicted == correct_answer

            n_correct += 1 if is_correct else 0

            pred_record = {
                "question_id": entry["question_id"],
                "language": entry["language"],
                "iso639_3": entry.get("iso639_3"),
                "test": entry["test"],
                "question": entry["question"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted,
                "correct": is_correct,
                "model_raw": response.raw_text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_s": response.latency_s,
            }
            predictions.append(pred_record)
            ckpt["completed_ids"].append(entry["question_id"])

            done = len(predictions)
            acc_so_far = n_correct / done * 100
            sys.stdout.write(
                f"\r  [{done}/{total}] accuracy={acc_so_far:.1f}%  "
                f"q={entry['question_id']}  pred={predicted}  gold={correct_answer}  "
                f"{'OK' if is_correct else 'WRONG'}   "
            )
            sys.stdout.flush()

            # Save checkpoint every 10 questions
            if done % 10 == 0:
                ckpt["predictions"] = predictions
                save_checkpoint(results_dir, ckpt)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving checkpoint...")
    finally:
        ckpt["predictions"] = predictions
        save_checkpoint(results_dir, ckpt)
        backend.unload()

    elapsed = time.time() - eval_start
    print(f"\n\n{'='*60}")

    # -----------------------------------------------------------------------
    # Compute and report scores
    # -----------------------------------------------------------------------
    results = compute_results(predictions, total)
    results["model"] = model_spec
    results["timestamp"] = datetime.now().isoformat()
    results["elapsed_seconds"] = elapsed

    # Save full results
    results_path = results_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save predictions as JSONL
    preds_path = results_dir / "predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print_results(results)
    print(f"\n  Results saved to: {results_dir}/")

    return results


def compute_results(predictions: list[dict], total: int) -> dict:
    n_correct = sum(1 for p in predictions if p["correct"])
    overall_acc = n_correct / total * 100 if total > 0 else 0

    # Per-language breakdown
    lang_stats = {}
    for p in predictions:
        lang = p["language"]
        if lang not in lang_stats:
            lang_stats[lang] = {"correct": 0, "total": 0, "by_test": {}}
        lang_stats[lang]["total"] += 1
        lang_stats[lang]["correct"] += 1 if p["correct"] else 0

        test = p["test"]
        if test not in lang_stats[lang]["by_test"]:
            lang_stats[lang]["by_test"][test] = {"correct": 0, "total": 0}
        lang_stats[lang]["by_test"][test]["total"] += 1
        lang_stats[lang]["by_test"][test]["correct"] += 1 if p["correct"] else 0

    for lang in lang_stats:
        s = lang_stats[lang]
        s["accuracy"] = round(s["correct"] / s["total"] * 100, 2) if s["total"] else 0
        for test in s["by_test"]:
            t = s["by_test"][test]
            t["accuracy"] = round(t["correct"] / t["total"] * 100, 2) if t["total"] else 0

    # Unparseable predictions
    n_unparsed = sum(1 for p in predictions if p["predicted_answer"] is None)

    return {
        "overall": {
            "total": total,
            "correct": n_correct,
            "accuracy": round(overall_acc, 2),
            "unparsed": n_unparsed,
        },
        "per_language": lang_stats,
        "predictions": predictions,
    }


def print_results(results: dict) -> None:
    overall = results["overall"]
    print(f"  OVERALL ACCURACY: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    if overall["unparsed"] > 0:
        print(f"  (Unparsed responses: {overall['unparsed']})")
    print()

    print(f"  {'Language':<12} {'Acc %':>8} {'Correct':>8} {'Total':>8}")
    print(f"  {'-'*40}")
    for lang, stats in sorted(results["per_language"].items()):
        print(
            f"  {lang:<12} {stats['accuracy']:>7.2f}% {stats['correct']:>8} {stats['total']:>8}"
        )
        for test, ts in sorted(stats["by_test"].items()):
            print(
                f"    {test:<10} {ts['accuracy']:>7.2f}% {ts['correct']:>8} {ts['total']:>8}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ghanaian MMLU Benchmark Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model spec. Prefix with provider: openai/, anthropic/, gemini/, vllm/. "
             "No prefix = HuggingFace transformers. e.g. openai/gpt-4o",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/ghanaian_mmlu.jsonl",
        help="Path to dataset JSONL (default: data/ghanaian_mmlu.jsonl)",
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=None,
        help="Filter to specific languages (e.g. dagaare dagbani ewe)",
    )
    parser.add_argument(
        "--tests", "-t",
        nargs="+",
        default=None,
        help="Filter to specific tests (e.g. past1 past2)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test pipeline without model inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for local models (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for local models (default: auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature (default: 0.0)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (alternatively set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY env vars)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )

    args = parser.parse_args()

    if not args.model and not args.dry_run:
        parser.error("--model is required (or use --dry-run)")

    evaluate(
        model_spec=args.model or "dry-run",
        data_path=args.data,
        languages=args.languages,
        tests=args.tests,
        output_dir=args.output,
        max_questions=args.max_questions,
        resume=args.resume,
        dry_run=args.dry_run,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
