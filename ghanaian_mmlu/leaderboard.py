#!/usr/bin/env python3
"""
Compare results across multiple model runs and print a leaderboard.

Usage:
    python leaderboard.py results/
    python leaderboard.py results/ --languages dagaare ewe
    python leaderboard.py results/ --format csv
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    all_results = []
    for model_dir in sorted(results_dir.iterdir()):
        results_file = model_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                all_results.append(data)
    return all_results


def print_leaderboard(results: list[dict], languages: list[str] = None, fmt: str = "table"):
    if not results:
        print("No results found.")
        return

    # Sort by overall accuracy descending
    results.sort(key=lambda r: r["overall"]["accuracy"], reverse=True)

    if fmt == "csv":
        print("model,language,accuracy,total,correct")
        for r in results:
            model = r["model"]
            if languages:
                for lang in languages:
                    if lang in r.get("per_language", {}):
                        s = r["per_language"][lang]
                        print(f"{model},{lang},{s['accuracy']},{s['total']},{s['correct']}")
            else:
                for lang, s in sorted(r.get("per_language", {}).items()):
                    print(f"{model},{lang},{s['accuracy']},{s['total']},{s['correct']}")
        return

    # Table format
    langs = languages or sorted(
        set().union(*(r.get("per_language", {}).keys() for r in results))
    )

    header = f"{'Rank':<5} {'Model':<35} {'Overall':>10}"
    for lang in langs:
        header += f" {lang:>12}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        model = r["model"]
        if len(model) > 34:
            model = model[:31] + "..."
        acc = r["overall"]["accuracy"]
        line = f"{i:<5} {model:<35} {acc:>9.2f}%"
        for lang in langs:
            s = r.get("per_language", {}).get(lang)
            if s:
                line += f" {s['accuracy']:>11.2f}%"
            else:
                line += f" {'—':>12}"
        print(line)

    print(f"\n  Total models compared: {len(results)}")
    best = results[0]
    print(f"  Best overall: {best['model']} ({best['overall']['accuracy']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Ghanaian MMLU Leaderboard")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--languages", "-l", nargs="+", default=None)
    parser.add_argument("--format", "-f", choices=["table", "csv"], default="table")
    args = parser.parse_args()

    results = load_results(Path(args.results_dir))
    print_leaderboard(results, args.languages, args.format)


if __name__ == "__main__":
    main()
