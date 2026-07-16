"""
Ghanaian MMLU Leaderboard — Gradio app for HuggingFace Spaces.

Reads benchmark results from a HF dataset repo and displays
an interactive leaderboard sorted by accuracy.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────
RESULTS_REPO = "michsethowusu/ghanaian-mmlu-results"  # HF dataset repo
LANG_ORDER = [
    "dagaare", "dagbani", "dangme", "ewe", "fanti",
    "ga", "gonja", "gurene", "kasem", "nzema",
]
LANG_DISPLAY = {
    "dagaare": "Dagaare", "dagbani": "Dagbani", "dangme": "Dangme",
    "ewe": "Ewe", "fanti": "Fante", "ga": "Ga", "gonja": "Gonja",
    "gurene": "Gurene", "kasem": "Kasem", "nzema": "Nzema",
}


# ── Data loading ────────────────────────────────────────────────────────
def load_results() -> list[dict]:
    """Fetch all results.json files from the HF dataset repo."""
    api = HfApi()
    results = []

    try:
        files = api.list_repo_files(RESULTS_REPO, repo_type="dataset")
    except Exception as e:
        print(f"Could not list repo: {e}")
        return []

    for fname in files:
        if not fname.endswith("results.json"):
            continue
        try:
            path = hf_hub_download(
                repo_id=RESULTS_REPO,
                filename=fname,
                repo_type="dataset",
            )
            with open(path) as f:
                data = json.load(f)
                data["_file"] = fname
                results.append(data)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    return results


def build_overall_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in sorted(results, key=lambda x: x["overall"]["accuracy"], reverse=True):
        o = r["overall"]
        rows.append({
            "Rank": 0,
            "Model": r.get("model", "unknown"),
            "Accuracy (%)": o["accuracy"],
            "Correct": o["correct"],
            "Total": o["total"],
            "Unparsed": o.get("unparsed", 0),
        })
    df = pd.DataFrame(rows)
    df["Rank"] = range(1, len(df) + 1)
    return df


def build_language_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in sorted(results, key=lambda x: x["overall"]["accuracy"], reverse=True):
        row = {"Model": r.get("model", "unknown")}
        for lang in LANG_ORDER:
            s = r.get("per_language", {}).get(lang)
            if s and s["total"] > 0:
                row[LANG_DISPLAY[lang]] = f"{s['accuracy']:.1f}%"
            else:
                row[LANG_DISPLAY[lang]] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def refresh():
    results = load_results()
    if not results:
        empty = pd.DataFrame(columns=["Rank", "Model", "Accuracy (%)", "Correct", "Total"])
        empty_lang = pd.DataFrame(columns=["Model"] + [LANG_DISPLAY[l] for l in LANG_ORDER])
        return empty, empty_lang, f"_No results found in {RESULTS_REPO}_"

    overall = build_overall_table(results)
    lang_table = build_language_table(results)
    n_models = len(results)
    best = results[0] if results else None
    best_str = (
        f"**{n_models}** model(s) · "
        f"Best: **{best['model']}** at **{best['overall']['accuracy']:.2f}%**"
        if best else f"**{n_models}** model(s)"
    )
    return overall, lang_table, best_str


# ── Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="Ghanaian MMLU Leaderboard",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🇬🇭 Ghanaian MMLU Leaderboard

        Benchmarking LLMs on **1,795** multiple-choice questions across
        **10 Ghanaian languages** from the [NTC licensing practice tests](https://ntc.gov.gh/practice_test/ghanaian_languages/).

        **Languages:** Dagaare · Dagbani · Dangme · Ewe · Fante · Ga · Gonja · Gurene · Kasem · Nzema

        ---
        """
    )

    status = gr.Markdown("_Loading..._")

    with gr.Tabs():
        with gr.Tab("Overall"):
            overall_df = gr.Dataframe(
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Per-Language"):
            lang_df = gr.Dataframe(
                interactive=False,
                wrap=True,
            )

    refresh_btn = gr.Button("🔄 Refresh", variant="primary")
    refresh_btn.click(fn=refresh, outputs=[overall_df, lang_df, status])

    gr.Markdown(
        """
        ---

        ### Submit Your Results

        ```bash
        # 1. Run the benchmark
        python evaluate.py --model <your-model>

        # 2. Upload results.json to the results dataset
        #    or submit a PR to the GitHub repo
        ```
        """
    )

    # Load on startup
    demo.load(fn=refresh, outputs=[overall_df, lang_df, status])


if __name__ == "__main__":
    demo.launch()
