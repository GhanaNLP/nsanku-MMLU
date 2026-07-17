"""
Ghanaian MMLU Leaderboard — Gradio app for HuggingFace Spaces.

Reads benchmark results straight from the GitHub repo (the results/ directory)
and displays an interactive leaderboard sorted by accuracy. No separate upload
step: whatever is committed to GitHub is what the leaderboard shows.
"""

import json
import urllib.request

import gradio as gr
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
GITHUB_REPO = "GhanaNLP/nsanku-MMLU"
BRANCH = "master"
CONTENTS_API = f"https://api.github.com/repos/{GITHUB_REPO}/contents/results?ref={BRANCH}"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}"

# Clean display names for known model specs; anything else falls back to the
# last path component of the spec.
DISPLAY_NAMES = {
    "openai/gpt-4o": "GPT-4o",
    "gemini/gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "nvidia/deepseek-ai/deepseek-v4-flash": "DeepSeek v4 Flash",
    "nvidia/minimaxai/minimax-m3": "MiniMax M3",
    "nvidia/z-ai/glm-5.2": "GLM-5.2",
    "nvidia/nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B",
    "nvidia/abacusai/dracarys-llama-3.1-70b-instruct": "Dracarys Llama 3.1 70B",
    "mistral/mistral-large-latest": "Mistral Large",
}

LANG_ORDER = [
    "akuapem_twi", "asante_twi", "dagaare", "dagbani", "dangme", "ewe",
    "fanti", "ga", "gonja", "gurene", "kasem", "nzema",
]
LANG_DISPLAY = {
    "akuapem_twi": "Akuapem Twi", "asante_twi": "Asante Twi",
    "dagaare": "Dagaare", "dagbani": "Dagbani", "dangme": "Dangme",
    "ewe": "Ewe", "fanti": "Fante", "ga": "Ga", "gonja": "Gonja",
    "gurene": "Gurene", "kasem": "Kasem", "nzema": "Nzema",
}


def display_name(spec: str) -> str:
    return DISPLAY_NAMES.get(spec, spec.split("/")[-1])


# ── Data loading ────────────────────────────────────────────────────────
def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "ghanaian-mmlu-leaderboard"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def load_results() -> list[dict]:
    """Fetch every results/<model>/results.json committed to the GitHub repo."""
    results = []
    try:
        entries = _get_json(CONTENTS_API)
    except Exception as e:
        print(f"Could not list GitHub results dir: {e}")
        return []

    for entry in entries:
        if entry.get("type") != "dir":
            continue
        name = entry["name"]
        url = f"{RAW_BASE}/results/{name}/results.json"
        try:
            data = _get_json(url)
            data["_file"] = name
            results.append(data)
        except Exception as e:
            # Directories without a results.json (e.g. incomplete/dry-run) are skipped.
            print(f"Skipping {name}: {e}")

    return results


def build_overall_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in sorted(results, key=lambda x: x["overall"]["accuracy"], reverse=True):
        o = r["overall"]
        rows.append({
            "Rank": 0,
            "Model": display_name(r.get("model", "unknown")),
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
        row = {"Model": display_name(r.get("model", "unknown"))}
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
        return empty, empty_lang, f"_No results found in {GITHUB_REPO}_"

    overall = build_overall_table(results)
    lang_table = build_language_table(results)
    n_models = len(results)
    best = max(results, key=lambda x: x["overall"]["accuracy"])
    best_str = (
        f"**{n_models}** model(s) · "
        f"Best: **{display_name(best['model'])}** at **{best['overall']['accuracy']:.2f}%**"
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

        Benchmarking LLMs on **2,194** multiple-choice questions across
        **12 Ghanaian languages** from the [NTC licensing practice tests](https://ntc.gov.gh/practice_test/ghanaian_languages/).
        All models are evaluated with **thinking disabled** for fair comparison.

        **Languages:** Akuapem Twi · Asante Twi · Dagaare · Dagbani · Dangme · Ewe · Fante · Ga · Gonja · Gurene · Kasem · Nzema

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

        Results are read directly from the
        [GitHub repo](https://github.com/GhanaNLP/nsanku-MMLU). To add a model:

        ```bash
        # 1. Run the benchmark (unmodified)
        python evaluate.py --model <your-model> --max-new-tokens 2048

        # 2. Open a PR adding results/<model_slug>/
        ```

        See [CONTRIBUTING.md](https://github.com/GhanaNLP/nsanku-MMLU/blob/master/CONTRIBUTING.md).
        Once merged, hit **Refresh** and the model appears here.
        """
    )

    # Load on startup
    demo.load(fn=refresh, outputs=[overall_df, lang_df, status])


if __name__ == "__main__":
    demo.launch()
