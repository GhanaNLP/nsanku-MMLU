"""
Convert raw NTC extracted questions into MMLU-format JSONL dataset.

Each line:
{
    "question_id": "dagaare_past1_0",
    "language": "dagaare",
    "test": "past1",
    "question": "...",
    "choices": ["A", "B", "C", "D"],
    "options": ["opt1", "opt2", "opt3", "opt4"],
    "answer": "C",
    "source": "https://ntc.gov.gh/practice_test/ghanaian_languages/dagaare/past1/"
}
"""

import json
import re
import sys
from pathlib import Path

RAW_PATH = Path(__file__).parent / "raw" / "all_questions.json"
OUT_DIR = Path(__file__).parent / "data"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

LANG_DATA = {
    # name, iso639_3, speakers, family, region
    "akuapem_twi": ("Akuapem Twi", "twi", 2830000, "Kwa", "Eastern/Ashanti"),
    "asante_twi":  ("Asante Twi",  "twi", 7120000, "Kwa", "Ashanti"),
    "dagaare":     ("Dagaare",     "dga", 924000,  "Gur", "Upper West"),
    "dagbani":     ("Dagbani",     "dag", 3160000, "Gur", "Northern"),
    "dangme":      ("Dangme",      "ada", 1020000, "Kwa", "Greater Accra"),
    "ewe":         ("Ewe",         "ewe", 3820000, "Gbe", "Volta"),
    "fanti":       ("Fante",       "fat", 1170000, "Kwa", "Central"),
    "ga":          ("Ga",          "gaa", 745000,  "Kwa", "Greater Accra"),
    "gonja":       ("Gonja",       "gjn", 310000,  "Kwa", "Northern/Savannah"),
    "gurene":      ("Gurene",      "gur", 600000,  "Gur", "Upper East"),
    "kasem":       ("Kasem",       "xsm", 250000,  "Gur", "Upper East"),
    "nzema":       ("Nzema",       "nzi", 330000,  "Kwa", "Western"),
}
LANG_NAMES = {k: v[0] for k, v in LANG_DATA.items()}


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def normalize_answer(answer: str) -> str:
    return answer.strip()


def build_entry(key: str, idx: int, q: dict, raw_path: str) -> dict:
    lang, test = key.rsplit("_", 1)
    question_text = strip_html(q.get("question") or q.get("questio") or "")
    options = [strip_html(o) for o in (q.get("options") or q.get("option") or [])]
    answer = normalize_answer(q.get("answer", ""))

    choices = CHOICE_LABELS[: len(options)]

    source_url = f"https://ntc.gov.gh/practice_test/ghanaian_languages/{lang}/{test}/"

    lang_data = LANG_DATA.get(lang, (lang, None, None, None, None))

    return {
        "question_id": f"{lang}_{test}_{idx:04d}",
        "language": lang,
        "language_name": lang_data[0],
        "iso639_3": lang_data[1],
        "language_family": lang_data[3],
        "test": test,
        "subject": f"NTC {lang_data[0]} Licensing Exam",
        "question": question_text,
        "choices": choices,
        "options": options,
        "answer": answer,
        "answer_index": choices[options.index(answer)] if answer in options else None,
        "source": source_url,
        "n_options": len(options),
    }


def main():
    raw_path = RAW_PATH
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run extraction first.", file=sys.stderr)
        sys.exit(1)

    with open(raw_path) as f:
        raw = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = []
    per_lang = {}

    for key, questions in sorted(raw.items()):
        entries = []
        for idx, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            if not (q.get("question") or q.get("questio")):
                continue
            entry = build_entry(key, idx, q, str(raw_path))
            if len(entry["options"]) < 2:
                continue
            entries.append(entry)
        all_entries.extend(entries)

        lang = key.rsplit("_", 1)[0]
        per_lang.setdefault(lang, []).extend(entries)

    full_path = OUT_DIR / "ghanaian_mmlu.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    for lang, entries in sorted(per_lang.items()):
        lang_path = OUT_DIR / f"ghanaian_mmlu_{lang}.jsonl"
        with open(lang_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Dataset prepared:")
    print(f"  Total questions: {len(all_entries)}")
    print(f"  Languages: {len(per_lang)}")
    for lang, entries in sorted(per_lang.items()):
        tests = {}
        for e in entries:
            tests.setdefault(e["test"], 0)
            tests[e["test"]] += 1
        test_str = ", ".join(f"{t}: {c}" for t, c in sorted(tests.items()))
        print(f"    {LANG_NAMES.get(lang, lang):12s} ({len(entries):3d} questions) — {test_str}")
    print(f"  Full dataset: {full_path}")
    print(f"  Per-language:  {OUT_DIR}/ghanaian_mmlu_<lang>.jsonl")


if __name__ == "__main__":
    main()
