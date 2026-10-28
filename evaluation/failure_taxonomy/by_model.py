"""Cross-model failure taxonomy: tag every (LLM, OCR engine, dataset) run and
compare the failure-category counts per model. Builds on tag.py (same rules).

Grid: 3 LLMs (Phi-3, Qwen2.5-3B, Llama-3.2-3B) x 2 OCR engines x 2 datasets.
OCR is shared (one cache), so `value_not_in_ocr` cases are the same across the
three models; the LLM-stage categories are where the models differ.

Writes evaluation/failure_taxonomy/by_model_counts.json (category counts only,
no label text) and prints a per-engine/dataset table with the models as columns.

    python evaluation/failure_taxonomy/by_model.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from evaluation.failure_taxonomy.tag import NON_FAILURES, PRIMARY_ORDER, tag_run  # noqa: E402

SROIE = "evaluation/datasets/SROIE/subset_test_50.jsonl"
CORD = "evaluation/datasets/CORD/subset_valtest_50.jsonl"
MODELS = ["Phi-3", "Qwen2.5-3B", "Llama-3.2-3B"]

# (model, run_dir, manifest). Phi-3 runs contain both OCR engines; the Qwen/Llama
# runs are one engine each. OCR is shared via the cache, so the engines are comparable.
RUNS = [
    ("Phi-3", "runs/pipeline/20260930-063807", SROIE),          # paddleocr + tesseract
    ("Phi-3", "runs/pipeline/20261027-183737", CORD),           # paddleocr + tesseract (CORD examples)
    ("Qwen2.5-3B", "runs/pipeline/20261027-202036", SROIE),     # paddleocr
    ("Qwen2.5-3B", "runs/pipeline/20261027-204256", CORD),      # paddleocr
    ("Llama-3.2-3B", "runs/pipeline/20261027-204939", SROIE),   # paddleocr
    ("Llama-3.2-3B", "runs/pipeline/20261027-211038", CORD),    # paddleocr
    ("Qwen2.5-3B", "runs/pipeline/20261028-100916", SROIE),     # tesseract
    ("Qwen2.5-3B", "runs/pipeline/20261028-103251", CORD),      # tesseract
    ("Llama-3.2-3B", "runs/pipeline/20261028-103706", SROIE),   # tesseract
    ("Llama-3.2-3B", "runs/pipeline/20261028-105452", CORD),    # tesseract
]


def collect():
    """All failure/non-failure instances across the grid, each tagged with its model."""
    out = []
    for model, run, manifest in RUNS:
        run_dir = REPO / run
        if not (run_dir / "run.jsonl").is_file():
            print(f"[skip] missing {run}", file=sys.stderr)
            continue
        for inst in tag_run(run_dir, REPO / manifest):
            inst["model"] = model
            out.append(inst)
    return out


def counts_by_cell(instances):
    """{dataset: {engine: {model: {category: n}}}} for failures only, plus totals."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    totals = defaultdict(lambda: defaultdict(Counter))
    for i in instances:
        if i["primary"] in NON_FAILURES:
            continue
        cells[i["dataset"]][i["engine"]][i["model"]][i["primary"]] += 1
        totals[i["dataset"]][i["engine"]][i["model"]] += 1
    return cells, totals


def print_tables(cells, totals):
    for dataset in sorted(cells):
        for engine in sorted(cells[dataset]):
            per_model = cells[dataset][engine]
            cats = [c for c in PRIMARY_ORDER if any(c in per_model[m] for m in MODELS)]
            present = [m for m in MODELS if m in per_model]
            print(f"\n== {dataset}  /  {engine}  (failures per category)")
            print(f"   {'category':24} " + "".join(f"{m:>14}" for m in present))
            for cat in cats:
                print(f"   {cat:24} " + "".join(f"{per_model[m].get(cat, 0):>14}" for m in present))
            print(f"   {'TOTAL failures':24} " + "".join(f"{totals[dataset][engine][m]:>14}" for m in present))


def to_json(cells, totals):
    return {ds: {eng: {m: {"by_category": dict(cells[ds][eng][m]),
                           "total": totals[ds][eng][m]} for m in cells[ds][eng]}
                 for eng in cells[ds]} for ds in cells}


def main():
    instances = collect()
    cells, totals = counts_by_cell(instances)
    print_tables(cells, totals)
    out = HERE / "by_model_counts.json"
    out.write_text(json.dumps(to_json(cells, totals), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {out.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
