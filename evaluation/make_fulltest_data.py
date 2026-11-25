"""Build the committed full-test result files the figures and dashboard read.

Full-test counterpart of figure_data.py + failure_taxonomy/by_model.py: it reads
the full-test baseline run directories (PaddleOCR + Tesseract, v1 prompt, 2-shot)
and writes two committed JSON summaries, so figure rendering and the Streamlit
dashboard never need the run directories themselves.

    python evaluation/make_fulltest_data.py

Outputs (numbers only, no receipt text):
  evaluation/results/model_accuracy_full.json          (PaddleOCR accuracy + CIs)
  evaluation/failure_taxonomy/by_model_counts_full.json (both engines, counts)

The run directories are gitignored (as in figure_data.py); update RUNS after a
new full-test run. SROIE n=347, CORD n=100.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.compare_prompts import cell  # (summary, correct, cats) per engine  # noqa: E402
from evaluation.figure_data import cell_accuracy  # noqa: E402
from evaluation.failure_taxonomy.tag import NON_FAILURES, tag_run  # noqa: E402

DISPLAY = {"phi3": "Phi-3", "qwen2.5-3b": "Qwen2.5-3B", "llama3.2-3b": "Llama-3.2-3B"}

# (model, dataset) -> full-test baseline run dir (--engine all, v1 prompt, 2-shot, seed 42)
RUNS = {
    ("phi3", "SROIE"): "runs/pipeline/20261104-013736",
    ("qwen2.5-3b", "SROIE"): "runs/pipeline/20261104-165458",
    ("llama3.2-3b", "SROIE"): "runs/pipeline/20261104-212642",
    ("phi3", "CORD"): "runs/pipeline/20261105-012015",
    ("qwen2.5-3b", "CORD"): "runs/pipeline/20261105-025355",
    ("llama3.2-3b", "CORD"): "runs/pipeline/20261105-031508",
}
MANIFEST = {"SROIE": REPO / "evaluation/datasets/SROIE/full_test.jsonl",
            "CORD": REPO / "evaluation/datasets/CORD/full_test.jsonl"}
N_PER_CELL = {"SROIE": 347, "CORD": 100}


def build_accuracy():
    """PaddleOCR per-field + all-fields normalised accuracy with CIs, per model."""
    out = {"meta": {"engine": "paddleocr", "prompt": "v1", "shots": 2,
                    "ci": "Wilson 95%", "n_per_cell": N_PER_CELL, "split": "full_test",
                    "note": "full test split; SROIE n=347, CORD n=100; "
                            "baseline PaddleOCR v1 2-shot"},
           "SROIE": {}, "CORD": {}}
    for (model, dataset), run_dir in RUNS.items():
        summary, _, _ = cell(run_dir, MANIFEST[dataset], "paddleocr")
        out[dataset][DISPLAY[model]] = cell_accuracy(summary)
    return out


def build_counts():
    """Failure-category counts per dataset/engine/model (both OCR engines)."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    totals = defaultdict(lambda: defaultdict(Counter))
    for (model, dataset), run_dir in RUNS.items():
        disp = DISPLAY[model]
        for inst in tag_run(REPO / run_dir, MANIFEST[dataset]):
            if inst["primary"] in NON_FAILURES:
                continue
            cells[dataset][inst["engine"]][disp][inst["primary"]] += 1
            totals[dataset][inst["engine"]][disp] += 1
    return {ds: {eng: {m: {"by_category": dict(cells[ds][eng][m]),
                           "total": totals[ds][eng][m]} for m in cells[ds][eng]}
                 for eng in cells[ds]} for ds in cells}


def main():
    acc = build_accuracy()
    (HERE / "results" / "model_accuracy_full.json").write_text(
        json.dumps(acc, indent=1) + "\n", encoding="utf-8", newline="\n")
    counts = build_counts()
    (HERE / "failure_taxonomy" / "by_model_counts_full.json").write_text(
        json.dumps(counts, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("written: evaluation/results/model_accuracy_full.json")
    print("written: evaluation/failure_taxonomy/by_model_counts_full.json")
    for ds in ("SROIE", "CORD"):
        for m, d in acc[ds].items():
            print(f"  {ds:5} {m:14} all-fields {d['all_fields']['acc']:5.1f}  "
                  f"schema {d['schema_valid_rate']:.0f}%")


if __name__ == "__main__":
    main()
