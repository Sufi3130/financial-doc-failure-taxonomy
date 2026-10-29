"""Build the committed data file the figures read (evaluation/results/model_accuracy.json).

Separates the slow, run-dependent step (needs the run directories, which are
gitignored) from figure rendering (make_figures.py, needs only committed JSON).
To refresh after a new/full-dataset run: update the baseline run dirs in
compare_prompts.V1 (or re-run those baselines), then:

    python evaluation/figure_data.py      # writes evaluation/results/model_accuracy.json
    python evaluation/make_figures.py      # re-renders docs/figures/*

Numbers only (per-field normalised accuracy + Wilson CI, all-fields, schema-valid
rate), no receipt text.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.compare_prompts import MANIFEST, V1, cell  # noqa: E402

# registry name -> taxonomy display name (matches by_model_counts.json)
DISPLAY = {"phi3": "Phi-3", "qwen2.5-3b": "Qwen2.5-3B", "llama3.2-3b": "Llama-3.2-3B"}


def cell_accuracy(summary):
    """Pull the per-field + all-fields normalised accuracy and CIs out of an evaluate summary."""
    fields = {f: {"acc": summary["fields"][f]["normalized_acc"],
                  "ci": summary["fields"][f]["normalized_acc_ci95"]}
              for f in summary["fields"]}
    return {
        "fields": fields,
        "all_fields": {"acc": summary.get("all_fields_normalized_acc"),
                       "ci": summary.get("all_fields_normalized_acc_ci95")},
        "schema_valid_rate": summary["schema"]["schema_valid_rate"],
    }


def main():
    out = {"meta": {"engine": "paddleocr", "prompt": "v1", "shots": 2,
                    "n_per_cell": 50, "ci": "Wilson 95%",
                    "note": "normalised match %, 50-receipt subsets; baselines in compare_prompts.V1"},
           "SROIE": {}, "CORD": {}}
    for (model, dataset), run_dir in V1.items():
        summary, _, _ = cell(run_dir, MANIFEST[dataset])
        out[dataset][DISPLAY[model]] = cell_accuracy(summary)

    outpath = HERE / "results" / "model_accuracy.json"
    outpath.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {outpath.relative_to(REPO).as_posix()}")
    for ds in ("SROIE", "CORD"):
        for m, d in out[ds].items():
            af = d["all_fields"]
            print(f"  {ds:5} {m:14} all-fields {af['acc']:5.1f}  schema-valid {d['schema_valid_rate']:.0f}%")


if __name__ == "__main__":
    main()
