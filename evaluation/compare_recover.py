"""Recovery ablation: recover OFF (baseline) vs ON, for the derailment-prone cell.

Same OCR (PaddleOCR, cached), same model, subset, prompt and 2-shot examples;
only the opt-in retry+all-null fallback differs (issue #33). Tests the acceptance:
Phi-3 SROIE malformed_json drops to 0 without regressing field accuracy or the
other cells.

    python evaluation/compare_recover.py   # after the --recover run(s) finish

Experiment runs are expected at runs/pipeline/exp-recover-<model>-<dataset>
(PaddleOCR, v1, 2-shot, seed 42, --recover). Writes
evaluation/results/recover_on_vs_off.json (numbers only, no receipt text).
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.compare_grammar import MALFORMED, malformed_count  # noqa: E402
from evaluation.compare_prompts import MANIFEST, MODELS, V1, cell, paired  # noqa: E402


def rec_dir(model, dataset):
    return REPO / f"runs/pipeline/exp-recover-{model}-{dataset}"


def recovery_methods(run_dir, engine="paddleocr"):
    """How the recovered receipts were resolved: {'retry': n, 'fallback_all_null': n}."""
    rows = [json.loads(l) for l in (Path(run_dir) / "run.jsonl").open(encoding="utf-8")]
    return dict(Counter(r["recovery"] for r in rows
                        if r.get("engine") == engine and r.get("recovery")))


def main():
    out = {}
    for dataset in ("SROIE", "CORD"):
        for model in MODELS:
            rec = rec_dir(model, dataset)
            if not (rec / "run.jsonl").is_file():
                continue
            base = V1[(model, dataset)]              # recover off
            manifest = MANIFEST[dataset]
            s0, c0, _ = cell(base, manifest)         # off
            s1, c1, _ = cell(rec, manifest)          # on
            fixed, broke = paired(c0, c1)
            out[f"{dataset}/{model}"] = {
                "all_fields": {"off": s0.get("all_fields_normalized_acc"),
                               "on": s1.get("all_fields_normalized_acc")},
                "malformed_json": {"off": malformed_count(base), "on": malformed_count(rec)},
                "schema_valid_rate": {"off": s0["schema"]["schema_valid_rate"],
                                      "on": s1["schema"]["schema_valid_rate"]},
                "recovery_methods": recovery_methods(rec),
                "paired": {"recover_fixed": fixed, "recover_broke": broke},
            }

    if not out:
        print("no exp-recover-* runs found; launch a --recover run first, e.g.:\n"
              "  python -m src.run --manifest evaluation/datasets/SROIE/subset_test_50.jsonl "
              "--engine paddleocr --model phi3 --prompt v1 --shots 2 --seed 42 --recover")
        return

    for key, r in out.items():
        print(f"\n== {key}")
        af = r["all_fields"]
        if af["off"] is not None:
            print(f"   all-fields  off {af['off']:5.1f}  ->  on {af['on']:5.1f}   ({af['on'] - af['off']:+.1f})")
        m = r["malformed_json"]
        print(f"   malformed_json  off {m['off']} -> on {m['on']}")
        sv = r["schema_valid_rate"]
        print(f"   schema_valid_rate  off {sv['off']:.1f} -> on {sv['on']:.1f}")
        print(f"   recovery methods: {r['recovery_methods'] or '-'}")
        p = r["paired"]
        print(f"   paired: recover fixed {p['recover_fixed']}, recover broke {p['recover_broke']}")

    outpath = HERE / "results" / "recover_on_vs_off.json"
    outpath.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {outpath.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
