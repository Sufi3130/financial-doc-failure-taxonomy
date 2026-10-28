"""Grammar ablation: v1 prompt, grammar OFF (baseline) vs ON, across the grid.

Same OCR (PaddleOCR, cached), same models, subsets, prompt and 2-shot examples;
only constrained decoding differs. Tests issue #25's acceptance: malformed_json
drops to 0 and schema-valid rate goes to 100% without regressing field accuracy.

    python evaluation/compare_grammar.py   # after the grammar runs finish

Writes evaluation/results/grammar_on_vs_off.json (numbers only, no receipt text).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.compare_prompts import MANIFEST, MODELS, V1, cell, paired  # noqa: E402

# malformed_json in the taxonomy = these parse failure types.
MALFORMED = ("no_json", "invalid_json", "not_object", "truncated")


def malformed_count(run_dir, engine="paddleocr"):
    """Count receipts whose output had a malformed-JSON failure, for one engine."""
    rows = [json.loads(l) for l in (Path(run_dir) / "run.jsonl").open(encoding="utf-8")]
    n = 0
    for r in rows:
        if r.get("engine") != engine or r.get("error") or not r.get("outcome"):
            continue
        if any(f["type"] in MALFORMED for f in r["outcome"]["failures"]):
            n += 1
    return n


def gr_dir(model, dataset):
    return REPO / f"runs/pipeline/exp-grammar-{model}-{dataset}"


def main():
    out = {}
    for dataset in ("SROIE", "CORD"):
        for model in MODELS:
            gr = gr_dir(model, dataset)
            if not (gr / "run.jsonl").is_file():
                print(f"[skip] missing {gr.relative_to(REPO).as_posix()} (run the grammar experiment first)")
                continue
            base = V1[(model, dataset)]                 # grammar off
            manifest = MANIFEST[dataset]
            s0, c0, cat0 = cell(base, manifest)         # off
            s1, c1, cat1 = cell(gr, manifest)           # on
            fixed, broke = paired(c0, c1)               # off-right -> on-wrong etc.
            out[f"{dataset}/{model}"] = {
                "all_fields": {"off": s0.get("all_fields_normalized_acc"),
                               "on": s1.get("all_fields_normalized_acc")},
                "fields": {f: {"off": s0["fields"][f]["normalized_acc"],
                               "on": s1["fields"][f]["normalized_acc"]} for f in s0["fields"]},
                "malformed_json": {"off": malformed_count(base), "on": malformed_count(gr)},
                "schema_valid_rate": {"off": s0["schema"]["schema_valid_rate"],
                                      "on": s1["schema"]["schema_valid_rate"]},
                "paired": {"grammar_fixed": fixed, "grammar_broke": broke},
            }

    for key, r in out.items():
        print(f"\n== {key}")
        af = r["all_fields"]
        if af["off"] is not None:
            print(f"   all-fields  off {af['off']:5.1f}  ->  on {af['on']:5.1f}   ({af['on'] - af['off']:+.1f})")
        m = r["malformed_json"]
        print(f"   malformed_json  off {m['off']} -> on {m['on']}")
        sv = r["schema_valid_rate"]
        print(f"   schema_valid_rate  off {sv['off']:.1f} -> on {sv['on']:.1f}")
        p = r["paired"]
        print(f"   paired: grammar fixed {p['grammar_fixed']}, grammar broke {p['grammar_broke']}")

    outpath = HERE / "results" / "grammar_on_vs_off.json"
    outpath.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {outpath.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
