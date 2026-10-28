"""Compare prompt v1 (baseline) vs v2 (disambiguation rules) across the grid.

Same OCR (PaddleOCR, cached), same models and subsets; only the prompt differs.
For each (model, dataset) cell it reports per-field normalised accuracy v1 vs v2,
the all-fields figure, the shift in the key failure categories, and a paired
count of receipts that flipped (v1 wrong -> v2 right, and regressions).

    python evaluation/compare_prompts.py

Writes evaluation/results/prompt_v1_vs_v2.json (numbers only, no receipt text).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.evaluate import evaluate  # noqa: E402
from evaluation.failure_taxonomy.tag import NON_FAILURES, tag_run  # noqa: E402

SROIE = REPO / "evaluation/datasets/SROIE/subset_test_50.jsonl"
CORD = REPO / "evaluation/datasets/CORD/subset_valtest_50.jsonl"
MODELS = ["phi3", "qwen2.5-3b", "llama3.2-3b"]
ENGINE = "paddleocr"
WATCH = ["wrong_entity", "wrong_total", "boundary_too_wide", "boundary_too_narrow",
         "partial_value", "malformed_json"]

# (model, dataset) -> v1 baseline run dir (PaddleOCR, 2-shot, prompt v1)
V1 = {
    ("phi3", "SROIE"): "runs/pipeline/20260930-063807",
    ("phi3", "CORD"): "runs/pipeline/20261027-183737",
    ("qwen2.5-3b", "SROIE"): "runs/pipeline/20261027-202036",
    ("qwen2.5-3b", "CORD"): "runs/pipeline/20261027-204256",
    ("llama3.2-3b", "SROIE"): "runs/pipeline/20261027-204939",
    ("llama3.2-3b", "CORD"): "runs/pipeline/20261027-211038",
}
MANIFEST = {"SROIE": SROIE, "CORD": CORD}


def v2_dir(model, dataset):
    return REPO / f"runs/pipeline/exp-v2-{model}-{dataset}"


def cell(run_dir, manifest, engine=ENGINE):
    """(summary, {receipt_id: all_correct}, {category: n}) for one engine.
    Some v1 runs hold both OCR engines, so we fix the engine (the v2 runs are
    PaddleOCR-only) to keep the comparison like-for-like."""
    run_dir = Path(run_dir)
    _, per_receipt, summaries = evaluate(run_dir, manifest)
    results = per_receipt[engine]
    correct = {r["receipt_id"]: (bool(r["fields"]) and all(x["normalized"] for x in r["fields"].values()))
               for r in results}
    cats = {}
    for i in tag_run(run_dir, manifest):
        if i["engine"] == engine and i["primary"] not in NON_FAILURES:
            cats[i["primary"]] = cats.get(i["primary"], 0) + 1
    return summaries[engine], correct, cats


def paired(c1, c2):
    """(v1_wrong_then_v2_right, v2_regressions) over receipts both runs scored."""
    shared = set(c1) & set(c2)
    fixed = sum(1 for r in shared if not c1[r] and c2[r])
    broke = sum(1 for r in shared if c1[r] and not c2[r])
    return fixed, broke


def main():
    out = {}
    for dataset in ("SROIE", "CORD"):
        for model in MODELS:
            v2 = v2_dir(model, dataset)
            if not (v2 / "run.jsonl").is_file():
                print(f"[skip] missing {v2.relative_to(REPO).as_posix()} (run the experiment first)")
                continue
            s1, c1, cat1 = cell(V1[(model, dataset)], MANIFEST[dataset])
            s2, c2, cat2 = cell(v2, MANIFEST[dataset])
            fixed, broke = paired(c1, c2)
            fields = {f: {"v1": s1["fields"][f]["normalized_acc"], "v2": s2["fields"][f]["normalized_acc"]}
                      for f in s1["fields"]}
            out[f"{dataset}/{model}"] = {
                "all_fields": {"v1": s1.get("all_fields_normalized_acc"),
                               "v2": s2.get("all_fields_normalized_acc")},
                "fields": fields,
                "categories": {c: {"v1": cat1.get(c, 0), "v2": cat2.get(c, 0)} for c in WATCH
                               if cat1.get(c) or cat2.get(c)},
                "paired": {"v1_wrong_to_v2_right": fixed, "v2_regressions": broke},
                "schema_valid_rate": {"v1": s1["schema"]["schema_valid_rate"],
                                      "v2": s2["schema"]["schema_valid_rate"]},
            }

    for key, r in out.items():
        print(f"\n== {key}")
        af = r["all_fields"]
        if af["v1"] is not None:
            print(f"   all-fields  v1 {af['v1']:5.1f}  ->  v2 {af['v2']:5.1f}   ({af['v2'] - af['v1']:+.1f})")
        for f, x in r["fields"].items():
            print(f"   {f:8}    v1 {x['v1']:5.1f}  ->  v2 {x['v2']:5.1f}   ({x['v2'] - x['v1']:+.1f})")
        for c, x in r["categories"].items():
            print(f"   [{c:20}] v1 {x['v1']:3} -> v2 {x['v2']:3}")
        p = r["paired"]
        print(f"   paired: {p['v1_wrong_to_v2_right']} fixed, {p['v2_regressions']} regressions")

    outpath = HERE / "results" / "prompt_v1_vs_v2.json"
    outpath.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {outpath.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
