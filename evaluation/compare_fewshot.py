"""Zero-shot ablation: v1 prompt, 2-shot (baseline) vs 0-shot, across the grid.

Same OCR (PaddleOCR, cached), same models and subsets, same v1 prompt; only the
few-shot examples differ (2 -> 0). Tests whether the examples help or hurt, and
whether they drive few-shot copying (a documented failure: the model copying an
example's answer instead of reading the receipt).

    python evaluation/compare_fewshot.py   # after the zero-shot runs finish

Writes evaluation/results/fewshot_2_vs_0.json (numbers only, no receipt text).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.compare_prompts import ENGINE, MANIFEST, MODELS, V1, WATCH, cell, paired  # noqa: E402
from evaluation.failure_taxonomy.tag import tag_run  # noqa: E402


def fewshot_copy_count(run_dir, manifest, engine=ENGINE):
    return sum(1 for i in tag_run(Path(run_dir), manifest)
               if i["engine"] == engine and "fewshot_copy" in (i.get("tags") or []))


def zs_dir(model, dataset):
    return REPO / f"runs/pipeline/exp-zeroshot-{model}-{dataset}"


def main():
    out = {}
    for dataset in ("SROIE", "CORD"):
        for model in MODELS:
            zs = zs_dir(model, dataset)
            if not (zs / "run.jsonl").is_file():
                print(f"[skip] missing {zs.relative_to(REPO).as_posix()} (run the zero-shot experiment first)")
                continue
            base = V1[(model, dataset)]                 # v1, 2-shot
            manifest = MANIFEST[dataset]
            s2, c2, cat2 = cell(base, manifest)         # 2-shot
            s0, c0, cat0 = cell(zs, manifest)           # 0-shot
            fixed, broke = paired(c2, c0)               # 2-shot -> 0-shot flips
            out[f"{dataset}/{model}"] = {
                "all_fields": {"2shot": s2.get("all_fields_normalized_acc"),
                               "0shot": s0.get("all_fields_normalized_acc")},
                "fields": {f: {"2shot": s2["fields"][f]["normalized_acc"],
                               "0shot": s0["fields"][f]["normalized_acc"]} for f in s2["fields"]},
                "categories": {c: {"2shot": cat2.get(c, 0), "0shot": cat0.get(c, 0)} for c in WATCH
                               if cat2.get(c) or cat0.get(c)},
                "fewshot_copy": {"2shot": fewshot_copy_count(base, manifest),
                                 "0shot": fewshot_copy_count(zs, manifest)},
                "paired": {"twoshot_right_to_zeroshot_wrong": broke,
                           "zeroshot_fixed": fixed},
                "schema_valid_rate": {"2shot": s2["schema"]["schema_valid_rate"],
                                      "0shot": s0["schema"]["schema_valid_rate"]},
            }

    for key, r in out.items():
        print(f"\n== {key}")
        af = r["all_fields"]
        if af["2shot"] is not None:
            print(f"   all-fields  2shot {af['2shot']:5.1f}  ->  0shot {af['0shot']:5.1f}   "
                  f"({af['0shot'] - af['2shot']:+.1f})")
        for f, x in r["fields"].items():
            print(f"   {f:8}    2shot {x['2shot']:5.1f}  ->  0shot {x['0shot']:5.1f}   "
                  f"({x['0shot'] - x['2shot']:+.1f})")
        fc = r["fewshot_copy"]
        print(f"   fewshot_copy tag  2shot {fc['2shot']} -> 0shot {fc['0shot']}")
        p = r["paired"]
        print(f"   paired: 0shot fixed {p['zeroshot_fixed']}, 0shot broke {p['twoshot_right_to_zeroshot_wrong']}")

    outpath = HERE / "results" / "fewshot_2_vs_0.json"
    outpath.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {outpath.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
