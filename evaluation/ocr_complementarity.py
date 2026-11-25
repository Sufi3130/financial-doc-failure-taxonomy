"""Quantify OCR-engine complementarity (issue #26).

The OCR stage sets a hard ceiling: if the gold value is not in the OCR text, no
model can extract it (`value_not_in_ocr`). PaddleOCR and Tesseract have
complementary weaknesses (PaddleOCR garbles stylised headers -> company/address;
Tesseract garbles numbers and collapses on CORD photos -> total). This script
measures, per field and per dataset, how many `value_not_in_ocr` misses each
single engine has, and how far two cheap strategies would push that ceiling
down:

  * field-aware routing  -- use, per field, whichever single engine misses less;
  * union (merge)        -- the value survives if it is in *either* engine's text.

It is a ceiling analysis on the cached OCR only: no model is run, so the numbers
are model-independent. Reads the full-test baseline runs (both engines).

    python evaluation/ocr_complementarity.py

Writes evaluation/results/ocr_complementarity.json and prints a table.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from evaluation.evaluate import FIELDS, load_last_rows, squash, value_in_ocr  # noqa: E402

# One full-test run per dataset; each carries both engines (OCR is cached and
# model-independent, so any baseline run gives the same OCR text).
RUNS = {"SROIE": "runs/pipeline/20261104-013736",
        "CORD": "runs/pipeline/20261105-012015"}
MANIFEST = {"SROIE": REPO / "evaluation/datasets/SROIE/full_test.jsonl",
            "CORD": REPO / "evaluation/datasets/CORD/full_test.jsonl"}
ENGINES = ("paddleocr", "tesseract")


def _gold_norm(field, record):
    gt = record["ground_truth"]
    if field in ("date", "total"):
        return gt["normalized"].get(field)
    g = gt["raw"].get(field)
    return squash(g) if g and str(g).strip() else None


def analyse(dataset):
    run_dir = REPO / RUNS[dataset]
    rows = load_last_rows(run_dir / "run.jsonl")  # {(engine, id): row}
    records = {json.loads(l)["id"]: json.loads(l)
               for l in open(MANIFEST[dataset], encoding="utf-8") if l.strip()}

    # per field: counts of receipts where the gold is annotated and missing from OCR
    fields = {}
    for field in FIELDS:
        n = 0
        miss = {e: 0 for e in ENGINES}
        miss_union = 0
        for rid, rec in records.items():
            g = _gold_norm(field, rec)
            if g is None:
                continue  # field not annotated on this receipt
            n += 1
            locale = rec.get("locale", "my")
            present = {}
            for e in ENGINES:
                row = rows.get((e, rid))
                text = ((row or {}).get("ocr") or {}).get("text", "")
                present[e] = value_in_ocr(field, g, text, locale)
                if not present[e]:
                    miss[e] += 1
            if not any(present.values()):
                miss_union += 1
        if n == 0:
            continue
        best_single = min(miss[e] for e in ENGINES)
        fields[field] = {
            "n": n,
            "miss": dict(miss),
            "miss_best_single": best_single,          # field-aware routing ceiling
            "miss_union": miss_union,                 # merge-both-texts ceiling
            "recovered_vs_best": best_single - miss_union,
            "recovered_vs_paddle": miss["paddleocr"] - miss_union,
        }
    return fields


def main():
    out = {}
    for dataset in ("SROIE", "CORD"):
        out[dataset] = analyse(dataset)

    (HERE / "results" / "ocr_complementarity.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")

    for dataset, fields in out.items():
        print(f"\n== {dataset}  (value_not_in_ocr counts; lower is better) ==")
        print(f"   {'field':9} {'n':>4} {'Paddle':>7} {'Tess':>7} "
              f"{'routing':>8} {'union':>7}  {'union gain vs best / vs paddle':>12}")
        tot = {"n": 0, "paddleocr": 0, "tesseract": 0, "best": 0, "union": 0}
        for field, d in fields.items():
            print(f"   {field:9} {d['n']:>4} {d['miss']['paddleocr']:>7} {d['miss']['tesseract']:>7} "
                  f"{d['miss_best_single']:>8} {d['miss_union']:>7}   "
                  f"-{d['recovered_vs_best']} / -{d['recovered_vs_paddle']}")
            tot["n"] += d["n"]; tot["paddleocr"] += d["miss"]["paddleocr"]
            tot["tesseract"] += d["miss"]["tesseract"]
            tot["best"] += d["miss_best_single"]; tot["union"] += d["miss_union"]
        print(f"   {'ALL':9} {tot['n']:>4} {tot['paddleocr']:>7} {tot['tesseract']:>7} "
              f"{tot['best']:>8} {tot['union']:>7}   "
              f"-{tot['best']-tot['union']} / -{tot['paddleocr']-tot['union']}")
    print("\nwritten: evaluation/results/ocr_complementarity.json")


if __name__ == "__main__":
    main()
