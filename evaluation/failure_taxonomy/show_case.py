"""Show one manual-review case: label, prediction, OCR text, raw model output, and open the image.

    python evaluation/failure_taxonomy/show_case.py --list        # all cases, status and verdicts
    python evaluation/failure_taxonomy/show_case.py --todo        # only the cases still to review
    python evaluation/failure_taxonomy/show_case.py R21           # one case (opens the receipt image)
    python evaluation/failure_taxonomy/show_case.py R21 --no-image

Reads review_v1.csv, the run logs in runs/pipeline/ and the tag details in tags/
(all local; the run logs and tag details are not in git). Run make_review_v1.py
first if the tag details are missing.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REVIEW = HERE / "review_v1.csv"

# (model, dataset, OCR engine) -> run directory. Phi-3 runs hold both engines.
RUN_FOR = {
    ("Phi-3", "SROIE", "paddleocr"): "20260930-063807",
    ("Phi-3", "SROIE", "tesseract"): "20260930-063807",
    ("Phi-3", "CORD", "paddleocr"): "20261027-183737",
    ("Phi-3", "CORD", "tesseract"): "20261027-183737",
    ("Qwen2.5-3B", "SROIE", "paddleocr"): "20261027-202036",
    ("Qwen2.5-3B", "CORD", "paddleocr"): "20261027-204256",
    ("Llama-3.2-3B", "SROIE", "paddleocr"): "20261027-204939",
    ("Llama-3.2-3B", "CORD", "paddleocr"): "20261027-211038",
    ("Qwen2.5-3B", "SROIE", "tesseract"): "20261028-100916",
    ("Qwen2.5-3B", "CORD", "tesseract"): "20261028-103251",
    ("Llama-3.2-3B", "SROIE", "tesseract"): "20261028-103706",
    ("Llama-3.2-3B", "CORD", "tesseract"): "20261028-105452",
}

OCR_SUBKINDS = ["char_substitution", "row_not_detected", "garbled_row", "empty_page", "label_format_diff"]


def load_review():
    with open(REVIEW, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find(path, engine, receipt_id, field=None):
    """Last matching JSONL row (the run log may hold retries), or None."""
    found = None
    if not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("engine") == engine and row.get("receipt_id") == receipt_id and \
                    (field is None or row.get("field") == field):
                found = row
    return found


def squash(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def show(case, open_image=True):
    run = RUN_FOR.get((case["model"], case["dataset"], case["engine"]))
    if run is None:
        sys.exit(f"no run mapped for model={case['model']} dataset={case['dataset']} engine={case['engine']}")
    row = find(REPO / "runs" / "pipeline" / run / "run.jsonl", case["engine"], case["receipt_id"])
    tag = find(HERE / "tags" / run / "tags_detail.jsonl", case["engine"], case["receipt_id"], case["field"])
    if row is None or tag is None:
        sys.exit("run log or tag details not found (local files; run make_review_v1.py to regenerate tags).")

    print("=" * 80)
    print(f"{case['review_id']}  [{case['status']}]  {case['dataset']}  {case['engine']}  "
          f"model={case['model']}  {case['receipt_id']}  field = {case['field']}")
    print("=" * 80)
    print(f"LABEL (gold)      : {tag['gold']!r}")
    print(f"MODEL (pred)      : {tag['pred']!r}")
    print(f"auto category     : {case['auto_primary']} {case['auto_subtype'] or ''}"
          f"  tags: {case['auto_tags'] or '-'}")
    if case["status"] == "done":
        print(f"reviewed (v0)     : {case['reviewed_category']}   note: {case['note']}  "
              f"[{case['reviewer']}]")
    else:
        print(f"TO REVIEW         : fill ocr_subkind (if OCR miss) + root_cause in review_v1.csv")
    if case["auto_primary"] == "value_not_in_ocr":
        print(f"ocr_subkind hint  : {case['ocr_subkind_hint'] or '(decide from the image/OCR below)'}")
        print(f"   choices: {', '.join(OCR_SUBKINDS)}")

    wanted = {squash(tag["gold"]), squash(tag["pred"])} - {""}
    print("\n--- OCR text (reading order); '>>' marks rows related to the label or prediction")
    for line in row["ocr"]["text"].splitlines() or ["(empty: the OCR returned no text)"]:
        sq = squash(line)
        hit = sq and any(len(sq) >= 3 and (sq in w or w in sq) for w in wanted)
        print(f"{'>>' if hit else '  '} {line}")
    print("\n--- raw model output")
    print((row.get("llm") or {}).get("raw_output"))

    image = REPO / row["image"]
    print(f"\n--- image: {image}")
    if open_image and image.is_file() and hasattr(os, "startfile"):
        os.startfile(image)  # opens in the default Windows image viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_id", nargs="?", help="e.g. R21")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--todo", action="store_true", help="list only the cases still to review")
    parser.add_argument("--no-image", action="store_true")
    args = parser.parse_args()
    cases = load_review()
    if args.list or args.todo or not args.review_id:
        for c in cases:
            if args.todo and c["status"] != "todo":
                continue
            mark = "   " if c["status"] == "done" else ">> "
            print(f"{mark}{c['review_id']}  [{c['status']:4}] {c['dataset']:5} {c['engine']:9} "
                  f"{c['model']:13} {c['receipt_id']:14} {c['field']:7} {c['auto_primary']}")
        return
    case = next((c for c in cases if c["review_id"].upper() == args.review_id.upper()), None)
    if case is None:
        sys.exit(f"no case {args.review_id} in {REVIEW.name}")
    show(case, open_image=not args.no_image)


if __name__ == "__main__":
    main()
