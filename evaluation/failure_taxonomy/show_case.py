"""Show one manual-review case: label, prediction, OCR text, raw model output, and open the image.

    python evaluation/failure_taxonomy/show_case.py --list        # all cases and their current verdicts
    python evaluation/failure_taxonomy/show_case.py R05           # one case (opens the receipt image)
    python evaluation/failure_taxonomy/show_case.py R05 --no-image

Reads review_v0.csv, the run logs in runs/pipeline/ and the tag details in tags/
(all local; the run logs and tag details are not in git).
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
RUNS = {"SROIE": "20260930-063807", "CORD": "20260930-092942"}
REVIEW = HERE / "review_v0.csv"


def load_review():
    with open(REVIEW, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find(path, engine, receipt_id, field=None):
    """Last matching JSONL row (the run log may hold retries), or None."""
    found = None
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
    run = RUNS[case["dataset"]]
    row = find(REPO / "runs" / "pipeline" / run / "run.jsonl", case["engine"], case["receipt_id"])
    tag = find(HERE / "tags" / run / "tags_detail.jsonl", case["engine"], case["receipt_id"], case["field"])
    if row is None or tag is None:
        sys.exit("run log or tag details not found (they are local files; were the runs deleted?)")

    print("=" * 78)
    print(f"{case['review_id']}  {case['dataset']}  {case['engine']}  {case['receipt_id']}  field = {case['field']}")
    print("=" * 78)
    print(f"LABEL (gold)      : {tag['gold']!r}")
    print(f"MODEL (pred)      : {tag['pred']!r}")
    print(f"auto category     : {case['auto_primary_final']} {case['auto_subtype_final'] or ''}"
          f"  tags: {', '.join(tag['tags']) or '-'}")
    if "first_pass_category" in case:  # after the author review
        print(f"first pass        : {case['first_pass_category']}")
        print(f"reviewed          : {case['verdict_vs_first_pass']} -> {case['reviewed_category']} "
              f"({case['reviewed_stage']}), by {case['reviewer']}")
    else:
        print(f"my first-pass     : {case['verdict']} -> {case['reviewed_category']} ({case['reviewed_stage']})")
    print(f"note              : {case['note']}")

    wanted = {squash(tag["gold"]), squash(tag["pred"])} - {""}
    print("\n--- OCR text (reading order); '>>' marks rows related to the label or prediction")
    for line in row["ocr"]["text"].splitlines() or ["(empty: the OCR returned no text)"]:
        sq = squash(line)
        hit = sq and any(len(sq) >= 3 and (sq in w or w in sq) for w in wanted)
        print(f"{'>>' if hit else '  '} {line}")
    print("\n--- raw model output")
    print(row["llm"]["raw_output"])

    image = REPO / row["image"]
    print(f"\n--- image: {image}")
    if open_image and image.is_file() and hasattr(os, "startfile"):
        os.startfile(image)  # opens in the default Windows image viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_id", nargs="?", help="e.g. R05")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    args = parser.parse_args()
    cases = load_review()
    if args.list or not args.review_id:
        for c in cases:
            print(f"{c['review_id']}  {c['dataset']:5} {c['engine']:9} {c['receipt_id']:14} {c['field']:7} "
                  f"{c['reviewed_category']:42} reviewer: {c['reviewer'][:20]}")
        return
    case = next((c for c in cases if c["review_id"].upper() == args.review_id.upper()), None)
    if case is None:
        sys.exit(f"no case {args.review_id} in {REVIEW.name}")
    show(case, open_image=not args.no_image)


if __name__ == "__main__":
    main()
