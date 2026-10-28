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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # receipts carry characters outside Windows cp1252

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
NON_FAILURES = {"normalised_only_mismatch", "hallucinated_unlabelled_field"}  # these still count as a pass
MODELS = ["Phi-3", "Qwen2.5-3B", "Llama-3.2-3B"]


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


def engine_data(model, dataset, engine, receipt_id, field):
    """Run row (+ tag detail if the field failed) for one engine. No tag = that engine passed."""
    run = RUN_FOR.get((model, dataset, engine))
    if run is None:
        return None
    row = find(REPO / "runs" / "pipeline" / run / "run.jsonl", engine, receipt_id)
    if row is None:
        return None
    tag = find(HERE / "tags" / run / "tags_detail.jsonl", engine, receipt_id, field)
    return {"run": run, "row": row, "tag": tag}


def print_engine(name, d, field, gold):
    row, tag = d["row"], d["tag"]
    pred = tag.get("pred") if tag else ((row.get("outcome") or {}).get("lenient") or {}).get(field)
    if tag is None:
        status = "PASS (exact match)"            # no tag entry = field was fully correct
    elif tag.get("primary") in NON_FAILURES:
        status = f"PASS ({tag.get('primary')})"  # correct after normalisation (formatting only)
    else:
        cat = tag.get("primary") + (f" {tag.get('subtype')}" if tag.get("subtype") else "")
        status = f"MISS [{cat}]"
    print(f"\n########## {name}  (run {d['run']}) ##########")
    print(f"pred  : {pred!r}   -> {status}")
    if tag and tag.get("tags"):
        print(f"tags  : {';'.join(tag['tags'])}")
    wanted = {squash(gold), squash(pred)} - {""}
    print("--- OCR text; '>>' marks rows related to gold/pred")
    for line in row["ocr"]["text"].splitlines() or ["(empty: OCR returned no text)"]:
        sq = squash(line)
        hit = sq and any(len(sq) >= 3 and (sq in w or w in sq) for w in wanted)
        print(f"{'>>' if hit else '  '} {line}")
    print("--- raw model output")
    print((row.get("llm") or {}).get("raw_output"))


def show_both(case, open_image=True):
    """Both OCR engines side by side for the same receipt/field, to attribute the failure."""
    model, dataset, field, rid = case["model"], case["dataset"], case["field"], case["receipt_id"]
    data = {e: engine_data(model, dataset, e, rid, field) for e in ("paddleocr", "tesseract")}
    have = [d for d in data.values() if d]
    if not have:
        sys.exit("no run data for either engine (check the RUN_FOR mapping).")
    gold = next((d["tag"]["gold"] for d in have if d["tag"]), None)
    print("=" * 80)
    print(f"{case['review_id']}  [{case['status']}]  {dataset}  model={model}  {rid}  "
          f"field = {field}   (case engine: {case['engine']})")
    print("=" * 80)
    print(f"LABEL (gold)      : {gold!r}")
    if case["auto_primary"] == "value_not_in_ocr":
        print(f"ocr_subkind choices: {', '.join(OCR_SUBKINDS)}")
    for e in ("paddleocr", "tesseract"):
        if data[e]:
            print_engine(e.upper(), data[e], field, gold)
        else:
            print(f"\n########## {e.upper()} ##########\n(no run mapped for this model+dataset)")
    image = REPO / have[0]["row"]["image"]
    print(f"\n--- image: {image}")
    if open_image and image.is_file() and hasattr(os, "startfile"):
        os.startfile(image)


def gold_for(case):
    """The gold value for this receipt/field, from whichever engine tagged it."""
    for e in ("paddleocr", "tesseract"):
        d = engine_data(case["model"], case["dataset"], e, case["receipt_id"], case["field"])
        if d and d["tag"]:
            return d["tag"]["gold"]
    return None


def show_models(case):
    """All three models x both engines on this receipt/field: is the failure shared or model-specific?"""
    dataset, rid, field = case["dataset"], case["receipt_id"], case["field"]
    gold = gold_for(case)
    print(f"\n=== all models x engines on {rid} / {field}   (gold: {gold!r}) ===")
    print("    (PASS/MISS is an approximate squash match - a review aid, not the official score)")
    for model in MODELS:
        for engine in ("paddleocr", "tesseract"):
            run = RUN_FOR.get((model, dataset, engine))
            if run is None:
                print(f"   {model:14} {engine:9} (no run mapped)")
                continue
            row = find(REPO / "runs" / "pipeline" / run / "run.jsonl", engine, rid)
            if row is None:
                print(f"   {model:14} {engine:9} (receipt not in run)")
                continue
            pred = ((row.get("outcome") or {}).get("lenient") or {}).get(field)
            mark = "PASS" if squash(pred) != "" and squash(pred) == squash(gold) else "MISS"
            print(f"   {model:14} {engine:9} {mark}  {pred!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_id", nargs="?", help="e.g. R21")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--todo", action="store_true", help="list only the cases still to review")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--single", action="store_true",
                        help="show only the case's own engine (default shows both OCR engines)")
    parser.add_argument("--models", action="store_true",
                        help="also show all three models x both engines on this field (shared vs model-specific)")
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
    (show if args.single else show_both)(case, open_image=not args.no_image)
    if args.models:
        show_models(case)


if __name__ == "__main__":
    main()
