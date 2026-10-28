"""Build review_v1.csv = the 20 already-reviewed cases (review_v0) carried over,
plus 30 new cases to review, drawn across the three models.

The 30 new cases exclude anything already reviewed (by dataset+engine+receipt+field),
and are split into:
  - ~15 OCR misses (value_not_in_ocr), deduplicated across models (OCR is shared),
    to characterise the OCR sub-kinds;
  - ~15 LLM-stage failures, preferring Qwen/Llama (the v0 sample was all Phi-3),
    spread over categories.

Adds two columns for the root-cause pass: `ocr_subkind` (for OCR misses) and
`root_cause`. No label text is written to the CSV (same as v0). Also writes each
run's tag details locally (gitignored) so show_case.py can display any case.

    python evaluation/failure_taxonomy/make_review_v1.py
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from evaluation.failure_taxonomy.by_model import RUNS  # noqa: E402  (model, run_dir, manifest)
from evaluation.failure_taxonomy.tag import NON_FAILURES, tag_run, write_tags  # noqa: E402

V0 = HERE / "review_v0.csv"
V1 = HERE / "review_v1.csv"
SEED = 42
N_OCR, N_LLM = 15, 15
LLM_STAGE = {"wrong_entity", "wrong_total", "boundary_too_wide", "boundary_too_narrow", "partial_value",
             "copy_corruption", "wrong_date_format", "wrong_value", "fewshot_copy", "empty_field", "malformed_json"}
COLS = ["review_id", "status", "dataset", "engine", "model", "receipt_id", "field", "auto_primary",
        "auto_subtype", "auto_tags", "ocr_subkind_hint", "ocr_subkind", "root_cause",
        "reviewed_category", "note", "reviewer"]


def pool():
    """Every tagged instance across the grid, tagged with its model; writes tag detail per run."""
    out = []
    for model, run, manifest in RUNS:
        instances = tag_run(REPO / run, REPO / manifest)
        write_tags(Path(run).name, instances)  # local tags_detail.jsonl for show_case (gitignored)
        for i in instances:
            i["model"] = model
        out += instances
    return out


def spread(items, n, key, rng):
    """Round-robin over groups defined by key(item), for a diverse sample of n."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    order = list(groups)
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(groups[g] for g in order):
        for g in order:
            if groups[g] and len(picked) < n:
                picked.append(groups[g].pop())
    return picked


def ocr_hint(inst):
    return "empty_page" if "ocr_empty_page" in (inst.get("tags") or []) else ""


def load_v0():
    with open(V0, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="overwrite review_v1.csv even if it exists (DISCARDS any filled-in review)")
    args = parser.parse_args()
    if V1.exists() and not args.force:
        sys.exit(f"{V1.name} already exists — refusing to overwrite your review. Pass --force to redraw "
                 f"(this discards any verdicts you have entered).")

    rng = random.Random(SEED)
    v0 = load_v0()
    reviewed = {(r["dataset"], r["engine"], r["receipt_id"], r["field"]) for r in v0}

    instances = [i for i in pool() if i["primary"] not in NON_FAILURES]
    by_key = {(i["dataset"], i["engine"], i["receipt_id"], i["field"], i["model"]): i for i in instances}

    # bucket A: OCR misses, deduped across models (same OCR), not already reviewed
    ocr_seen, ocr_cases = set(), []
    for i in instances:
        if i["primary"] != "value_not_in_ocr":
            continue
        k = (i["dataset"], i["engine"], i["receipt_id"], i["field"])
        if k in reviewed or k in ocr_seen:
            continue
        ocr_seen.add(k)
        ocr_cases.append(i)
    pick_ocr = spread(ocr_cases, N_OCR, lambda i: (i["dataset"], i["engine"]), rng)

    # bucket B: LLM-stage failures, prefer Qwen/Llama, not reviewed, spread over category+model
    llm_cases = [i for i in instances if i["primary"] in LLM_STAGE
                 and (i["dataset"], i["engine"], i["receipt_id"], i["field"]) not in reviewed]
    llm_cases.sort(key=lambda i: i["model"] == "Phi-3")  # non-Phi-3 first
    pick_llm = spread(llm_cases, N_LLM, lambda i: (i["primary"], i["model"]), rng)

    new = pick_ocr + pick_llm

    rows = []
    for r in v0:  # carry the 20, map to the new columns
        key = (r["dataset"], r["engine"], r["receipt_id"], r["field"], "Phi-3")
        inst = by_key.get(key)
        rows.append({
            "review_id": r["review_id"], "status": "done", "dataset": r["dataset"], "engine": r["engine"],
            "model": "Phi-3", "receipt_id": r["receipt_id"], "field": r["field"],
            "auto_primary": r.get("auto_primary_final", ""), "auto_subtype": r.get("auto_subtype_final", ""),
            "auto_tags": ";".join(inst["tags"]) if inst else "",
            "ocr_subkind_hint": ocr_hint(inst) if inst and r.get("auto_primary_final") == "value_not_in_ocr" else "",
            "ocr_subkind": "", "root_cause": "",
            "reviewed_category": r.get("reviewed_category", ""), "note": r.get("note", ""),
            "reviewer": r.get("reviewer", ""),
        })
    for k, inst in enumerate(new, len(v0) + 1):
        rows.append({
            "review_id": f"R{k:02d}", "status": "todo", "dataset": inst["dataset"], "engine": inst["engine"],
            "model": inst["model"], "receipt_id": inst["receipt_id"], "field": inst["field"],
            "auto_primary": inst["primary"], "auto_subtype": inst["subtype"] or "",
            "auto_tags": ";".join(inst["tags"]), "ocr_subkind_hint": ocr_hint(inst),
            "ocr_subkind": "", "root_cause": "", "reviewed_category": "", "note": "", "reviewer": "",
        })

    with open(V1, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    todo = [r for r in rows if r["status"] == "todo"]
    print(f"review_v1.csv: {len(rows)} cases ({len(v0)} carried done, {len(todo)} to do)")
    from collections import Counter
    print("new OCR-miss by (dataset/engine):",
          dict(Counter(f"{r['dataset']}/{r['engine']}" for r in todo if r["auto_primary"] == "value_not_in_ocr")))
    print("new LLM-stage by (model):",
          dict(Counter(r["model"] for r in todo if r["auto_primary"] != "value_not_in_ocr")))
    print("new LLM-stage by category:",
          dict(Counter(r["auto_primary"] for r in todo if r["auto_primary"] != "value_not_in_ocr")))


if __name__ == "__main__":
    main()
