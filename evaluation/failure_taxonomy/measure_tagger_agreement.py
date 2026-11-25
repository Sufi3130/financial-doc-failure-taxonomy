"""Measure how well the auto-tagger (tag.py) agrees with the 50-case manual
review (review_v1.csv) on the *primary* failure category (issue #27).

Re-tags the baseline run grid, joins each reviewed case by
(dataset, engine, model, receipt_id, field), and compares the current tagger's
primary against the manual verdict. Also prints the old `auto_primary` column
(the tagger output at review time) as the before-baseline.

    python evaluation/failure_taxonomy/measure_tagger_agreement.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from evaluation.failure_taxonomy.by_model import RUNS  # (model, run_dir, manifest)  # noqa: E402
from evaluation.failure_taxonomy.tag import tag_run  # noqa: E402

REVIEW = HERE / "review_v1.csv"


def primary_of(reviewed_category: str) -> str:
    """The manual verdict's primary category (strip any :subtype)."""
    return (reviewed_category or "").split(":")[0].strip()


def retag_lookup() -> dict:
    """(dataset, engine, model, receipt_id, field) -> current primary category."""
    lookup = {}
    for model, run, manifest in RUNS:
        run_dir = REPO / run
        if not (run_dir / "run.jsonl").is_file():
            continue
        for i in tag_run(run_dir, REPO / manifest):
            lookup[(i["dataset"], i["engine"], model, i["receipt_id"], i["field"])] = i["primary"]
    return lookup


def main():
    rows = list(csv.DictReader(REVIEW.open(encoding="utf-8")))
    lookup = retag_lookup()

    scored = [r for r in rows if primary_of(r["reviewed_category"])]
    old_ok = new_ok = 0
    changed = []
    for r in scored:
        verdict = primary_of(r["reviewed_category"])
        old = r["auto_primary"].strip()
        key = (r["dataset"], r["engine"], r["model"], r["receipt_id"], r["field"])
        new = lookup.get(key, "<not found>")
        old_ok += (old == verdict)
        new_ok += (new == verdict)
        if old != new:
            changed.append((r["review_id"], r["field"], old, new, verdict))

    n = len(scored)
    print(f"reviewed cases with a verdict: {n}")
    print(f"old tagger agreement: {old_ok}/{n}  ({100*old_ok/n:.0f}%)")
    print(f"new tagger agreement: {new_ok}/{n}  ({100*new_ok/n:.0f}%)")
    print(f"net change: {new_ok - old_ok:+d}\n")
    print("cases whose tag changed (review_id, field: old -> new | verdict):")
    for rid, field, old, new, verdict in changed:
        mark = "OK " if new == verdict else ("(was ok)" if old == verdict else "   ")
        print(f"  {mark} {rid} {field:7} {old} -> {new}   | verdict: {verdict}")


if __name__ == "__main__":
    main()
