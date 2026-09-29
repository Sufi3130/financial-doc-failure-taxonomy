"""Build the SROIE Task 3 manifest shared by the pipeline and the evaluator.

Reads raw/{train,test}/<ID>.jpg + <ID>.txt (JSON with company, date, address,
total), validates the labels, flags duplicate images, normalises date and
total, and writes:

    manifest.jsonl          one record per receipt (gitignored: contains labels)
    subset_test_50.jsonl    seeded evaluation subset (gitignored: contains labels)
    subset_test_50_ids.txt  receipt IDs of that subset (committed)
    prepare_report.json     dataset stats and data-quality findings (committed)

Usage:
    python evaluation/datasets/SROIE/prepare.py [--seed 42] [--n 50]
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.normalize import normalize_date, normalize_total  # noqa: E402

RAW = HERE / "raw"
SPLITS = ("train", "test")
EXPECTED = {"train": 626, "test": 347}
FIELDS = ("company", "date", "address", "total")
NUMBERED_COPY = re.compile(r"^(?P<base>.+)\(\d+\)$")  # Windows "X...(1).jpg" duplicates


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def load_labels(path):
    """Return (labels dict or None, issues list)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, [f"bad_json: {e.__class__.__name__}"]
    if not isinstance(data, dict):
        return None, ["bad_json: not an object"]

    issues = []
    labels = {}
    for key in FIELDS:
        if key not in data:
            issues.append(f"missing:{key}")
            labels[key] = None
        else:
            labels[key] = str(data[key])
            if not labels[key].strip():
                issues.append(f"empty:{key}")
    extra = sorted(set(data) - set(FIELDS))
    if extra:
        issues.append(f"extra_keys:{','.join(extra)}")
    return labels, issues


def scan_split(split):
    """Pair jpg/txt by exact stem; return (records, split stats)."""
    folder = RAW / split
    if not folder.is_dir():
        sys.exit(f"[FAIL] missing folder: {folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file())
    by_ext = defaultdict(dict)
    numbered, other = [], []
    for p in files:
        ext = p.suffix.lower()
        if ext not in (".jpg", ".txt"):
            other.append(p.name)
        elif m := NUMBERED_COPY.match(p.stem):
            base = folder / f"{m['base']}{p.suffix}"
            identical = base.is_file() and base.read_bytes() == p.read_bytes()
            numbered.append((p.name, identical))
        else:
            by_ext[ext][p.stem] = p

    jpgs, txts = by_ext[".jpg"], by_ext[".txt"]
    stems = sorted(jpgs.keys() & txts.keys())

    records = []
    for stem in stems:
        labels, issues = load_labels(txts[stem])
        normalized = {"date": None, "total": None}
        if labels:
            for key, fn in (("date", normalize_date), ("total", normalize_total)):
                if labels[key] and labels[key].strip():
                    normalized[key] = fn(labels[key])
                    if normalized[key] is None:
                        issues.append(f"unparsed:{key}")
        records.append({
            "id": stem,
            "split": split,
            "image": rel(jpgs[stem]),
            "sha256": sha256(jpgs[stem]),
            "duplicate_of": None,
            "issues": issues,
            "ground_truth": {"raw": labels, "normalized": normalized},
        })

    if numbered:
        n_diff = sum(not same for _, same in numbered)
        print(f"  [warn] {split}: ignored {len(numbered)} Windows numbered copies "
              f"(e.g. {', '.join(n for n, _ in numbered[:3])}); "
              f"{len(numbered) - n_diff} identical to their base file, {n_diff} not")

    stats = {
        "pairs": len(stems),
        "numbered_copies_ignored": len(numbered),
        "numbered_copies_not_identical_to_base": sorted(n for n, same in numbered if not same),
        "jpg_without_txt": sorted(jpgs.keys() - txts.keys()),
        "txt_without_jpg": sorted(txts.keys() - jpgs.keys()),
        "other_files": other,
    }
    return records, stats


def mark_duplicates(records):
    """Group byte-identical images. Within a split, all but the lowest ID get
    duplicate_of; groups spanning train and test are reported separately."""
    groups = defaultdict(list)
    for r in records:
        groups[r["sha256"]].append(r)

    within = defaultdict(list)
    cross = []
    for members in groups.values():
        if len(members) < 2:
            continue
        splits = {r["split"] for r in members}
        if len(splits) > 1:
            cross.append(sorted(f"{r['split']}/{r['id']}" for r in members))
        for split in splits:
            same = sorted((r for r in members if r["split"] == split), key=lambda r: r["id"])
            if len(same) > 1:
                within[split].append([r["id"] for r in same])
                for r in same[1:]:
                    r["duplicate_of"] = same[0]["id"]
    return {s: sorted(g) for s, g in within.items()}, sorted(cross)


def label_conflicts(records):
    """Duplicate images whose ground-truth labels disagree (annotation noise).
    Only field names are reported, never label text: the report is committed."""
    groups = defaultdict(list)
    for r in records:
        groups[r["sha256"]].append(r)
    conflicts = []
    for members in groups.values():
        if len(members) < 2 or any(r["ground_truth"]["raw"] is None for r in members):
            continue
        raws = [r["ground_truth"]["raw"] for r in members]
        fields = [k for k in FIELDS if len({x[k] for x in raws}) > 1]
        if fields:
            conflicts.append({
                "receipts": [f"{r['split']}/{r['id']}" for r in members],
                "fields": fields,
            })
    return sorted(conflicts, key=lambda c: c["receipts"])


def field_stats(records):
    missing, empty, unparsed = Counter(), Counter(), Counter()
    for r in records:
        for issue in r["issues"]:
            kind, _, key = issue.partition(":")
            {"missing": missing, "empty": empty, "unparsed": unparsed}.get(kind, Counter())[key] += 1
    return {"missing": dict(missing), "empty": dict(empty), "unparsed": dict(unparsed)}


def pick_subset(records, n, seed):
    eligible = sorted(
        r["id"] for r in records
        if r["split"] == "test" and r["duplicate_of"] is None and not r["issues"]
    )
    if len(eligible) < n:
        sys.exit(f"[FAIL] only {len(eligible)} eligible test receipts, need {n}")
    return sorted(random.Random(seed).sample(eligible, n)), len(eligible)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=50, help="subset size")
    args = parser.parse_args()

    print("== Scanning SROIE Task 3")
    records, report = [], {"splits": {}}
    for split in SPLITS:
        recs, stats = scan_split(split)
        records += recs
        report["splits"][split] = stats

    within, cross = mark_duplicates(records)
    for split in SPLITS:
        recs = [r for r in records if r["split"] == split]
        report["splits"][split]["duplicate_image_groups"] = within.get(split, [])
        report["splits"][split]["fields"] = field_stats(recs)
        report["splits"][split]["receipts_with_issues"] = {r["id"]: r["issues"] for r in recs if r["issues"]}
    report["cross_split_duplicate_groups"] = cross
    report["duplicate_label_conflicts"] = conflicts = label_conflicts(records)

    subset_ids, n_eligible = pick_subset(records, args.n, args.seed)
    report["subset"] = {"split": "test", "n": args.n, "seed": args.seed, "eligible": n_eligible,
                        "rule": "test receipts, not a duplicate of a lower ID, no label issues"}

    manifest = HERE / "manifest.jsonl"
    subset_jsonl = HERE / f"subset_test_{args.n}.jsonl"
    subset_ids_txt = HERE / f"subset_test_{args.n}_ids.txt"
    report_json = HERE / "prepare_report.json"

    write_jsonl(manifest, records)
    chosen = set(subset_ids)
    write_jsonl(subset_jsonl, [r for r in records if r["id"] in chosen and r["split"] == "test"])
    subset_ids_txt.write_text("\n".join(subset_ids) + "\n", encoding="utf-8", newline="\n")
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print("\n== Stats")
    ok = True
    for split in SPLITS:
        s = report["splits"][split]
        exp = EXPECTED[split]
        flag = "OK" if s["pairs"] == exp else f"MISMATCH, expected {exp}"
        ok &= s["pairs"] == exp
        print(f"  {split}: {s['pairs']} pairs [{flag}]")
        print(f"    numbered copies ignored : {s['numbered_copies_ignored']}")
        print(f"    unmatched jpg / txt     : {len(s['jpg_without_txt'])} / {len(s['txt_without_jpg'])}")
        print(f"    other files             : {len(s['other_files'])}")
        print(f"    missing fields          : {s['fields']['missing'] or '-'}")
        print(f"    empty fields            : {s['fields']['empty'] or '-'}")
        print(f"    unparsed date/total     : {s['fields']['unparsed'] or '-'}")
        groups = s["duplicate_image_groups"]
        print(f"    duplicate image groups  : {len(groups)} {[' / '.join(g) for g in groups]}")
    print(f"  cross-split duplicates    : {len(cross)} {[' / '.join(g) for g in cross]}")
    print(f"  duplicates with differing labels: {len(conflicts)} "
          f"{[' / '.join(c['receipts']) + ' (' + ','.join(c['fields']) + ')' for c in conflicts]}")
    print(f"  subset: {args.n} of {n_eligible} eligible test receipts (seed {args.seed})")

    print("\n== Wrote")
    for p in (manifest, subset_jsonl, subset_ids_txt, report_json):
        print(f"  {rel(p)}")

    if not ok:
        sys.exit("\n[FAIL] pair counts do not match the official release")
    print("\n[OK] SROIE manifest ready")


if __name__ == "__main__":
    main()
