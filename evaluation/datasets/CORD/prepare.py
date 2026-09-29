"""Build the CORD v2 manifest in the same format as SROIE.

Reads the Hugging Face `save_to_disk` copy in raw/cord-v2/ (Arrow stream
files; image + ground_truth JSON). CORD has no company/date/address, so only
total.total_price is mapped (to `total`) and menu items are kept as optional
`line_items`. Amounts are rupiah and normalised with locale="id".

Validation + test receipts are exported to raw/export/{split}/<id>.png and
<id>.json and go into the manifest; train is only read for stats and the
duplicate check. Writes:

    manifest.jsonl              validation + test records (gitignored: contains labels)
    subset_valtest_50.jsonl     seeded evaluation subset (gitignored: contains labels)
    subset_valtest_50_ids.txt   receipt IDs of that subset (committed)
    prepare_report.json         counts, field coverage, data-quality findings (committed)

Usage:
    python evaluation/datasets/CORD/prepare.py [--seed 42] [--n 50]
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.ipc as ipc

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.normalize import parse_amount  # noqa: E402

RAW = HERE / "raw" / "cord-v2"
EXPORT = HERE / "raw" / "export"
SPLITS = ("train", "validation", "test")
EXPORT_SPLITS = ("validation", "test")
EXPECTED = {"train": 800, "validation": 100, "test": 100}
LOCALE = "id"
SROIE_FIELDS = ("company", "date", "address", "total")


def rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def read_split(split):
    """Yield (image bytes, ground_truth str) for every row of a split."""
    shards = sorted((RAW / split).glob("data-*.arrow"))
    if not shards:
        sys.exit(f"[FAIL] no Arrow files in {RAW / split}")
    for shard in shards:
        with ipc.open_stream(shard) as reader:
            for batch in reader:
                images = batch.column("image").field("bytes").to_pylist()
                gts = batch.column("ground_truth").to_pylist()
                yield from zip(images, gts)


def key_paths(obj, prefix=""):
    """All dotted key paths in a gt_parse tree (lists are transparent)."""
    paths = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            paths.add(prefix + k)
            paths |= key_paths(v, prefix + k + ".")
    elif isinstance(obj, list):
        for v in obj:
            paths |= key_paths(v, prefix)
    return paths


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def line_items(menu):
    """menu is a dict for single-item receipts and a list otherwise; always
    return a list, with nested `sub` items also as lists and prices normalised."""
    items = []
    for item in as_list(menu):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item["sub"] = line_items(item.get("sub"))
        item["price_normalized"] = parse_amount(item.get("price"), LOCALE)[0]
        items.append(item)
    return items


def map_total(total_block):
    """Return (raw total string or None, normalised or None, issues)."""
    value = total_block.get("total_price") if isinstance(total_block, dict) else None
    if value is None:
        return None, None, ["missing:total"]
    if isinstance(value, list):
        return " | ".join(map(str, value)), None, ["multiple:total"]
    value = str(value)
    if not value.strip():
        return value, None, ["empty:total"]
    normalized, irregular = parse_amount(value, LOCALE)
    if normalized is None:
        return value, None, ["unparsed:total"]
    return value, normalized, ["irregular:total"] if irregular else []


def build_record(split, image_bytes, gt):
    gt_parse = gt.get("gt_parse", {})
    image_id = gt["meta"]["image_id"]
    rid = f"{split}_{image_id:03d}"
    raw_total, norm_total, issues = map_total(gt_parse.get("total"))
    image_path = EXPORT / split / f"{rid}.png"
    return {
        "id": rid,
        "split": split,
        "image": rel(image_path),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "duplicate_of": None,
        "issues": issues,
        "locale": LOCALE,
        "fields_available": ["total"],
        "ground_truth": {
            "raw": {"company": None, "date": None, "address": None, "total": raw_total},
            "normalized": {"date": None, "total": norm_total},
            "line_items": line_items(gt_parse.get("menu")),
        },
    }


def export(record, image_bytes, gt_str):
    """Write <id>.png and <id>.json (full original ground truth) once."""
    png = REPO_ROOT / record["image"]
    png.parent.mkdir(parents=True, exist_ok=True)
    if not png.is_file() or png.stat().st_size != len(image_bytes):
        png.write_bytes(image_bytes)
    png.with_suffix(".json").write_text(gt_str, encoding="utf-8", newline="\n")


def mark_duplicates(records):
    """Same rule as SROIE: within a split all but the lowest ID get
    duplicate_of; groups spanning splits are reported separately."""
    groups = defaultdict(list)
    for r in records:
        groups[r["sha256"]].append(r)
    within, cross = defaultdict(list), []
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


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=50, help="subset size")
    args = parser.parse_args()

    print("== Reading CORD v2 (this reads ~2.3 GB of Arrow files)")
    records, report = [], {"locale": LOCALE, "splits": {}}
    for split in SPLITS:
        coverage = Counter()
        ids = []
        n_items = 0
        for image_bytes, gt_str in read_split(split):
            gt = json.loads(gt_str)
            coverage.update(key_paths(gt.get("gt_parse", {})))
            rec = build_record(split, image_bytes, gt)
            ids.append(rec["id"])
            n_items += len(rec["ground_truth"]["line_items"])
            records.append(rec)
            if split in EXPORT_SPLITS:
                export(rec, image_bytes, gt_str)
        if len(ids) != len(set(ids)):
            sys.exit(f"[FAIL] {split}: duplicate meta.image_id values")
        recs = [r for r in records if r["split"] == split]
        issue_counts = Counter(i for r in recs for i in r["issues"])
        report["splits"][split] = {
            "receipts": len(recs),
            "exported": split in EXPORT_SPLITS,
            "line_items": n_items,
            "total_issues": dict(sorted(issue_counts.items())),
            "receipts_with_issues": {r["id"]: r["issues"] for r in recs if r["issues"]},
            "gt_parse_field_coverage": dict(sorted(coverage.items())),
        }
        print(f"  {split}: {len(recs)} receipts")

    within, cross = mark_duplicates(records)
    for split in SPLITS:
        report["splits"][split]["duplicate_image_groups"] = within.get(split, [])
    report["cross_split_duplicate_groups"] = cross

    eligible = sorted(
        r["id"] for r in records
        if r["split"] in EXPORT_SPLITS and r["duplicate_of"] is None and not r["issues"]
    )
    if len(eligible) < args.n:
        sys.exit(f"[FAIL] only {len(eligible)} eligible receipts, need {args.n}")
    subset_ids = sorted(random.Random(args.seed).sample(eligible, args.n))
    report["subset"] = {"splits": list(EXPORT_SPLITS), "n": args.n, "seed": args.seed,
                        "eligible": len(eligible),
                        "rule": "validation+test receipts, not a duplicate of a lower ID, "
                                "exactly one regular total_price"}

    manifest_rows = [r for r in records if r["split"] in EXPORT_SPLITS]
    chosen = set(subset_ids)
    manifest = HERE / "manifest.jsonl"
    subset_jsonl = HERE / f"subset_valtest_{args.n}.jsonl"
    subset_ids_txt = HERE / f"subset_valtest_{args.n}_ids.txt"
    report_json = HERE / "prepare_report.json"

    write_jsonl(manifest, manifest_rows)
    write_jsonl(subset_jsonl, [r for r in manifest_rows if r["id"] in chosen])
    subset_ids_txt.write_text("\n".join(subset_ids) + "\n", encoding="utf-8", newline="\n")
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print("\n== Stats")
    ok = True
    for split in SPLITS:
        s = report["splits"][split]
        exp = EXPECTED[split]
        flag = "OK" if s["receipts"] == exp else f"MISMATCH, expected {exp}"
        ok &= s["receipts"] == exp
        cov = s["gt_parse_field_coverage"]
        print(f"  {split}: {s['receipts']} receipts [{flag}]{'  (exported)' if s['exported'] else ''}")
        print(f"    total issues            : {s['total_issues'] or '-'}")
        print(f"    has total.total_price   : {cov.get('total.total_price', 0)}/{s['receipts']}")
        print(f"    has menu / sub_total    : {cov.get('menu', 0)} / {cov.get('sub_total', 0)}")
        print(f"    line items              : {s['line_items']}")
        print(f"    duplicate image groups  : {len(s['duplicate_image_groups'])} {s['duplicate_image_groups']}")
    print(f"  missing SROIE fields      : {', '.join(f for f in SROIE_FIELDS if f != 'total')} "
          f"(not annotated in CORD; null in every record)")
    print(f"  cross-split duplicates    : {len(cross)} {[' / '.join(g) for g in cross]}")
    print(f"  manifest: {len(manifest_rows)} validation+test receipts; "
          f"subset: {args.n} of {len(eligible)} eligible (seed {args.seed})")

    print("\n== Wrote")
    for p in (manifest, subset_jsonl, subset_ids_txt, report_json):
        print(f"  {rel(p)}")
    print(f"  {rel(EXPORT)}/{{{','.join(EXPORT_SPLITS)}}}/<id>.png + .json")

    if not ok:
        sys.exit("\n[FAIL] receipt counts do not match the official release")
    print("\n[OK] CORD manifest ready")


if __name__ == "__main__":
    main()
