"""Flag SROIE gold labels that contain OCR-like noise/typos (issue #28).

Several SROIE ground-truth labels carry scanner/transcription errors (e.g.
`KOPITAM SND` for `KOPITIAM SDN`, `SHAN ALAM` for `SHAH ALAM`), so a strict
exact-match score under-counts correct extractions on those receipts. This
scans the manifest for a curated set of known-noisy tokens and writes a light
quality flag; the trusted scores remain the normalised/lenient match and the
manual review, not strict match.

    python evaluation/datasets/SROIE/flag_label_noise.py

Reads the (gitignored) manifest.jsonl and writes the committed
evaluation/datasets/SROIE/label_noise.jsonl. The pattern list is curated from
the 50-case manual review (taxonomy_v1.md Section 5); extend it as more cases
surface.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.jsonl"
OUT = HERE / "label_noise.jsonl"

# (regex, field, human description) — curated, not exhaustive.
PATTERNS = [
    (re.compile(r"KOPITAM", re.I), "company", "KOPITAM -> KOPITIAM (missing I)"),
    (re.compile(r"\bSND\s+BHD\b", re.I), "company", "SND -> SDN (transposed)"),
    (re.compile(r"SHAN ALAM", re.I), "address", "SHAN ALAM -> SHAH ALAM (N/H swap)"),
    (re.compile(r"DLNARG", re.I), "address", "JALAN DLNARG -> garbled street name"),
    (re.compile(r"\bB1750\b"), "address", "B1750 -> 81750 (postcode B/8)"),
    (re.compile(r"SEITA\s+(ALAM|PRIMA)", re.I), "address", "SEITA -> SETIA"),
]


def scan(manifest: Path):
    flags, seen = [], set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = (r.get("ground_truth") or {}).get("raw") or {}
        for rx, field, why in PATTERNS:
            val = raw.get(field) or ""
            key = (r["id"], field, why)
            if val and rx.search(val) and key not in seen:
                seen.add(key)
                flags.append({"id": r["id"], "split": r.get("split"),
                              "field": field, "gold": val, "issue": why})
    return flags


def main():
    if not MANIFEST.is_file():
        raise SystemExit(f"manifest not found: {MANIFEST} (it is gitignored; build it first)")
    flags = scan(MANIFEST)
    OUT.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in flags) + "\n",
                   encoding="utf-8")
    test = sum(f["split"] == "test" for f in flags)
    print(f"flagged {len(flags)} (receipt, field) cases ({test} in the test split)")
    print(f"written: {OUT.relative_to(HERE.parents[2]).as_posix()}")


if __name__ == "__main__":
    main()
