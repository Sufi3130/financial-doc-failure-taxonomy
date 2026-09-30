"""Few-shot examples from the SROIE train split only.

Eligible: train receipts with no label issues, not a duplicate of a lower ID,
and not byte-identical to any test receipt (7 such train/test pairs exist).
Selection is seeded and independent of the OCR engine, so every engine gets
the same example receipts; only their OCR text differs (it comes from the
same engine as the target, so examples look like the real input).
"""

import json
import random
from pathlib import Path

from src.ocr import cached_ocr

REPO_ROOT = Path(__file__).resolve().parents[2]
SROIE_MANIFEST = REPO_ROOT / "evaluation" / "datasets" / "SROIE" / "manifest.jsonl"


def load_manifest(path=SROIE_MANIFEST):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} missing: run python evaluation/datasets/SROIE/prepare.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eligible_train_ids(records):
    test_hashes = {r["sha256"] for r in records if r["split"] == "test"}
    return sorted(
        r["id"] for r in records
        if r["split"] == "train" and not r["issues"] and r["duplicate_of"] is None
        and r["sha256"] not in test_hashes
    )


def select_fewshot_ids(records, k=2, seed=42):
    if k == 0:
        return []
    return random.Random(seed).sample(eligible_train_ids(records), k)


def load_examples(engine, k=2, seed=42, records=None):
    """Return (ids, [(reading-order OCR text, raw ground-truth dict), ...])."""
    if k == 0:
        return [], []
    records = records or load_manifest()
    by_id = {r["id"]: r for r in records if r["split"] == "train"}
    ids = select_fewshot_ids(records, k, seed)
    examples = []
    for rid in ids:
        rec = by_id[rid]
        _, reading, _, _ = cached_ocr(engine, REPO_ROOT / rec["image"], receipt_id=rid)
        examples.append((reading.text, rec["ground_truth"]["raw"]))
    return ids, examples
