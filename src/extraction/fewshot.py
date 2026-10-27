"""Few-shot examples, drawn from the train split of the dataset being evaluated.

Pools:
    SROIE  evaluation/datasets/SROIE/manifest.jsonl (train records); eligible train
           receipts have no label issues, are not a duplicate of a lower ID, and are
           not byte-identical to any test receipt (7 such train/test pairs exist).
    CORD   evaluation/datasets/CORD/fewshot_pool.jsonl, a seeded pool of 20 CORD train
           receipts written by CORD/prepare.py with the same rules (and exactly one
           regular total). Their answers have company/date/address = null, because
           CORD does not annotate them.

Selection is seeded and independent of the OCR engine, so every engine gets
the same example receipts; only their OCR text differs (it comes from the
same engine as the target, so examples look like the real input).
"""

import json
import random
from pathlib import Path

from src.ocr import cached_ocr

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "evaluation" / "datasets"
SROIE_MANIFEST = DATASETS / "SROIE" / "manifest.jsonl"
CORD_POOL = DATASETS / "CORD" / "fewshot_pool.jsonl"
POOLS = {"SROIE": SROIE_MANIFEST, "CORD": CORD_POOL}


def pool_for(manifest=None):
    """Few-shot pool for a manifest path: the pool of the dataset folder it lives in
    (evaluation/datasets/<DATASET>/...). Default and fallback: SROIE."""
    if manifest:
        for part in Path(manifest).resolve().parts:
            if part.upper() in POOLS:
                return POOLS[part.upper()]
    return SROIE_MANIFEST


def load_manifest(path=SROIE_MANIFEST):
    if not Path(path).is_file():
        dataset = Path(path).parent.name
        raise FileNotFoundError(f"{path} missing: run python evaluation/datasets/{dataset}/prepare.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eligible_train_ids(records):
    eval_hashes = {r["sha256"] for r in records if r["split"] != "train"}
    return sorted(
        r["id"] for r in records
        if r["split"] == "train" and not r["issues"] and r["duplicate_of"] is None
        and r["sha256"] not in eval_hashes
    )


def select_fewshot_ids(records, k=2, seed=42):
    if k == 0:
        return []
    return random.Random(seed).sample(eligible_train_ids(records), k)


def load_examples(engine, k=2, seed=42, records=None, pool=SROIE_MANIFEST):
    """Return (ids, [(reading-order OCR text, raw ground-truth dict), ...])."""
    if k == 0:
        return [], []
    records = records or load_manifest(pool)
    by_id = {r["id"]: r for r in records if r["split"] == "train"}
    ids = select_fewshot_ids(records, k, seed)
    examples = []
    for rid in ids:
        rec = by_id[rid]
        _, reading, _, _ = cached_ocr(engine, REPO_ROOT / rec["image"], receipt_id=rid)
        examples.append((reading.text, rec["ground_truth"]["raw"]))
    return ids, examples
