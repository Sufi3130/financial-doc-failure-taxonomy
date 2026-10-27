"""Automatic failure tagging for pipeline runs (failure taxonomy v0).

A failure instance is one field of one receipt for one OCR engine whose
normalised value is wrong. Each instance gets one primary category (the first
pipeline stage that went wrong) and a list of tags (which may overlap).
Two non-failure kinds are recorded as well, for the taxonomy: fields that are
right only after normalisation, and unlabelled fields the model filled in anyway.

Primary categories, in precedence order:
    pipeline_error            the receipt crashed (OCR or LLM exception)
    malformed_json            no usable JSON (subtype: derailment | prose_instead_of_json |
                              invalid_json | not_object | prompt_too_long)
    value_not_in_ocr          the gold value is not in the OCR text (OCR failure)
    fewshot_copy              the value was copied from a few-shot example answer and is not on
                              the receipt (also a tag, e.g. on unlabelled CORD fields)
    empty_field               value was in the OCR text, the model returned null/empty
    copy_corruption           the model changed 1-2 characters of a correctly read value
    wrong_date_format         unparseable date, or day and month swapped
    wrong_total               wrong amount (subtype: scale_error | other_amount_on_receipt |
                              amount_not_in_ocr), CORD also names the amount (cash, subtotal...)
    wrong_entity              another name/address printed on the receipt (brand vs operator,
                              buyer vs seller); generic words like SDN BHD are ignored
    boundary_too_wide         all gold tokens present, plus extra ones (subtype: adjacent_text |
                              hallucinated_extension)
    boundary_too_narrow       only a subset of the gold tokens
    partial_value             overlapping but neither a subset nor a superset
    wrong_value               any other wrong value (subtype other_date | not_in_ocr)
Non-failures:
    normalised_only_mismatch  strict mismatch, normalised match (formatting only)
    hallucinated_unlabelled_field  a value for a field the dataset does not annotate
Extra tags:
    ocr_confidence_low        OCR pieces carrying the gold value have confidence < 0.8,
                              or (receipt level) > 30% of the receipt's pieces do
    ocr_empty_page            fewer than 3 OCR pieces for the whole receipt
    fewshot_copy              see above

Usage:
    python evaluation/failure_taxonomy/tag.py \\
        --run runs/pipeline/<sroie_run> evaluation/datasets/SROIE/subset_test_50.jsonl \\
        --run runs/pipeline/<cord_run>  evaluation/datasets/CORD/subset_valtest_50.jsonl \\
        [--review 20 --seed 42]

Writes evaluation/failure_taxonomy/tags/<run_id>/{tags.csv, tags_detail.jsonl, counts.json}
(tags_detail.jsonl holds text and is gitignored) and, with --review, review_v0.csv.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import FIELDS, _NUMBER, load_last_rows, normalise, squash, tokens  # noqa: E402
from evaluation.normalize import normalize_date, normalize_total  # noqa: E402

HERE = Path(__file__).resolve().parent
TAGS_DIR = HERE / "tags"
LOW_CONF = 0.8
PRIMARY_ORDER = [
    "pipeline_error", "malformed_json", "value_not_in_ocr", "fewshot_copy", "empty_field", "copy_corruption",
    "wrong_date_format", "wrong_total", "wrong_entity", "boundary_too_wide", "boundary_too_narrow",
    "partial_value", "wrong_value", "normalised_only_mismatch", "hallucinated_unlabelled_field",
]
NON_FAILURES = {"normalised_only_mismatch", "hallucinated_unlabelled_field"}
STAGE = {
    "pipeline_error": "pipeline", "malformed_json": "llm_output", "value_not_in_ocr": "ocr",
    "normalised_only_mismatch": "formatting", "hallucinated_unlabelled_field": "llm_extraction",
}
CORD_AMOUNTS = {  # gt_parse paths of competing amounts on CORD receipts
    "cash": ("total", "cashprice"), "change": ("total", "changeprice"),
    "credit_card": ("total", "creditcardprice"), "subtotal": ("sub_total", "subtotal_price"),
    "tax": ("sub_total", "tax_price"), "service": ("sub_total", "service_price"),
}


def _amounts_in(text, locale):
    return {normalize_total(t, locale) for t in _NUMBER.findall(text or "")} - {None}


def _decimal(s):
    try:
        return Decimal(s)
    except (InvalidOperation, TypeError):
        return None


def cord_amounts(record):
    """Named competing amounts from the exported full CORD label, if available."""
    path = (REPO_ROOT / record["image"]).with_suffix(".json")
    if record.get("locale") != "id" or not path.is_file():
        return {}
    gt = json.loads(path.read_text(encoding="utf-8")).get("gt_parse", {})
    out = {}
    for name, (block, key) in CORD_AMOUNTS.items():
        value = gt.get(block, {}).get(key) if isinstance(gt.get(block), dict) else None
        if isinstance(value, str) and (n := normalize_total(value, "id")):
            out[name] = n
    return out


def gold_pieces_confidence(field, gold_norm, lines, locale):
    """Lowest confidence among OCR pieces that carry (part of) the gold value, or None."""
    confs = []
    for ln in lines:
        text = ln.get("text", "")
        if field == "total":
            hit = gold_norm in _amounts_in(text, locale)
        elif field == "date":
            hit = normalize_date(text) == gold_norm
        else:
            piece = squash(text)
            hit = len(piece) >= 3 and piece in gold_norm
        if hit:
            confs.append(ln.get("confidence", 1.0))
    return min(confs) if confs else None


GENERIC_TOKENS = {"SDN", "BHD", "BERHAD", "CO", "M", "PLT", "ENTERPRISE", "ENTERPRISES", "TRADING",
                  "THE", "AND", "OF", "NO", "JALAN", "TAMAN", "LOT", "SELANGOR", "KUALA", "LUMPUR"}


def edit_distance(a, b, limit=3):
    """Levenshtein distance, capped at `limit` (enough to spot 1-2 character copy errors)."""
    if abs(len(a) - len(b)) >= limit:
        return limit
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(prev[-1], limit)


def classify_field(field, pred, gold, gold_norm, locale, ocr_text, extra_amounts):
    """Primary category and subtype for a wrong field whose gold value is in the OCR text.
    Rules refined after the v0 manual review (see taxonomy_v0.md)."""
    if pred is None or not str(pred).strip():
        return "empty_field", None
    pred_sq, gold_sq, ocr_sq = squash(pred), squash(gold), squash(ocr_text)
    # the model changed characters of a value the OCR had read correctly: 1 edit in the
    # date as written ("2018" -> "2e18"), up to 2 in names/addresses; never for amounts
    if pred_sq and pred_sq not in ocr_sq:
        if field == "date" and edit_distance(pred.strip().upper(), gold.strip().upper()) <= 1:
            return "copy_corruption", None
        if field in ("company", "address") and edit_distance(pred_sq, gold_sq) <= 2:
            return "copy_corruption", None
    if field == "date":
        pred_norm = normalize_date(pred)
        if pred_norm is None:
            return "wrong_date_format", "unparseable"
        y, m, d = gold_norm.split("-")
        if pred_norm == f"{y}-{d}-{m}":
            return "wrong_date_format", "day_month_swap"
        return "wrong_value", "other_date"
    if field == "total":
        pred_norm = normalize_total(pred, locale)
        p, g = _decimal(pred_norm), _decimal(gold_norm)
        if p is not None and g is not None and g != 0 and p != 0 and (p * 1000 == g or p == g * 1000 or
                                                                       p * 100 == g or p == g * 100):
            return "wrong_total", "scale_error"
        named = [name for name, v in extra_amounts.items() if v == pred_norm]
        if named:
            return "wrong_total", "other_amount_on_receipt:" + "+".join(sorted(named))
        if pred_norm in _amounts_in(ocr_text, locale):
            return "wrong_total", "other_amount_on_receipt"
        # (no copy_corruption rule for amounts: 1-2 digit edits on short numbers are too
        # often a different amount, e.g. rescaled cash "50.00" vs total "41,000")
        return "wrong_total", "amount_not_in_ocr"
    p, g = set(tokens(pred)), set(tokens(gold))
    ocr_tokens = set(tokens(ocr_text))
    if g <= p:
        return "boundary_too_wide", "adjacent_text" if (p - g) <= ocr_tokens else "hallucinated_extension"
    if p <= g:
        return "boundary_too_narrow", None  # e.g. a name printed over two lines, first line only
    distinctive = (p & g) - GENERIC_TOKENS
    if not distinctive or len(distinctive) / max(len(g - GENERIC_TOKENS), 1) < 0.5:
        # another name/address that is printed on the receipt (brand vs operator, buyer vs seller)
        if pred_sq in ocr_sq or len(p & ocr_tokens) >= 0.8 * len(p):
            return "wrong_entity", None
        return "wrong_value", "not_in_ocr"
    return "partial_value", None


def malformed_subtype(outcome, generation):
    types = {f["type"] for f in outcome.get("failures", [])}
    raw = (generation or {}).get("raw_output") or ""
    if "prompt_too_long" in types:
        return "prompt_too_long"
    if "no_json" in types:
        return "prose_instead_of_json"
    if "truncated" in types and raw.lstrip().startswith("{"):
        return "derailment"
    if "not_object" in types:
        return "not_object"
    return "invalid_json"


def fewshot_answers(run_dir, engine):
    """Field values of the few-shot example answers used in a run, per field."""
    meta_path = Path(run_dir) / "run_meta.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    msgs = meta.get("engines", {}).get(engine, {}).get("prompt_prefix_messages", [])
    values = defaultdict(set)
    for m in msgs:
        if m["role"] == "assistant":
            try:
                for k, v in json.loads(m["content"]).items():
                    if isinstance(v, str) and v.strip():
                        values[k].add(squash(v))
            except (json.JSONDecodeError, AttributeError):
                pass
    return values


def tag_receipt(record, row, run_id, dataset, fewshot=None):
    """All failure / non-failure instances for one receipt and one engine row."""
    fewshot = fewshot or {}
    locale = record.get("locale", "my")
    scored = record.get("fields_available") or list(FIELDS)
    gt_raw, gt_norm = record["ground_truth"]["raw"], record["ground_truth"]["normalized"]
    outcome = row.get("outcome") or {}
    ocr = row.get("ocr") or {}
    lines, ocr_text = ocr.get("lines") or [], ocr.get("text", "")
    pred = outcome.get("lenient") or {}
    usable = bool(outcome.get("lenient")) and not row.get("error")
    low_share = sum(ln.get("confidence", 1) < LOW_CONF for ln in lines) / len(lines) if lines else 1.0
    receipt_tags = []
    if len(lines) < 3:
        receipt_tags.append("ocr_empty_page")
    if low_share > 0.3:
        receipt_tags.append("ocr_confidence_low:receipt")
    extra_amounts = cord_amounts(record) if "total" in scored else {}

    out = []
    base = {"run_id": run_id, "dataset": dataset, "engine": row["engine"], "receipt_id": record["id"]}
    for field in FIELDS:
        gold = gt_raw.get(field)
        p = pred.get(field)
        copied = bool(p) and squash(p) in fewshot.get(field, set()) and squash(p) not in squash(ocr_text)
        if field not in scored:
            if usable and p:
                out.append({**base, "field": field, "primary": "hallucinated_unlabelled_field",
                            "subtype": None, "tags": ["fewshot_copy"] if copied else [], "pred": p, "gold": None})
            continue
        if gold is None or not str(gold).strip():
            continue
        g_norm = gt_norm.get(field) if field in ("date", "total") else squash(gold)
        normalized_ok = g_norm is not None and normalise(field, p, locale) == g_norm
        strict_ok = p is not None and p.strip() == gold.strip()
        if normalized_ok:
            if not strict_ok:
                out.append({**base, "field": field, "primary": "normalised_only_mismatch", "subtype": None,
                            "tags": [], "pred": p, "gold": gold})
            continue

        from evaluation.evaluate import value_in_ocr
        in_ocr = value_in_ocr(field, g_norm, ocr_text, locale)
        tags = list(receipt_tags)
        if copied:
            tags.append("fewshot_copy")
        if not in_ocr:
            tags.append("value_not_in_ocr")
        else:
            conf = gold_pieces_confidence(field, g_norm, lines, locale)
            if conf is not None and conf < LOW_CONF:
                tags.append("ocr_confidence_low")
        if row.get("error"):
            primary, subtype = "pipeline_error", row["error"].get("type")
        elif not usable:
            primary, subtype = "malformed_json", malformed_subtype(outcome, row.get("llm"))
        elif not in_ocr:
            primary, subtype = "value_not_in_ocr", None
            if p is None or not str(p).strip():
                tags.append("empty_field")
        elif copied:
            primary, subtype = "fewshot_copy", None
        else:
            primary, subtype = classify_field(field, p, gold, g_norm, locale, ocr_text, extra_amounts)
        out.append({**base, "field": field, "primary": primary, "subtype": subtype,
                    "tags": sorted(set(tags)), "pred": p, "gold": gold})
    for inst in out:
        inst["stage"] = STAGE.get(inst["primary"], "llm_extraction")
        inst["ocr_excerpt"] = _excerpt(ocr_text, inst)
    return out


def _excerpt(ocr_text, inst):
    """The OCR row(s) most related to the gold or predicted value (for the detail file only)."""
    wanted = squash(inst.get("gold") or inst.get("pred") or "")
    rows = ocr_text.splitlines()
    hits = [r for r in rows if wanted and (squash(r) and (squash(r) in wanted or wanted in squash(r)))]
    return " | ".join(hits[:3]) if hits else " | ".join(rows[:2])


def tag_run(run_dir, manifest):
    run_dir = Path(run_dir)
    dataset = "CORD" if "CORD" in str(manifest).upper() else "SROIE"
    rows = load_last_rows(run_dir / "run.jsonl")
    records = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
    fewshot = {e: fewshot_answers(run_dir, e) for e in {e for e, _ in rows}}
    instances = []
    for (engine, rid), row in sorted(rows.items()):
        rec = next((r for r in records if r["id"] == rid), None)
        if rec is not None:
            instances += tag_receipt(rec, row, run_dir.name, dataset, fewshot[engine])
    return instances


def counts(instances):
    table = defaultdict(Counter)
    tag_table = defaultdict(Counter)
    subtypes = defaultdict(Counter)
    for i in instances:
        key = f"{i['dataset']}/{i['engine']}"
        table[key][i["primary"]] += 1
        for t in i["tags"]:
            tag_table[key][t.split(":")[0]] += 1
        if i["subtype"]:
            subtypes[i["primary"]][i["subtype"].split(":")[0] if i["primary"] != "wrong_total" else i["subtype"]] += 1
    return {"primary": {k: dict(v) for k, v in table.items()},
            "tags": {k: dict(v) for k, v in tag_table.items()},
            "subtypes": {k: dict(v) for k, v in subtypes.items()}}


def write_tags(run_id, instances, out_root=TAGS_DIR):
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "engine", "receipt_id", "field", "stage", "primary", "subtype", "tags"]
    with open(out / "tags.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for i in instances:
            w.writerow([i[c] if c != "tags" else ";".join(i["tags"]) for c in cols[:-1]] + [";".join(i["tags"])])
    with open(out / "tags_detail.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in instances:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    (out / "counts.json").write_text(json.dumps(counts(instances), indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def sample_review(instances, n=20, seed=42):
    """Seeded sample of failures spread over primary categories (round-robin)."""
    by_cat = defaultdict(list)
    for i in instances:
        if i["primary"] not in NON_FAILURES:
            by_cat[i["primary"]].append(i)
    rng = random.Random(seed)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
    picked = []
    order = [c for c in PRIMARY_ORDER if c in by_cat]
    while len(picked) < n and any(by_cat[c] for c in order):
        for cat in order:
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop())
    return picked


def write_review(picked, path):
    cols = ["review_id", "dataset", "engine", "receipt_id", "field", "auto_primary", "auto_subtype",
            "auto_tags", "verdict", "reviewed_category", "note", "reviewer"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for k, i in enumerate(picked, 1):
            w.writerow([f"R{k:02d}", i["dataset"], i["engine"], i["receipt_id"], i["field"], i["primary"],
                        i["subtype"] or "", ";".join(i["tags"]), "", "", "", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN_DIR", "MANIFEST"))
    parser.add_argument("--review", type=int, default=0, help="write a seeded review sample of N failures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    everything = []
    for run_dir, manifest in args.run:
        instances = tag_run(run_dir, manifest)
        out = write_tags(Path(run_dir).name, instances)
        everything += instances
        print(f"{Path(run_dir).name}: {len(instances)} instances -> {out.relative_to(REPO_ROOT).as_posix()}/")
    c = counts(everything)
    for key, cats in sorted(c["primary"].items()):
        print(f"\n== {key}")
        for cat in PRIMARY_ORDER:
            if cat in cats:
                print(f"   {cat:31} {cats[cat]:4}")
    print("\nsubtypes:", json.dumps(c["subtypes"], indent=1))
    if args.review:
        path = HERE / "review_v0.csv"
        write_review(sample_review(everything, args.review, args.seed), path)
        print(f"\nreview sample: {path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
