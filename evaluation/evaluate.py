"""Score a pipeline run (runs/pipeline/<run>/run.jsonl) against a dataset manifest.

Metrics per OCR engine:
    - field accuracy, strict (exact string after trimming) and normalised
      (date/total via evaluation/normalize.py with the record's locale;
      company/address ignoring case, spaces and punctuation)
    - token F1 for company/address (partial credit for boundary errors)
    - value-in-OCR rate, split into OCR misses (value never read) and LLM
      misses (value was in the OCR text but not extracted correctly)
    - schema compliance: strict_valid and schema_valid rates
    - latency per page: mean and median OCR / LLM / total seconds

The denominator is every receipt in the manifest: a missing row, a crash or
an unparseable output counts as wrong. Only fields listed in a record's
`fields_available` are scored (CORD: total only). For receipts retried with
--resume, the last row per engine and receipt is used.

Writes evaluation/results/<run_id>/:
    summary.json               metrics (committed)
    per_receipt.csv            match flags, failures, timings; no text (committed)
    per_receipt_detail.jsonl   the same plus predicted and ground-truth text (gitignored)

Usage:
    python evaluation/evaluate.py runs/pipeline/<run> --manifest evaluation/datasets/SROIE/subset_test_50.jsonl
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.normalize import normalize_date, normalize_total  # noqa: E402

FIELDS = ("company", "date", "address", "total")
TEXT_FIELDS = ("company", "address")
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")


def squash(s):
    """Uppercase alphanumerics only: 'Sdn. Bhd.' -> 'SDNBHD'."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def tokens(s):
    return re.findall(r"[A-Z0-9]+", (s or "").upper())


def token_f1(pred, gold):
    p, g = Counter(tokens(pred)), Counter(tokens(gold))
    overlap = sum((p & g).values())
    if not p or not g or not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(g.values())
    return round(2 * precision * recall / (precision + recall), 4)


def normalise(field, value, locale):
    if value is None or not str(value).strip():
        return None
    if field == "date":
        return normalize_date(value)
    if field == "total":
        return normalize_total(value, locale)
    return squash(value) or None


def value_in_ocr(field, gold_norm, ocr_text, locale):
    """Whether the ground-truth value can be found anywhere in the OCR text."""
    if gold_norm is None or not ocr_text:
        return False
    if field == "total":
        return gold_norm in {normalize_total(t, locale) for t in _NUMBER.findall(ocr_text)}
    if field == "date":
        return any(normalize_date(line) == gold_norm for line in ocr_text.splitlines())
    return gold_norm in squash(ocr_text)


def load_last_rows(run_jsonl):
    """Last row per (engine, receipt_id)."""
    rows = {}
    with open(run_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[(row["engine"], row["receipt_id"])] = row
    return rows


def score_receipt(record, row):
    """Per-receipt result for one engine; row may be None (receipt never run)."""
    locale = record.get("locale", "my")
    fields = record.get("fields_available") or list(FIELDS)
    gt_raw, gt_norm = record["ground_truth"]["raw"], record["ground_truth"]["normalized"]
    outcome = (row or {}).get("outcome") or {}
    pred = outcome.get("lenient") or {}
    ocr_text = ((row or {}).get("ocr") or {}).get("text", "")

    result = {
        "receipt_id": record["id"],
        "engine": (row or {}).get("engine"),
        "error": ((row or {}).get("error") or {}).get("type") if row else "not_run",
        "strict_valid": bool(outcome.get("strict_valid")),
        "schema_valid": bool(outcome.get("schema_valid")),
        "failures": sorted({f["type"] for f in outcome.get("failures", [])}),
        "timings": (row or {}).get("timings") or {},
        "power": ((row or {}).get("power") or {}).get("source"),
        "fields": {},
    }
    for field in fields:
        gold = gt_raw.get(field)
        if gold is None or not str(gold).strip():
            continue  # unannotated for this receipt
        gold_n = gt_norm.get(field) if field in ("date", "total") else squash(gold)
        p = pred.get(field)
        entry = {
            "pred": p,
            "gold": gold,
            "strict": p is not None and p.strip() == gold.strip(),
            "normalized": gold_n is not None and normalise(field, p, locale) == gold_n,
            "in_ocr": value_in_ocr(field, gold_n, ocr_text, locale),
        }
        if field in TEXT_FIELDS:
            entry["token_f1"] = token_f1(p, gold)
        result["fields"][field] = entry
    return result


def _pct(n, d):
    return round(100 * n / d, 1) if d else None


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(statistics.mean(values), 1), "median": round(statistics.median(values), 1),
            "n": len(values)}


def summarize(results):
    n = len(results)
    out = {"receipts": n, "errors": sum(bool(r["error"]) for r in results), "fields": {}}
    fields = [f for f in FIELDS if any(f in r["fields"] for r in results)]
    for field in fields:
        scored = [r["fields"][field] for r in results if field in r["fields"]]
        d = len(scored)
        s = {
            "scored": d,
            "strict_acc": _pct(sum(e["strict"] for e in scored), d),
            "normalized_acc": _pct(sum(e["normalized"] for e in scored), d),
            "in_ocr": _pct(sum(e["in_ocr"] for e in scored), d),
            "ocr_miss": sum(not e["in_ocr"] for e in scored),
            "llm_miss_given_in_ocr": sum(e["in_ocr"] and not e["normalized"] for e in scored),
        }
        if field in TEXT_FIELDS:
            s["mean_token_f1"] = round(100 * statistics.mean(e["token_f1"] for e in scored), 1)
        out["fields"][field] = s
    complete = [r for r in results if all(f in r["fields"] for f in fields)]
    out["all_fields_normalized_acc"] = _pct(
        sum(all(r["fields"][f]["normalized"] for f in fields) for r in complete), len(complete))
    out["schema"] = {
        "strict_valid_rate": _pct(sum(r["strict_valid"] for r in results), n),
        "schema_valid_rate": _pct(sum(r["schema_valid"] for r in results), n),
        "failure_counts": dict(Counter(f for r in results for f in r["failures"])),
    }
    timed = [r for r in results if not r["error"]]
    out["latency_s"] = {k: _stats(r["timings"].get(k) for r in timed)
                        for k in ("ocr_s", "llm_s", "llm_prompt_s", "llm_gen_s", "total_s")}
    out["latency_s"]["power"] = dict(Counter(str(r["power"]) for r in timed))
    return out


def evaluate(run_dir, manifest):
    run_dir = Path(run_dir)
    rows = load_last_rows(run_dir / "run.jsonl")
    with open(manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    engines = sorted({e for e, _ in rows})
    per_receipt = {e: [score_receipt(rec, rows.get((e, rec["id"]))) for rec in records] for e in engines}
    for e, results in per_receipt.items():
        for r in results:
            r["engine"] = e
    return records, per_receipt, {e: summarize(res) for e, res in per_receipt.items()}


def write_results(run_dir, manifest, records, per_receipt, summaries, out_root=RESULTS_DIR):
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_dir.name,
        "manifest": Path(manifest).resolve().relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "n_receipts": len(records),
        "run": {k: meta.get(k) for k in ("started", "finished", "command", "git", "prompt_version",
                                         "shots", "fewshot_seed", "ocr_cache")},
        "fewshot_ids": {e: v.get("fewshot_ids") for e, v in meta.get("engines", {}).items()},
        "engines": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")

    fields = [f for f in FIELDS if any(f in r["fields"] for res in per_receipt.values() for r in res)]
    header = ["engine", "receipt_id", "error", "strict_valid", "schema_valid", "failures"]
    for f in fields:
        header += [f"{f}_strict", f"{f}_normalized", f"{f}_in_ocr"] + ([f"{f}_token_f1"] if f in TEXT_FIELDS else [])
    header += ["ocr_s", "llm_s", "total_s", "power"]
    with open(out_dir / "per_receipt.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for e in sorted(per_receipt):
            for r in per_receipt[e]:
                line = [e, r["receipt_id"], r["error"] or "", int(r["strict_valid"]), int(r["schema_valid"]),
                        ";".join(r["failures"])]
                for f in fields:
                    x = r["fields"].get(f)
                    vals = [int(x["strict"]), int(x["normalized"]), int(x["in_ocr"])] if x else ["", "", ""]
                    if f in TEXT_FIELDS:
                        vals.append(x["token_f1"] if x else "")
                    line += vals
                line += [r["timings"].get("ocr_s", ""), r["timings"].get("llm_s", ""),
                         r["timings"].get("total_s", ""), r["power"] or ""]
                w.writerow(line)

    with open(out_dir / "per_receipt_detail.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for e in sorted(per_receipt):
            for r in per_receipt[e]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_dir


def print_summary(summaries):
    for engine, s in summaries.items():
        print(f"\n== {engine}: {s['receipts']} receipts, {s['errors']} errors")
        print(f"   {'field':8} {'strict':>7} {'norm':>7} {'tokF1':>6} {'inOCR':>6} {'OCRmiss':>8} {'LLMmiss':>8}")
        for f, x in s["fields"].items():
            f1 = x.get("mean_token_f1")
            print(f"   {f:8} {x['strict_acc']:>6}% {x['normalized_acc']:>6}% {'' if f1 is None else f1:>6} "
                  f"{x['in_ocr']:>5}% {x['ocr_miss']:>8} {x['llm_miss_given_in_ocr']:>8}")
        print(f"   all fields correct (normalised): {s['all_fields_normalized_acc']}%")
        sc = s["schema"]
        print(f"   schema: strict_valid {sc['strict_valid_rate']}%, schema_valid {sc['schema_valid_rate']}%, "
              f"failures {sc['failure_counts'] or '-'}")
        lat = s["latency_s"]
        print("   latency s/page: " + ", ".join(
            f"{k[:-2]} {v['median']} (mean {v['mean']})" for k, v in lat.items() if k != "power" and v)
              + f"  power {lat['power']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    records, per_receipt, summaries = evaluate(args.run_dir, args.manifest)
    out_dir = write_results(args.run_dir, args.manifest, records, per_receipt, summaries)
    print_summary(summaries)
    print(f"\nresults: {out_dir.relative_to(REPO_ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
