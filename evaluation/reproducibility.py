"""Compare two pipeline runs of the same manifest: are the outputs identical?

    python evaluation/reproducibility.py runs/pipeline/<run_a> runs/pipeline/<run_b>

Pairs the last row per (engine, receipt) of both run logs and compares the
reading-order OCR text, the raw model output and the extracted (lenient)
field values. Equality is proven with SHA-256 fingerprints, so the committed
result contains no receipt text or labels. Anyone with the dataset and the
code can rerun and compare fingerprints. Timings are expected to differ and
are summarised separately; run settings from run_meta.json are compared too.

Writes evaluation/results/reproducibility_<a>_vs_<b>.json (committed).
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import load_last_rows  # noqa: E402

RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
META_KEYS = ("manifest_sha256", "prompt_version", "shots", "fewshot_seed", "ocr_cache", "versions")
LLM_KEYS = ("model_file", "model_bytes", "n_ctx", "temperature", "seed", "max_tokens")


def fingerprint(value):
    """SHA-256 of a value's canonical JSON form (strings hash their exact text)."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_fingerprints(row):
    return {
        "ocr_text": fingerprint((row.get("ocr") or {}).get("text")),
        "raw_output": fingerprint((row.get("llm") or {}).get("raw_output")),
        "fields": fingerprint((row.get("outcome") or {}).get("lenient")),
    }


def compare_rows(rows_a, rows_b):
    """Per (engine, receipt) comparison; keys present in only one run are reported."""
    keys = sorted(set(rows_a) | set(rows_b))
    pairs, only_a, only_b = [], [], []
    for key in keys:
        if key not in rows_b:
            only_a.append(list(key))
            continue
        if key not in rows_a:
            only_b.append(list(key))
            continue
        fa, fb = row_fingerprints(rows_a[key]), row_fingerprints(rows_b[key])
        pairs.append({"engine": key[0], "receipt_id": key[1],
                      **{f"{k}_identical": fa[k] == fb[k] for k in fa},
                      "a": fa, "b": fb,
                      "a_ocr_cached": rows_a[key].get("ocr_cached"), "b_ocr_cached": rows_b[key].get("ocr_cached")})
    return pairs, only_a, only_b


def timing_summary(rows, key="total_s"):
    vals = [r["timings"].get(key) for r in rows.values() if (r.get("timings") or {}).get(key) is not None]
    return {"median": round(statistics.median(vals), 1), "mean": round(statistics.mean(vals), 1),
            "max": round(max(vals), 1), "n": len(vals)} if vals else None


def settings(meta):
    return {
        **{k: meta.get(k) for k in META_KEYS},
        "llm": {k: (meta.get("llm") or {}).get(k) for k in LLM_KEYS},
        "engines": {e: {"settings": v.get("settings"), "fewshot_ids": v.get("fewshot_ids"),
                        "prompt_prefix_sha256": fingerprint(v.get("prompt_prefix_messages"))}
                    for e, v in (meta.get("engines") or {}).items()},
    }


def diff(a, b, prefix=""):
    """Dotted paths whose values differ between two nested dicts."""
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff(va, vb, f"{prefix}{k}.")
        elif va != vb:
            out.append(f"{prefix}{k}")
    return out


def compare_runs(run_a, run_b):
    run_a, run_b = Path(run_a), Path(run_b)
    rows_a, rows_b = load_last_rows(run_a / "run.jsonl"), load_last_rows(run_b / "run.jsonl")
    meta_a = json.loads((run_a / "run_meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((run_b / "run_meta.json").read_text(encoding="utf-8"))
    pairs, only_a, only_b = compare_rows(rows_a, rows_b)

    per_engine = {}
    for engine in sorted({p["engine"] for p in pairs}):
        ps = [p for p in pairs if p["engine"] == engine]
        per_engine[engine] = {f"{k}_identical": f"{sum(p[f'{k}_identical'] for p in ps)}/{len(ps)}"
                              for k in ("ocr_text", "raw_output", "fields")}
    n = len(pairs)
    set_a, set_b = settings(meta_a), settings(meta_b)
    return {
        "runs": {"a": run_a.name, "b": run_b.name},
        "pairs_compared": n,
        "only_in_a": only_a, "only_in_b": only_b,
        "identical": {k: f"{sum(p[f'{k}_identical'] for p in pairs)}/{n}" for k in ("ocr_text", "raw_output", "fields")},
        "identical_per_engine": per_engine,
        "all_identical": all(p["ocr_text_identical"] and p["raw_output_identical"] and p["fields_identical"]
                             for p in pairs) and not only_a and not only_b,
        "ocr_from_cache": {"a": dict(Counter(str(p["a_ocr_cached"]) for p in pairs)),
                           "b": dict(Counter(str(p["b_ocr_cached"]) for p in pairs))},
        "conditions": {
            run: {"git": m.get("git"), "started": m.get("started"),
                  "power_at_start": (m.get("machine") or {}).get("power_at_start"),
                  "total_s_per_page": timing_summary(r)}
            for run, m, r in (("a", meta_a, rows_a), ("b", meta_b, rows_b))
        },
        "settings": {"a": set_a, "b": set_b, "differing_keys": diff(set_a, set_b)},
        "rows": [{k: v for k, v in p.items() if not k.endswith("_cached")} for p in pairs],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()
    report = compare_runs(args.run_a, args.run_b)
    a, b = report["runs"]["a"], report["runs"]["b"]
    out = RESULTS_DIR / f"reproducibility_{a}_vs_{b}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(f"runs: {a} vs {b}, {report['pairs_compared']} receipt x engine pairs")
    print(f"identical: {report['identical']}  per engine: {report['identical_per_engine']}")
    print(f"all identical: {report['all_identical']}  (only in a: {len(report['only_in_a'])}, "
          f"only in b: {len(report['only_in_b'])})")
    print(f"OCR from cache: {report['ocr_from_cache']}")
    print(f"settings that differ: {report['settings']['differing_keys'] or 'none'}")
    for run in ("a", "b"):
        c = report["conditions"][run]
        print(f"run {run}: commit {(c['git'] or {}).get('commit', '')[:7]}, uncommitted "
              f"{len((c['git'] or {}).get('uncommitted_changes') or [])} files, power {c['power_at_start']}, "
              f"total s/page {c['total_s_per_page']}")
    print(f"\nwritten: {out.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
