"""End-to-end pipeline: image -> OCR -> LLM -> validated JSON, one receipt or a whole manifest.

Every processed receipt is appended as one line to runs/pipeline/<run>/run.jsonl
(OCR output, raw LLM output, parsed result, timings, error). run_meta.json
next to it records how the run was made (command, git commit, versions,
model, prompt, few-shot IDs, engine settings, warm-up) and a summary.

Usage (from the repo root):
    python -m src.run IMAGE [--engine paddleocr|tesseract|all]
    python -m src.run --manifest evaluation/datasets/SROIE/subset_test_50.jsonl --engine all
    python -m src.run --manifest ... --resume runs/pipeline/<run>   # continue an interrupted run

Timing notes:
    - The model is loaded once. Before each engine's receipts, a warm-up call
      evaluates the fixed prompt prefix (instructions + few-shot examples), so
      every page is timed under the same warm conditions; the warm-up is
      logged in run_meta.json, not per page.
    - With the OCR cache (default), a cached receipt's ocr_s is the time
      measured when its OCR actually ran on this machine (ocr_cached: true).
      --no-cache re-runs OCR.
    - With --resume, receipts already logged without an error are skipped;
      failed ones are retried and appended again (the evaluator should take
      the last line per engine and receipt).
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

from src.extraction import LocalLLM, build_messages, parse_output, to_phi3
from src.extraction.fewshot import load_examples
from src.extraction.prompts import PROMPT_VERSION
from src.ocr import ENGINES, cached_ocr, get_engine, new_run_dir
from src.ocr.runlog import CACHE_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(path):
    path = Path(path).resolve()
    return (path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path).as_posix()


def _git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, check=True).stdout.split("\n")
        return {"commit": commit, "uncommitted_changes": [c[3:] for c in changed if c.strip()]}
    except Exception as e:
        return {"commit": None, "error": str(e)}


def _versions():
    out = {"python": platform.python_version()}
    for pkg in ("paddleocr", "paddlepaddle", "pytesseract", "llama-cpp-python", "pydantic"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power_state():
    """{'source': 'ac'|'battery'|'unknown', 'battery_percent': int|None} on Windows, else None.
    Laptops throttle the CPU on battery, so timings are only comparable on AC."""
    if sys.platform != "win32":
        return None
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    source = {0: "battery", 1: "ac"}.get(status.ACLineStatus, "unknown")
    pct = status.BatteryLifePercent
    return {"source": source, "battery_percent": None if pct == 255 else pct}


def keep_awake(on=True):
    """Stop Windows from sleeping (idle timeout) while a run is in progress.
    Uses SetThreadExecutionState; the flag is cleared when the process exits
    or when called with on=False. No-op elsewhere."""
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def load_records(manifest=None, image=None):
    if image:
        return [{"id": Path(image).stem, "split": None, "image": str(Path(image).resolve())}]
    with open(manifest, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys(run_jsonl):
    """(engine, receipt_id) pairs already logged without an error."""
    keys = set()
    if Path(run_jsonl).is_file():
        with open(run_jsonl, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if not row.get("error"):
                    keys.add((row["engine"], row["receipt_id"]))
    return keys


def process_receipt(record, engine, llm, examples, fewshot_ids, run_id, use_cache=True,
                    cache_dir=CACHE_DIR):
    """Run one receipt through OCR -> LLM -> parse. Never raises: errors go into the row."""
    t_start = time.perf_counter()
    row = {
        "run_id": run_id, "receipt_id": record["id"], "split": record.get("split"),
        "engine": engine.name, "image": _rel(REPO_ROOT / record["image"]),
        "ocr_cached": None, "ocr": None, "llm": None, "outcome": None, "timings": {}, "error": None,
        "power": power_state(),
    }
    stage = "ocr"
    try:
        if record["id"] in fewshot_ids:
            raise ValueError("receipt is one of the few-shot examples")
        result, reading, _, hit = cached_ocr(engine, REPO_ROOT / record["image"], record["id"],
                                             cache_dir=cache_dir, force=not use_cache)
        row["ocr_cached"] = hit
        row["ocr"] = {
            "engine_version": result.engine_version,
            "text": reading.text,
            "rows": reading.rows,
            "lines": [ln.model_dump(mode="json") for ln in result.lines],
            "image_size": result.image_size,
        }
        row["timings"]["ocr_s"] = result.timings["ocr_s"]

        stage = "llm"
        gen = llm.generate(to_phi3(build_messages(reading.text, examples)))
        row["llm"] = gen.model_dump(mode="json")
        row["timings"].update(llm_s=gen.seconds, llm_prompt_s=gen.prompt_eval_s, llm_gen_s=gen.gen_s)

        stage = "parse"
        t0 = time.perf_counter()
        outcome = parse_output(gen.raw_output, gen.finish_reason)
        row["timings"]["parse_s"] = round(time.perf_counter() - t0, 4)
        row["outcome"] = outcome.model_dump(mode="json")
        t = row["timings"]
        t["total_s"] = round(t["ocr_s"] + t["llm_s"] + t["parse_s"], 3)
    except Exception as e:
        row["error"] = {"stage": stage, "type": type(e).__name__, "message": str(e)[:500]}
    row["timings"]["wall_s"] = round(time.perf_counter() - t_start, 3)
    return row


def summarize(rows):
    out = {}
    for engine in sorted({r["engine"] for r in rows}):
        rs = [r for r in rows if r["engine"] == engine]
        ok = [r for r in rs if not r["error"]]

        def stat(key):
            vals = [r["timings"][key] for r in ok if r["timings"].get(key) is not None]
            return {"mean": round(statistics.mean(vals), 2), "median": round(statistics.median(vals), 2)} if vals else None

        out[engine] = {
            "receipts": len(rs),
            "errors": len(rs) - len(ok),
            "strict_valid": sum(r["outcome"]["strict_valid"] for r in ok),
            "schema_valid": sum(r["outcome"]["schema_valid"] for r in ok),
            "failure_counts": dict(Counter(f["type"] for r in ok for f in r["outcome"]["failures"])),
            "repair_counts": dict(Counter(x for r in ok for x in r["outcome"]["repairs"])),
            "timings_s": {k: stat(k) for k in ("ocr_s", "llm_s", "llm_prompt_s", "llm_gen_s", "total_s")},
        }
    return out


def run(records, engines, llm, run_dir, shots=2, seed=42, use_cache=True, cache_dir=CACHE_DIR,
        fewshot_loader=load_examples, meta=None, log=print):
    """Process records with each OCREngine (engine-major order), appending to run.jsonl."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_jsonl, meta_path = run_dir / "run.jsonl", run_dir / "run_meta.json"
    meta = meta or {}
    meta.setdefault("run_id", run_dir.name)
    meta.setdefault("engines", {})
    skip = done_keys(run_jsonl)

    def write_meta():
        meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    todo = [(e.name, r) for e in engines for r in records if (e.name, r["id"]) not in skip]
    if skip:
        log(f"resume: {len(skip)} receipt runs already done, {len(todo)} to go")
    write_meta()

    done_times = []
    n_done = 0
    for engine in engines:
        pending = [r for name, r in todo if name == engine.name]
        if not pending:
            continue
        ids, examples = fewshot_loader(engine, k=shots, seed=seed)

        t0 = time.perf_counter()
        warm = llm.generate(to_phi3(build_messages("", examples)), max_tokens=1)
        meta["engines"][engine.name] = {
            "settings": engine.settings(), "version": engine.version(), "fewshot_ids": ids,
            "prompt_prefix_messages": to_phi3(build_messages("<RECEIPT>", examples)),
            "warmup": {"seconds": round(time.perf_counter() - t0, 3),
                       "prompt_tokens": warm.prompt_tokens,
                       "prompt_tokens_evaluated": warm.prompt_tokens_evaluated},
        }
        meta["llm"] = llm.info()
        write_meta()
        log(f"\n== {engine.name}: {len(pending)} receipts, few-shot {ids or 'none'}, "
            f"warm-up {meta['engines'][engine.name]['warmup']['seconds']:.1f}s")

        for record in pending:
            row = process_receipt(record, engine, llm, examples, ids, meta["run_id"], use_cache, cache_dir)
            with open(run_jsonl, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            done_times.append(row["timings"]["wall_s"])
            eta_min = statistics.mean(done_times) * (len(todo) - n_done) / 60
            t = row["timings"]
            if row["error"]:
                status = f"ERROR {row['error']['stage']}: {row['error']['type']}: {row['error']['message'][:80]}"
            else:
                o = row["outcome"]
                fails = ",".join(f["type"] + (":" + f["field"] if f.get("field") else "") for f in o["failures"])
                status = (f"ocr {t['ocr_s']:5.1f}s{'*' if row['ocr_cached'] else ' '} llm {t['llm_s']:5.1f}s "
                          f"total {t['total_s']:5.1f}s  "
                          f"{'strict_valid' if o['strict_valid'] else 'schema_valid' if o['schema_valid'] else 'INVALID'}"
                          f"  failures: {fails or '-'}")
            log(f"[{n_done}/{len(todo)}] {engine.name:9} {record['id']:14} {status}  ETA {eta_min:.0f} min")

    rows = []
    if run_jsonl.is_file():
        with open(run_jsonl, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["summary"] = summarize(rows)
    write_meta()
    return run_jsonl, meta


def main():
    parser = argparse.ArgumentParser(prog="python -m src.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("image", nargs="?", help="one receipt image")
    src.add_argument("--manifest", help="manifest or subset .jsonl (id, split, image per line)")
    parser.add_argument("--engine", default="paddleocr", choices=[*ENGINES, "all"])
    parser.add_argument("--shots", type=int, default=2, help="few-shot examples from SROIE train")
    parser.add_argument("--seed", type=int, default=42, help="few-shot selection seed")
    parser.add_argument("--limit", type=int, help="only the first N receipts")
    parser.add_argument("--no-cache", action="store_true", help="re-run OCR even if cached")
    parser.add_argument("--resume", help="existing run directory to continue")
    args = parser.parse_args()

    records = load_records(args.manifest, args.image)[: args.limit]
    engines = [get_engine(n) for n in (ENGINES if args.engine == "all" else [args.engine])]
    run_dir = Path(args.resume) if args.resume else new_run_dir("pipeline")

    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if args.resume and meta_path.is_file() else {
        "started": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(["python -m src.run", *sys.argv[1:]]),
        "manifest": _rel(args.manifest) if args.manifest else None,
        "manifest_sha256": _sha256(args.manifest) if args.manifest else None,
        "n_receipts": len(records),
        "prompt_version": PROMPT_VERSION, "shots": args.shots, "fewshot_seed": args.seed,
        "ocr_cache": not args.no_cache,
        "git": _git_state(), "versions": _versions(),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "logical_cpus": os.cpu_count(), "power_at_start": power_state()},
    }
    if args.resume:
        meta.setdefault("resumed", []).append(datetime.now().isoformat(timespec="seconds"))

    power = power_state()
    print(f"run: {_rel(run_dir)}  ({len(records)} receipts x {len(engines)} engine(s))")
    if power and power["source"] != "ac":
        print(f"[warn] running on {power['source']} power: timings will not be comparable to AC runs")
    keep_awake(True)
    try:
        run_jsonl, meta = run(records, engines, LocalLLM(), run_dir, args.shots, args.seed,
                              use_cache=not args.no_cache, meta=meta)
    finally:
        keep_awake(False)

    print("\n== Summary")
    for engine, s in meta["summary"].items():
        t = s["timings_s"]
        print(f"  {engine}: {s['receipts']} receipts, {s['errors']} errors, "
              f"strict_valid {s['strict_valid']}, schema_valid {s['schema_valid']}")
        print(f"    failures: {s['failure_counts'] or '-'}  repairs: {s['repair_counts'] or '-'}")
        for k in ("ocr_s", "llm_s", "llm_prompt_s", "llm_gen_s", "total_s"):
            if t[k]:
                print(f"    {k:13} mean {t[k]['mean']:6.1f}  median {t[k]['median']:6.1f}")
    print(f"\nrun log: {_rel(run_jsonl)}\nmeta:    {_rel(run_dir / 'run_meta.json')}")


if __name__ == "__main__":
    main()
