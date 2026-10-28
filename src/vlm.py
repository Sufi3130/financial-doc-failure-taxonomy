"""OCR-free ablation: a vision-language model reads the receipt image and outputs JSON directly.

End-to-end counterpart to the OCR -> LLM pipeline (src/run.py). The controlled
comparison is Qwen2.5-3B fed PaddleOCR text vs Qwen2.5-VL-3B fed the image:
same family and size, so the only difference is the OCR stage.

Writes a run log in the same format as src/run.py (so evaluation/evaluate.py
scores it unchanged); the OCR fields are empty because there is no OCR stage.

Usage (from the repo root):
    python -m src.vlm --manifest evaluation/datasets/SROIE/subset_test_50.jsonl
    python -m src.vlm --manifest ... --resume runs/vlm/<run>
"""

import argparse
import base64
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.extraction import parse_output
from src.extraction.prompts import SYSTEM
from src.ocr.runlog import RUNS_DIR
from src.run import (_git_state, _rel, _sha256, _versions, done_keys, keep_awake, load_records,
                     power_state, summarize)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
VLM_MODELS = {
    "qwen2.5-vl-3b": ("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"),
}
DEFAULT_VLM = "qwen2.5-vl-3b"
PROMPT = SYSTEM + "\n\nExtract the four fields from this receipt image:"


class QwenVL:
    """Qwen2.5-VL via llama-cpp-python's multimodal chat handler, CPU only."""

    def __init__(self, model="qwen2.5-vl-3b", n_ctx=8192, max_tokens=256, seed=0, max_side=1280):
        self.model = model
        self.model_path, self.mmproj_path = (str(MODELS_DIR / f) for f in VLM_MODELS[model])
        self.n_ctx, self.max_tokens, self.seed, self.max_side = n_ctx, max_tokens, seed, max_side
        self._llm = None
        self.load_s = 0.0

    def info(self):
        p = Path(self.model_path)
        return {"model": self.model, "model_file": p.name, "model_bytes": p.stat().st_size if p.is_file() else None,
                "mmproj_file": Path(self.mmproj_path).name, "n_ctx": self.n_ctx, "max_tokens": self.max_tokens,
                "seed": self.seed, "max_image_side": self.max_side, "n_gpu_layers": 0,
                "load_s": round(self.load_s, 3)}

    def load(self):
        if self._llm is not None:
            return
        for p in (self.model_path, self.mmproj_path):
            if not Path(p).is_file():
                raise FileNotFoundError(f"VLM file not found: {p} (see README 'Download the model')")
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        t0 = time.perf_counter()
        handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path, verbose=False)
        self._llm = Llama(model_path=self.model_path, chat_handler=handler, n_ctx=self.n_ctx,
                          n_gpu_layers=0, seed=self.seed, verbose=False)
        self.load_s = time.perf_counter() - t0

    def _data_uri(self, image_path):
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            if max(im.size) > self.max_side:  # cap size: vision-encoder cost scales with image tokens
                im.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
            size = im.size
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}", size

    def extract(self, image_path):
        """Return a Generation-like dict for one image."""
        self.load()
        uri, size = self._data_uri(image_path)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": uri}}]}]
        t0 = time.perf_counter()
        out = self._llm.create_chat_completion(messages=messages, temperature=0.0, seed=self.seed,
                                               max_tokens=self.max_tokens)
        seconds = round(time.perf_counter() - t0, 3)
        return {
            "raw_output": out["choices"][0]["message"]["content"],
            "finish_reason": out["choices"][0]["finish_reason"],
            "prompt_tokens": out["usage"]["prompt_tokens"],
            "completion_tokens": out["usage"]["completion_tokens"],
            "seconds": seconds, "image_resized_to": list(size),
        }


def process_receipt(record, vlm, run_id):
    t0 = time.perf_counter()
    row = {"run_id": run_id, "receipt_id": record["id"], "split": record.get("split"),
           "engine": "vlm", "model": vlm.model, "image": _rel(REPO_ROOT / record["image"]),
           "ocr_cached": None, "ocr": {"text": "", "rows": [], "lines": [], "end_to_end": True},
           "llm": None, "outcome": None, "timings": {}, "error": None, "power": power_state()}
    try:
        gen = vlm.extract(REPO_ROOT / record["image"])
        row["llm"] = gen
        outcome = parse_output(gen["raw_output"], gen["finish_reason"])
        row["outcome"] = outcome.model_dump(mode="json")
        row["timings"] = {"llm_s": gen["seconds"], "total_s": gen["seconds"]}
    except Exception as e:
        row["error"] = {"stage": "vlm", "type": type(e).__name__, "message": str(e)[:500]}
    row["timings"]["wall_s"] = round(time.perf_counter() - t0, 3)
    return row


def run(records, vlm, run_dir, meta, log=print):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_jsonl, meta_path = run_dir / "run.jsonl", run_dir / "run_meta.json"
    skip = done_keys(run_jsonl)
    todo = [r for r in records if ("vlm", r["id"]) not in skip]
    if skip:
        log(f"resume: {len(skip)} already done, {len(todo)} to go")
    meta["vlm"] = vlm.info()
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    times = []
    for i, record in enumerate(todo, 1):
        row = process_receipt(record, vlm, meta["run_id"])
        with open(run_jsonl, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        times.append(row["timings"]["wall_s"])
        eta = sum(times) / len(times) * (len(todo) - i) / 60
        t = row["timings"]
        if row["error"]:
            status = f"ERROR {row['error']['type']}: {row['error']['message'][:70]}"
        else:
            o = row["outcome"]
            status = (f"{t['total_s']:6.1f}s  "
                      f"{'strict_valid' if o['strict_valid'] else 'schema_valid' if o['schema_valid'] else 'INVALID'}")
        log(f"[{i}/{len(todo)}] {record['id']:14} {status}  ETA {eta:.0f} min")

    rows = [json.loads(line) for line in open(run_jsonl, encoding="utf-8")] if run_jsonl.is_file() else []
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["summary"] = summarize(rows)  # {"vlm": {...}}
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return run_jsonl, meta


def main():
    parser = argparse.ArgumentParser(prog="python -m src.vlm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default=DEFAULT_VLM, choices=list(VLM_MODELS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-side", type=int, default=1280, help="cap image long side (speed vs legibility)")
    parser.add_argument("--resume", help="existing run directory to continue")
    args = parser.parse_args()

    records = load_records(args.manifest)[: args.limit]
    run_dir = Path(args.resume) if args.resume else RUNS_DIR / "vlm" / datetime.now().strftime("%Y%m%d-%H%M%S")
    vlm = QwenVL(model=args.model, max_side=args.max_side)

    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if args.resume and meta_path.is_file() else {
        "started": datetime.now().isoformat(timespec="seconds"), "run_id": run_dir.name,
        "command": " ".join(["python -m src.vlm", *sys.argv[1:]]),
        "manifest": _rel(args.manifest), "manifest_sha256": _sha256(args.manifest),
        "n_receipts": len(records), "mode": "end_to_end_vlm", "model": args.model,
        "prompt": "SYSTEM + image (zero-shot)",
        "git": _git_state(), "versions": _versions(),
        "machine": {"power_at_start": power_state()},
    }

    print(f"VLM run: {_rel(run_dir)}  ({len(records)} receipts, model {args.model}, end-to-end)")
    keep_awake(True)
    try:
        run_jsonl, meta = run(records, vlm, run_dir, meta)
    finally:
        keep_awake(False)

    s = meta["summary"]["vlm"]
    print(f"\n== {s['receipts']} receipts, {s['errors']} errors, strict-valid JSON {s['strict_valid']}, "
          f"schema-valid {s['schema_valid']}")
    print(f"   failures: {s['failure_counts'] or '-'}")
    t = s["timings_s"].get("total_s")
    if t:
        print(f"   time/receipt median/mean: {t['median']}/{t['mean']}s")
    print(f"\nrun log: {_rel(run_jsonl)}")
    print(f"field accuracy: python evaluation/evaluate.py {_rel(run_dir)} --manifest {_rel(args.manifest)}")


if __name__ == "__main__":
    main()
