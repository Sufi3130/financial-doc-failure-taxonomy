"""Extract one receipt: OCR (cached) -> few-shot prompt -> Phi-3 -> validated JSON.

Usage (from the repo root):
    python -m src.extraction IMAGE [--engine paddleocr|tesseract] [--shots 2] [--seed 42]
"""

import argparse
import json
from pathlib import Path

from src.ocr import ENGINES, cached_ocr, get_engine, new_run_dir

from . import LocalLLM, extract, save_extraction
from .fewshot import load_examples


def main():
    parser = argparse.ArgumentParser(prog="python -m src.extraction", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--engine", default="paddleocr", choices=list(ENGINES))
    parser.add_argument("--shots", type=int, default=2, help="few-shot examples from SROIE train")
    parser.add_argument("--seed", type=int, default=42, help="few-shot selection seed")
    args = parser.parse_args()

    receipt_id = Path(args.image).stem
    engine = get_engine(args.engine)

    ids, examples = load_examples(engine, k=args.shots, seed=args.seed)
    if receipt_id in ids:
        raise SystemExit(f"{receipt_id} is one of the few-shot examples; pick another receipt")
    print(f"few-shot examples ({args.engine} OCR): {ids or 'none'}")

    ocr, reading, ocr_log, hit = cached_ocr(engine, args.image, receipt_id)
    print(f"OCR {'(cached) ' if hit else ''}{ocr.engine}: {len(ocr.lines)} pieces, {len(reading.rows)} rows")

    record = extract(reading.text, LocalLLM(), examples, ids, receipt_id, ocr.engine, ocr_log)
    path = save_extraction(record, new_run_dir("extraction"))

    g, o = record.generation, record.outcome
    print(f"\n--- raw model output (finish_reason={g.finish_reason})\n{g.raw_output}")
    print(f"\n--- repairs: {o.repairs or '-'}")
    print(f"--- failures: {[f.model_dump(exclude_none=True) for f in o.failures] or '-'}")
    print(f"--- schema_valid={o.schema_valid} strict_valid={o.strict_valid}")
    print(f"--- lenient result:\n{json.dumps(o.lenient, indent=2, ensure_ascii=False)}")
    print(f"\n--- tokens: prompt {g.prompt_tokens}, completion {g.completion_tokens}; "
          f"llm load {record.llm['load_s']:.1f}s, generate {g.seconds:.1f}s")
    print(f"--- log: {path.as_posix()}")


if __name__ == "__main__":
    main()
