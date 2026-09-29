"""Run OCR on one image and save the run log.

Usage (from the repo root):
    python -m src.ocr IMAGE [--engine paddleocr|tesseract|all] [--quiet]
"""

import argparse

from . import ENGINES, get_engine, new_run_dir, reading_order, save_run_log


def main():
    parser = argparse.ArgumentParser(prog="python -m src.ocr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--engine", default="all", choices=[*ENGINES, "all"])
    parser.add_argument("--quiet", action="store_true", help="skip the per-line table")
    args = parser.parse_args()

    names = list(ENGINES) if args.engine == "all" else [args.engine]
    run_dir = new_run_dir("ocr")
    for name in names:
        result = get_engine(name).recognize(args.image)
        reading = reading_order(result.lines)
        log = save_run_log(result, reading, run_dir)

        confs = [ln.confidence for ln in result.lines]
        print(f"\n===== {result.engine} {result.engine_version}  {result.settings}")
        print("--- reading-order text (LLM prompt input)")
        print(reading.text)
        if not args.quiet:
            print("--- pieces: index  confidence  box (x1,y1,x2,y2)  text")
            for ln in result.lines:
                print(f"  {ln.index:3}  {ln.confidence:.3f}  {str(ln.box):22}  {ln.text}")
        print(f"--- {len(result.lines)} pieces in {len(reading.rows)} rows; "
              f"confidence min {min(confs, default=0):.3f} / mean {sum(confs) / max(len(confs), 1):.3f}; "
              f"load {result.timings['load_s']:.1f}s, ocr {result.timings['ocr_s']:.1f}s")
        print(f"--- run log: {log.as_posix()}")


if __name__ == "__main__":
    main()
