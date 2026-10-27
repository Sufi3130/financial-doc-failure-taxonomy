"""Diagnose why Tesseract does poorly on CORD, and test fixes (issue #19).

CORD receipts are phone photos (0.1-12 MP, perspective, shadows), unlike the
clean SROIE scans. Tesseract with the pipeline's default (grayscale, --psm 4)
finds the gold total in only ~34 % of the CORD subset, vs ~80 % on SROIE.

This measures, OCR-only (no LLM), how often the gold total appears in
Tesseract's output under different preprocessing and page-segmentation
settings, to see whether a setup change rescues Tesseract on CORD.

    python evaluation/datasets/CORD/diagnose_tesseract.py \
        --manifest evaluation/datasets/CORD/subset_valtest_50.jsonl --locale id

Writes evaluation/results/tesseract_diagnosis_<dataset>.json (counts only, no text).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.normalize import normalize_total  # noqa: E402

RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")

# (name, preprocessing ops, psm). Upscale targets a readable x-height; huge
# photos are also capped so Tesseract stays fast.
CONFIGS = [
    ("gray psm4 (current)", "gray", 4),
    ("gray psm6", "gray", 6),
    ("gray psm3", "gray", 3),
    ("gray psm11", "gray", 11),
    ("otsu psm4", "otsu", 4),
    ("otsu psm6", "otsu", 6),
    ("resize psm4", "resize", 4),
    ("resize psm6", "resize", 6),
    ("resize+otsu psm4", "resize_otsu", 4),
    ("resize+otsu psm6", "resize_otsu", 6),
    ("resize+adapt psm6", "resize_adapt", 6),
]


def preprocess(path, mode, min_side=1200, max_side=2600):
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if "resize" in mode:
        h, w = im.shape
        scale = min(max(min_side / min(h, w), 1.0), max_side / max(h, w))
        if abs(scale - 1.0) > 0.01:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if "otsu" in mode:
        im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif "adapt" in mode:
        im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return im


def total_in_text(text, gold_norm, locale):
    found = {normalize_total(t, locale) for t in _NUMBER.findall(text or "")} - {None}
    return gold_norm in found


def run(records, locale, lang, configs):
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    scored = [r for r in records if (r["ground_truth"]["normalized"] or {}).get("total")]
    results = []
    for name, mode, psm in configs:
        hits, seconds = 0, 0.0
        for r in scored:
            img = Image.fromarray(preprocess(REPO_ROOT / r["image"], mode))
            t0 = time.perf_counter()
            text = pytesseract.image_to_string(img, lang=lang, config=f"--oem 1 --psm {psm}")
            seconds += time.perf_counter() - t0
            hits += total_in_text(text, r["ground_truth"]["normalized"]["total"], locale)
        pct = round(100 * hits / len(scored), 1)
        results.append({"config": name, "mode": mode, "psm": psm, "total_found": hits,
                        "scored": len(scored), "total_found_pct": pct, "avg_s": round(seconds / len(scored), 2)})
        print(f"  {name:22} total in OCR {hits:>3}/{len(scored)}  ({pct:>5}%)   {seconds / len(scored):.2f}s/img")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--locale", default="id")
    parser.add_argument("--lang", default="eng", help="Tesseract language(s), e.g. eng or eng+ind")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.manifest, encoding="utf-8") if line.strip()][: args.limit]
    dataset = "CORD" if "CORD" in args.manifest.upper() else "SROIE"
    print(f"Tesseract diagnosis on {dataset} ({len(records)} receipts), lang={args.lang}, locale={args.locale}")
    print("metric: gold total found anywhere in the OCR text (OCR only, no LLM)\n")
    results = run(records, args.locale, args.lang, CONFIGS)

    best = max(results, key=lambda r: r["total_found_pct"])
    print(f"\nbest: {best['config']} at {best['total_found_pct']}% "
          f"(current 'gray psm4' = {next(r['total_found_pct'] for r in results if r['mode'] == 'gray' and r['psm'] == 4)}%)")
    out = RESULTS_DIR / f"tesseract_diagnosis_{dataset}.json"
    out.write_text(json.dumps({"dataset": dataset, "manifest": args.manifest, "lang": args.lang,
                               "locale": args.locale, "n": len(records), "configs": results,
                               "best": best}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {out.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
