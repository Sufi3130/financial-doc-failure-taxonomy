"""Smoke test for the local, GPU-free stack: OCR (PaddleOCR, Tesseract) -> Phi-3 Mini Q4 GGUF.

Imports all core libs, OCRs one SROIE receipt with every OCR engine, runs one
short GGUF generation and prints timings. Exits non-zero on the first failure.

Usage:
    python scripts/check_env.py [--image PATH] [--model PATH]
"""

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ocr import ENGINES, get_engine, reading_order  # noqa: E402

DEFAULT_IMAGE = REPO_ROOT / "evaluation/datasets/SROIE/raw/test/X00016469670.jpg"
DEFAULT_MODEL = REPO_ROOT / "models/Phi-3-mini-4k-instruct-q4.gguf"

LIBS = ["paddle", "paddleocr", "pytesseract", "llama_cpp", "pydantic", "numpy", "pandas", "pyarrow",
        "PIL", "tqdm", "pytest"]


def fail(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def check_imports():
    print("== Imports")
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            fail(f"import {name}: {e}")
        print(f"  {name:<10} {getattr(mod, '__version__', '?')}")

    import paddle

    if paddle.device.is_compiled_with_cuda():
        print("  [warn] paddle was built with CUDA; expected the CPU build")
    print(f"  paddle device: {paddle.device.get_device()}")


def check_ocr(image):
    """Run every registered engine; return PaddleOCR's reading-order text for the LLM check."""
    if not image.is_file():
        fail(f"receipt image not found: {image}")

    texts, timings = {}, {}
    for name in ENGINES:
        print(f"\n== OCR: {name} ({image.name})")
        try:
            result = get_engine(name).recognize(image)
        except Exception as e:
            fail(f"{name}: {e}")
        if not result.lines:
            fail(f"{name} returned no text lines")
        texts[name] = reading_order(result.lines).text
        print(f"  {result.engine} {result.engine_version}: {len(result.lines)} pieces, "
              f"first: {[ln.text for ln in result.lines[:3]]}")
        print(f"  load {result.timings['load_s']:.1f}s, ocr {result.timings['ocr_s']:.1f}s")
        timings[f"{name}_load"] = result.timings["load_s"]
        timings[f"{name}_ocr"] = result.timings["ocr_s"]
    return texts["paddleocr"], timings


def check_llm(model, ocr_text):
    print(f"\n== LLM ({model.name})")
    if not model.is_file():
        fail(f"GGUF model not found: {model} (see README 'Download the model')")

    from llama_cpp import Llama

    t0 = time.perf_counter()
    llm = Llama(model_path=str(model), n_ctx=2048, n_gpu_layers=0, verbose=False)
    t_load = time.perf_counter() - t0

    receipt = "\n".join(ocr_text.splitlines()[:15])
    messages = [
        {"role": "user", "content": f"Receipt text:\n{receipt}\n\nWhat is the store name? Answer in a few words."},
    ]

    t0 = time.perf_counter()
    out = llm.create_chat_completion(messages=messages, max_tokens=32, temperature=0.0)
    t_gen = time.perf_counter() - t0

    answer = out["choices"][0]["message"]["content"].strip()
    n_tokens = out["usage"]["completion_tokens"]
    if not answer:
        fail("LLM returned an empty completion")

    print(f"  answer: {answer!r}")
    print(f"  model load {t_load:.1f}s, generation {t_gen:.1f}s ({n_tokens} tokens, {n_tokens / t_gen:.1f} tok/s)")
    return {"llm_load": t_load, "llm_gen": t_gen}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Python {platform.python_version()} on {platform.platform()} ({platform.processor()})\n")

    t_start = time.perf_counter()
    check_imports()
    text, timings = check_ocr(args.image)
    timings |= check_llm(args.model, text)

    print("\n== Timings (s)")
    for k, v in timings.items():
        print(f"  {k:<15} {v:6.1f}")
    print(f"  {'total':<15} {time.perf_counter() - t_start:6.1f}")
    print("\n[OK] environment check passed")


if __name__ == "__main__":
    main()
