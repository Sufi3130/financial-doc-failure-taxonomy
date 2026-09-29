"""Smoke test for the local, GPU-free stack: PaddleOCR -> Phi-3 Mini Q4 GGUF.

Imports all core libs, OCRs one SROIE receipt, runs one short GGUF generation
and prints timings. Exits non-zero on the first failure.

Usage:
    python scripts/check_env.py [--image PATH] [--model PATH]
"""

import argparse
import importlib
import os
import platform
import sys
import time
from pathlib import Path

# Skip PaddleX's online model-host check on every run
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "evaluation/datasets/SROIE/raw/test/X00016469670.jpg"
DEFAULT_MODEL = REPO_ROOT / "models/Phi-3-mini-4k-instruct-q4.gguf"

LIBS = ["paddle", "paddleocr", "llama_cpp", "pydantic", "numpy", "pandas", "PIL", "tqdm", "pytest"]


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
    print(f"\n== OCR ({image.name})")
    if not image.is_file():
        fail(f"receipt image not found: {image}")

    from paddleocr import PaddleOCR

    t0 = time.perf_counter()
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="en",
        enable_mkldnn=False,
    )
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = ocr.predict(str(image))
    t_ocr = time.perf_counter() - t0

    texts = [t for res in result for t in res["rec_texts"]]
    if not texts:
        fail("OCR returned no text lines")

    print(f"  lines: {len(texts)}  first: {texts[:3]}")
    print(f"  model load {t_load:.1f}s, inference {t_ocr:.1f}s")
    return texts, {"ocr_load": t_load, "ocr_infer": t_ocr}


def check_llm(model, ocr_texts):
    print(f"\n== LLM ({model.name})")
    if not model.is_file():
        fail(f"GGUF model not found: {model} (see README 'Download the model')")

    from llama_cpp import Llama

    t0 = time.perf_counter()
    llm = Llama(model_path=str(model), n_ctx=2048, n_gpu_layers=0, verbose=False)
    t_load = time.perf_counter() - t0

    receipt = "\n".join(ocr_texts[:15])
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
    texts, timings = check_ocr(args.image)
    timings |= check_llm(args.model, texts)

    print("\n== Timings (s)")
    for k, v in timings.items():
        print(f"  {k:<10} {v:6.1f}")
    print(f"  {'total':<10} {time.perf_counter() - t_start:6.1f}")
    print("\n[OK] environment check passed")


if __name__ == "__main__":
    main()
