"""Extraction stage: OCR text -> Phi-3 Mini (llama-cpp, CPU) -> validated Receipt JSON.

    from src.extraction import LocalLLM, build_messages, parse_output, to_phi3
    gen = LocalLLM().generate(to_phi3(build_messages(ocr_text, examples)))
    outcome = parse_output(gen.raw_output, gen.finish_reason)

The end-to-end command (OCR -> LLM -> JSON, single image or batch) is src/run.py.
"""

from .llm import Generation, LLMSettings, LocalLLM
from .parse import Failure, ParseOutcome, parse_output
from .prompts import PROMPT_VERSION, build_messages, to_phi3
from .schema import FIELDS, Receipt

__all__ = [
    "FIELDS", "PROMPT_VERSION", "Failure", "Generation", "LLMSettings", "LocalLLM",
    "ParseOutcome", "Receipt", "build_messages", "parse_output", "to_phi3",
]
