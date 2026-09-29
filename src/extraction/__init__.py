"""Extraction stage: OCR text -> Phi-3 Mini (llama-cpp, CPU) -> validated Receipt JSON.

    from src.extraction import LocalLLM, extract
    record = extract(ocr_text, LocalLLM(), examples)
    record.generation.raw_output, record.outcome.failures, record.outcome.lenient
"""

from .llm import Generation, LLMSettings, LocalLLM
from .parse import Failure, ParseOutcome, parse_output
from .pipeline import ExtractionRecord, extract, save_extraction
from .prompts import PROMPT_VERSION, build_messages, to_phi3
from .schema import FIELDS, Receipt

__all__ = [
    "FIELDS", "PROMPT_VERSION", "ExtractionRecord", "Failure", "Generation", "LLMSettings",
    "LocalLLM", "ParseOutcome", "Receipt", "build_messages", "extract", "parse_output",
    "save_extraction", "to_phi3",
]
