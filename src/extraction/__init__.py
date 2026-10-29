"""Extraction stage: OCR text -> Phi-3 Mini (llama-cpp, CPU) -> validated Receipt JSON.

    from src.extraction import LocalLLM, build_messages, parse_output, to_phi3
    gen = LocalLLM().generate(to_phi3(build_messages(ocr_text, examples)))
    outcome = parse_output(gen.raw_output, gen.finish_reason)

The end-to-end command (OCR -> LLM -> JSON, single image or batch) is src/run.py.
"""

from .grammar import all_null_json, grammar_gbnf, load_grammar
from .llm import DEFAULT_MODEL_NAME, MODELS, Generation, LLMSettings, LocalLLM, resolve_model
from .parse import Failure, ParseOutcome, parse_output
from .prompts import PROMPT_VERSION, PROMPT_VERSIONS, PROMPTS, build_messages, to_phi3
from .schema import FIELDS, Receipt

__all__ = [
    "DEFAULT_MODEL_NAME", "FIELDS", "MODELS", "PROMPT_VERSION", "PROMPT_VERSIONS", "PROMPTS",
    "Failure", "Generation", "LLMSettings", "LocalLLM", "ParseOutcome", "Receipt", "all_null_json",
    "build_messages", "grammar_gbnf", "load_grammar", "parse_output", "resolve_model", "to_phi3",
]
