"""Real-stack integration test: loads an actual GGUF model and runs one
generation, covering the model load + inference path that the fast suite
deliberately fakes.

Skipped automatically when the model file or llama-cpp are absent, so CI and a
fresh clone skip it; run it locally (with the models in `models/`) to exercise
the real path:  `pytest tests/test_integration.py`
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PHI3 = REPO / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

_skip = importlib.util.find_spec("llama_cpp") is None or not PHI3.is_file()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_skip, reason="real model / llama-cpp not present (local-only integration test)"),
]


def test_llm_loads_and_generates():
    from src.extraction import LLMSettings, LocalLLM

    llm = LocalLLM(LLMSettings(model_path=str(PHI3)))
    messages = [{"role": "user", "content":
                 'Reply with exactly this JSON and nothing else: '
                 '{"company": null, "date": null, "address": null, "total": "9.90"}'}]
    gen = llm.generate(messages, max_tokens=64)
    assert gen.raw_output and gen.finish_reason
    assert gen.prompt_tokens > 0
    assert isinstance(llm.info(), dict)
