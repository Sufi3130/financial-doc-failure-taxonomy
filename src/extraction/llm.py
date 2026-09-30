"""Local LLM inference with llama-cpp-python (CPU, greedy decoding).

No grammar / JSON-schema constrained decoding: forcing valid JSON would hide
the format failures the taxonomy measures.
"""

import time
from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "Phi-3-mini-4k-instruct-q4.gguf"


class LLMSettings(BaseModel):
    model_path: str = str(DEFAULT_MODEL)
    n_ctx: int = 4096
    n_threads: int | None = None  # None = llama.cpp default (physical cores)
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 256


class Generation(BaseModel):
    raw_output: str | None
    finish_reason: str | None  # "stop", "length", or "prompt_too_long"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float
    # llama.cpp perf counters: only the tokens actually evaluated count, so a
    # reused prompt prefix (same few-shot examples) shows up as fewer tokens
    prompt_tokens_evaluated: int | None = None
    prompt_eval_s: float | None = None
    gen_s: float | None = None


class LocalLLM:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()
        self._llm = None
        self.load_s = 0.0

    def info(self):
        path = Path(self.settings.model_path)
        return {**self.settings.model_dump(), "model_file": path.name,
                "model_bytes": path.stat().st_size if path.is_file() else None,
                "n_gpu_layers": 0, "load_s": round(self.load_s, 3)}

    def load(self):
        if self._llm is not None:
            return
        if not Path(self.settings.model_path).is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.settings.model_path} (see README)")
        from llama_cpp import Llama

        t0 = time.perf_counter()
        s = self.settings
        self._llm = Llama(model_path=s.model_path, n_ctx=s.n_ctx, n_threads=s.n_threads,
                          n_gpu_layers=0, seed=s.seed, verbose=False)
        self.load_s = time.perf_counter() - t0

    def _perf(self, reset=False):
        """llama.cpp context perf counters, or None if unavailable."""
        try:
            import llama_cpp

            ctx = self._llm._ctx.ctx
            if reset:
                llama_cpp.llama_perf_context_reset(ctx)
                return None
            return llama_cpp.llama_perf_context(ctx)
        except Exception:
            return None

    def generate(self, messages, max_tokens=None) -> Generation:
        """messages must already be in a form the chat template accepts (see prompts.to_phi3)."""
        self.load()
        s = self.settings
        self._perf(reset=True)
        t0 = time.perf_counter()
        try:
            out = self._llm.create_chat_completion(
                messages=messages, temperature=s.temperature, seed=s.seed,
                max_tokens=max_tokens or s.max_tokens)
        except ValueError as e:
            if "context window" not in str(e):
                raise
            return Generation(raw_output=None, finish_reason="prompt_too_long",
                              seconds=round(time.perf_counter() - t0, 3))
        seconds = round(time.perf_counter() - t0, 3)
        perf = self._perf()
        return Generation(
            raw_output=out["choices"][0]["message"]["content"],
            finish_reason=out["choices"][0]["finish_reason"],
            prompt_tokens=out["usage"]["prompt_tokens"],
            completion_tokens=out["usage"]["completion_tokens"],
            seconds=seconds,
            prompt_tokens_evaluated=perf.n_p_eval if perf else None,
            prompt_eval_s=round(perf.t_p_eval_ms / 1000, 3) if perf else None,
            gen_s=round(perf.t_eval_ms / 1000, 3) if perf else None,
        )
