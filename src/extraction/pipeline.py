"""OCR text -> prompt -> LLM -> parsed and validated result, with a full log."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .llm import Generation, LocalLLM
from .parse import ParseOutcome, parse_output
from .prompts import PROMPT_VERSION, build_messages, to_phi3


REPO_ROOT = Path(__file__).resolve().parents[2]


def _rel(path):
    path = Path(path).resolve()
    return (path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path).as_posix()


class ExtractionRecord(BaseModel):
    receipt_id: str
    created: str
    ocr_engine: str | None = None
    ocr_log: str | None = None  # run log the OCR text came from
    prompt_version: str
    fewshot_ids: list[str]
    messages: list[dict]  # exactly what was sent to the model
    llm: dict  # model file and settings
    generation: Generation  # includes the raw model output
    outcome: ParseOutcome  # parsed value, repairs, failures, validity, lenient result


def extract(ocr_text, llm: LocalLLM, examples=(), fewshot_ids=(), receipt_id="receipt",
            ocr_engine=None, ocr_log=None) -> ExtractionRecord:
    messages = to_phi3(build_messages(ocr_text, examples))
    generation = llm.generate(messages)
    return ExtractionRecord(
        receipt_id=receipt_id,
        created=datetime.now().isoformat(timespec="seconds"),
        ocr_engine=ocr_engine,
        ocr_log=_rel(ocr_log) if ocr_log else None,
        prompt_version=PROMPT_VERSION,
        fewshot_ids=list(fewshot_ids),
        messages=messages,
        llm=llm.info(),
        generation=generation,
        outcome=parse_output(generation.raw_output, generation.finish_reason),
    )


def save_extraction(record: ExtractionRecord, run_dir) -> Path:
    """Write <run_dir>/<receipt_id>.<ocr_engine>.json."""
    path = Path(run_dir) / f"{record.receipt_id}.{record.ocr_engine or 'text'}.json"
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=1, ensure_ascii=False),
                    encoding="utf-8")
    return path
