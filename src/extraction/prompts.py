"""Prompt template: instructions + few-shot examples + the target receipt.

Phi-3's GGUF chat template has no system role: it silently drops `system`
messages (verified: the instructions never reached the model). `to_phi3()`
therefore folds the system text into the first user turn. Keep examples
before the target so the prompt prefix is identical across receipts and
llama-cpp can reuse it.
"""

import json

from .schema import FIELDS

PROMPT_VERSION = "v1"

SYSTEM = """Extract four fields from the OCR text of a receipt and return them as JSON.

Return only one JSON object with exactly these keys:
- "company": the name of the business that issued the receipt
- "date": the date of the purchase
- "address": the address of the business
- "total": the final total amount paid

Rules:
- Copy each value exactly as it appears in the text (same spelling, punctuation and number format).
- Use null for a field that does not appear in the text.
- Do not add other keys, comments or explanations."""


def user_turn(ocr_text):
    return f"Receipt OCR text:\n{ocr_text.strip()}"


def answer_turn(answer):
    return json.dumps({f: answer.get(f) for f in FIELDS}, ensure_ascii=False)


def build_messages(ocr_text, examples=()):
    """examples: sequence of (ocr_text, answer dict). Returns chat messages with
    a `system` message first (see to_phi3 for what is actually sent)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_text, ex_answer in examples:
        messages.append({"role": "user", "content": user_turn(ex_text)})
        messages.append({"role": "assistant", "content": answer_turn(ex_answer)})
    messages.append({"role": "user", "content": user_turn(ocr_text)})
    return messages


def to_phi3(messages):
    """Fold system messages into the first user message."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest:
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    return rest
