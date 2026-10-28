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

# v2: adds targeted disambiguation rules for the model-stage failures found in the
# taxonomy (seller vs buyer / brand vs legal name, final total vs cash/change/subtotal,
# full multi-line values, and null-not-prose on empty OCR). Same structure as v1 so the
# comparison isolates the wording; v1 stays the default baseline.
SYSTEM_V2 = """Extract four fields from the OCR text of a receipt and return them as JSON.

Return only one JSON object with exactly these keys:
- "company": the business that issued the receipt (the seller or merchant). If both a brand or
  trading name and a registered legal name appear (for example one ending in SDN BHD, BHD, S/B or
  PLT), use the registered legal name. Never use the customer or buyer, and never a product or
  brand name on its own.
- "date": the date of the purchase.
- "address": the address of the issuing business. Copy it in full even when it spans several lines.
- "total": the final amount payable (the grand total). Not the cash tendered, the change, or the
  subtotal.

Rules:
- Copy each value exactly as it appears in the text (same spelling, punctuation and number format).
- Do not add any text that is not part of the field.
- Use null for a field that does not appear in the text. If the text is empty or unreadable, return
  null for all four fields; never write a sentence, an apology or an explanation.
- Do not add other keys, comments or explanations."""

PROMPTS = {"v1": SYSTEM, "v2": SYSTEM_V2}
PROMPT_VERSIONS = tuple(PROMPTS)


def user_turn(ocr_text):
    return f"Receipt OCR text:\n{ocr_text.strip()}"


def answer_turn(answer):
    return json.dumps({f: answer.get(f) for f in FIELDS}, ensure_ascii=False)


def build_messages(ocr_text, examples=(), version=PROMPT_VERSION):
    """examples: sequence of (ocr_text, answer dict). `version` selects the
    instruction text ("v1" default, "v2" adds disambiguation rules). Returns chat
    messages with a `system` message first (see to_phi3 for what is actually sent)."""
    messages = [{"role": "system", "content": PROMPTS[version]}]
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
