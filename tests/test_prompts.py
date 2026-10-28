"""Tests for prompt versioning: v1 (baseline) vs v2 (disambiguation rules)."""

from src.extraction import PROMPT_VERSION, PROMPT_VERSIONS, PROMPTS, build_messages, to_phi3


def test_versions_registry():
    assert PROMPT_VERSION == "v1"                     # v1 stays the default baseline
    assert set(PROMPT_VERSIONS) == {"v1", "v2"}
    assert set(PROMPTS) == {"v1", "v2"}


def test_build_messages_default_is_v1():
    assert build_messages("x")[0]["content"] == PROMPTS["v1"]


def test_v2_adds_disambiguation_rules():
    v2 = build_messages("x", version="v2")[0]["content"]
    low = v2.lower()
    assert "seller" in low                 # seller vs buyer
    assert "legal name" in low             # brand vs legal name
    assert "grand total" in low            # final total vs cash/change/subtotal
    assert "null for all four" in low      # prose-on-empty guard
    assert v2 != PROMPTS["v1"]


def test_to_phi3_folds_selected_version_into_first_user_turn():
    msgs = build_messages(
        "RECEIPT_TEXT",
        examples=[("ex ocr", {"company": "A", "date": None, "address": None, "total": "1"})],
        version="v2",
    )
    folded = to_phi3(msgs)
    assert folded[0]["role"] == "user"
    assert "seller" in folded[0]["content"].lower()    # v2 instructions folded into first user turn
    assert "RECEIPT_TEXT" in folded[-1]["content"]     # target receipt is last
