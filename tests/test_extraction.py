"""LLM-free tests: lenient parsing, failure recording, prompt building, few-shot selection."""

import pytest

from src.extraction import build_messages, parse_output, to_phi3
from src.extraction.fewshot import eligible_train_ids, select_fewshot_ids

GOOD = '{"company": "OJC MARKETING SDN BHD", "date": "15/01/2019", "address": "NO 2 & 4", "total": "193.00"}'


def types(outcome):
    return sorted((f.type, f.field) for f in outcome.failures)


def test_clean_json_is_strictly_valid():
    o = parse_output(GOOD, "stop")
    assert o.strict_valid and o.schema_valid and not o.failures and not o.repairs
    assert o.lenient["total"] == "193.00"


@pytest.mark.parametrize("raw, repair", [
    (f"```json\n{GOOD}\n```", "code_fence"),
    (f"Here is the JSON:\n{GOOD}\nHope this helps!", "extra_text"),
    (GOOD.replace('"193.00"}', '"193.00",}'), "trailing_comma"),
    (GOOD.replace('"', "'"), "python_literal"),
])
def test_repaired_output_is_schema_valid_but_not_strict(raw, repair):
    o = parse_output(raw, "stop")
    assert repair in o.repairs
    assert o.schema_valid and not o.strict_valid
    assert o.lenient["company"] == "OJC MARKETING SDN BHD"


def test_no_json():
    o = parse_output("This appears to be a Tax Invoice for OJC MARKETING.", "length")
    assert types(o) == [("no_json", None), ("truncated", None)]
    assert o.lenient is None and not o.schema_valid


def test_truncated_object_is_invalid_json():
    o = parse_output('{"company": "OJC", "date": "15/01', "length")
    assert types(o) == [("invalid_json", None), ("truncated", None)]


def test_missing_wrong_type_extra_and_null_fields():
    raw = '{"company": "OJC", "date": null, "total": 193.0, "tax": "0.00"}'
    o = parse_output(raw, "stop")
    assert types(o) == [("extra_field", "tax"), ("missing_field", "address"),
                        ("null_field", "date"), ("wrong_type", "total")]
    assert not o.schema_valid
    assert o.lenient == {"company": "OJC", "date": None, "address": None, "total": "193.0"}


def test_null_fields_alone_are_schema_valid():
    o = parse_output('{"company": null, "date": "", "address": null, "total": "60.000"}', "stop")
    assert o.schema_valid and o.strict_valid
    assert types(o) == [("null_field", "address"), ("null_field", "company"), ("null_field", "date")]


def test_list_of_objects_is_not_object_but_still_scored():
    o = parse_output(f"[{GOOD}]", "stop")
    assert ("not_object", None) in types(o) and not o.schema_valid
    assert o.lenient["total"] == "193.00"


def test_address_as_list_of_lines_is_joined_leniently():
    raw = '{"company": "A", "date": "1/1/2018", "address": ["NO 2", "JOHOR"], "total": "1.00"}'
    o = parse_output(raw, "stop")
    assert types(o) == [("wrong_type", "address")]
    assert o.lenient["address"] == "NO 2 JOHOR"


def test_prompt_too_long():
    assert types(parse_output(None, "prompt_too_long")) == [("prompt_too_long", None)]


def test_system_prompt_folded_into_first_user_turn():
    msgs = to_phi3(build_messages("TARGET", [("EX1", {"company": "A", "date": None, "address": None, "total": "1"})]))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"].startswith("Extract four fields") and msgs[0]["content"].endswith("EX1")
    assert msgs[1]["content"] == '{"company": "A", "date": null, "address": null, "total": "1"}'
    assert msgs[2]["content"].endswith("TARGET") and "Extract" not in msgs[2]["content"]


def test_zero_shot_puts_instructions_before_target():
    msgs = to_phi3(build_messages("TARGET"))
    assert len(msgs) == 1 and msgs[0]["content"].startswith("Extract") and msgs[0]["content"].endswith("TARGET")


def rec(rid, split, sha, issues=(), dup=None):
    return {"id": rid, "split": split, "sha256": sha, "issues": list(issues), "duplicate_of": dup}


def test_fewshot_only_from_clean_train_not_overlapping_test():
    records = [
        rec("T1", "test", "h1"),
        rec("A", "train", "h1"),  # same image as a test receipt
        rec("B", "train", "h2", issues=["empty:total"]),
        rec("C", "train", "h3", dup="D"),
        rec("D", "train", "h3"),
        rec("E", "train", "h4"),
    ]
    assert eligible_train_ids(records) == ["D", "E"]
    assert sorted(select_fewshot_ids(records, k=2, seed=1)) == ["D", "E"]
    assert select_fewshot_ids(records, k=0) == []
    assert select_fewshot_ids(records, k=1, seed=7) == select_fewshot_ids(records, k=1, seed=7)
