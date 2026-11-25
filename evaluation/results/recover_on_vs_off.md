# Derailment recovery: recover off vs on (issue #33)

Grammar-constrained decoding (#25/#32) guarantees valid JSON for the stable
models but cannot stop Phi-3's intrinsic derailment: it relocates the spiral
inside the string value until `max_tokens`, so the output is still truncated.
A syntactic constraint cannot bound string *content*. Issue #33 adds a
**non-structural** recovery instead: detect the unusable output and recover from
it rather than trying to prevent it.

`--recover` (opt-in, off by default so the baseline keeps measuring
`malformed_json`): when an output is unusable (empty lenient parse), retry once
with a stricter "JSON only" prompt and a tighter `max_tokens`; if the retry is
still unusable, fall back to the all-null object. It fires only on genuinely
unusable output, so it never discards fields a partly-usable output salvaged.

## Phi-3, SROIE, 50-receipt subset (PaddleOCR, v1, 2-shot)

| Metric | recover off | recover on |
|---|---|---|
| `malformed_json` | 2 | **0** |
| Schema-valid rate | 96.0 % | **100.0 %** |
| All-4-fields accuracy | 26.0 | 26.0 |
| Receipts recovered | — | 3 (retry 1, fallback 2) |

Reproduce: `python -m src.run --manifest evaluation/datasets/SROIE/subset_test_50.jsonl
--engine paddleocr --model phi3 --prompt v1 --shots 2 --seed 42 --recover`
(into `runs/pipeline/exp-recover-phi3-SROIE`), then
`python evaluation/compare_recover.py`.

## Full SROIE test split (n = 347)

The subset above satisfies the issue's `compare_grammar.py`-style acceptance, but
the thesis headline reports the full test split, where Phi-3/SROIE has
`malformed_json = 44` field-instances. Those come from **11 unusable receipts**
(11 x 4 fields) in the committed recover-off baseline. Re-running exactly those
11 with `--recover`:

| | recover off (baseline) | recover on |
|---|---|---|
| Unusable receipts | 11 | **0** |
| `malformed_json` field-instances | 44 | **0** |
| Schema-valid | — | 11 / 11 |

All 11 end schema-valid: **8 were recovered** (3 by the retry, 5 by the all-null
fallback) and **3 did not re-derail** on this run (llama.cpp CPU decoding is
mildly non-deterministic even at temperature 0, so a borderline receipt can land
differently). Recovery removes every derailment it encounters, so the full-test
`malformed_json` goes to 0 regardless of which receipts derail on a given run.
(Only the derailed receipts need re-running: every other output is byte-identical
with recovery on or off, since recovery fires only on an unusable output.)

## Findings

1. **Recovery removes the malformed output: `malformed_json` 2 → 0, schema-valid
   96 % → 100 %.** The three unusable Phi-3 outputs were resolved — one by the
   stricter retry (the terse "JSON only" prompt stopped the spiral and produced
   a parseable record), two by the all-null fallback where the retry also
   derailed.
2. **No accuracy regression.** All-4-fields stays 26.0 and no receipt flipped
   correct/wrong (`recover_fixed 0, recover_broke 0`): the derailed receipts were
   already all-wrong, so making their output valid loses nothing and may only
   help (the one retry recovered a parseable, if not fully correct, record).
3. **It succeeds where grammar failed.** Grammar left Phi-3's SROIE derailment in
   place (malformed 2 → 3; see `grammar_on_vs_off.md`); this non-structural
   detect-and-recover drives it to 0. The clean counterpart to the grammar
   result: a *syntactic* constraint cannot stop a model-intrinsic derailment, but
   a *post-hoc* retry-and-fallback can bound its effect.

The feature is opt-in: the default pipeline still emits and measures
`malformed_json`, which the taxonomy depends on; `--recover` is the mitigation a
deployment would turn on.
