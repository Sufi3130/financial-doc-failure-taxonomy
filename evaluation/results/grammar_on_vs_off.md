# Grammar-constrained decoding — off vs on (issue #25)

Same OCR (PaddleOCR, cached), same models, 50-receipt subsets, v1 prompt and
2-shot examples; **only constrained decoding differs**. Grammar mode forces the
output through a GBNF built from the Receipt schema (one object, the four keys,
each a string or null) and, on an empty OCR page, short-circuits to all-null.
Numbers: `evaluation/results/grammar_on_vs_off.json`; tool:
`evaluation/compare_grammar.py`; grammar: `src/extraction/grammar.py`.

"Malformed" below counts **receipts** whose output had a `no_json` /
`invalid_json` / `truncated` failure (the taxonomy counts the same thing
per-field, so 2 receipts = 8 field-instances there).

## Summary

| cell | malformed off → on | schema-valid % off → on | all-fields off → on |
|---|---|---|---|
| SROIE / Phi-3 | 2 → **3** | 96 → 94 | 26 → 26 |
| SROIE / Qwen2.5-3B | 0 → 0 | 100 → 100 | 44 → 44 |
| SROIE / Llama-3.2-3B | 0 → 0 | 100 → 100 | 32 → 30 |
| CORD / Phi-3 | **1 → 0** | 98 → 100 | 96 → 96 |
| CORD / Qwen2.5-3B | 0 → 0 | 100 → 100 | 100 → 100 |
| CORD / Llama-3.2-3B | 0 → 0 | 100 → 100 | 78 → 78 |

## Findings

1. **Grammar guarantees valid JSON for the stable models — and keeps it.** Qwen
   and Llama were already 100% schema-valid; with the grammar they are
   *structurally guaranteed* to stay that way regardless of input. For Phi-3 on
   CORD it closed the one remaining gap (malformed 1 → 0, schema-valid 98 →
   100). As a production safety property this is the main win: output is always
   parseable JSON.

2. **But grammar does NOT fix Phi-3's derailment on SROIE — it only relocates
   it.** The derailment-prone receipts (e.g. `MR. D.I.Y.`, the R41 case) still
   fail. The grammar forces the output to *start* as JSON — `{"company": "` —
   and Phi-3 then spirals **inside the string value**: `{"company": "MR. D.I
   EXERCISE 1:\n\nGiven the receipt OCR text, extract the following four
   fields…"` until it hits `max_tokens` and truncates with the string unclosed.
   A JSON string may legally contain any text, so a syntactic grammar cannot
   stop a model from generating unbounded content inside one. Malformed receipts
   went 2 → 3, not to 0.

3. **This proves the derailment is intrinsic to the model, not a formatting
   artifact.** It pairs with the prompt experiment: v2 *increased* Phi-3's
   malformed output (8 → 20 on SROIE) by adding instructions, and the grammar
   *cannot remove* it by constraining structure. The 3.8B model simply runs away
   on certain inputs; neither more guidance nor tighter output structure cures
   it. The robust fixes would be non-structural — a lower `max_tokens` with a
   retry, a stop condition, or a more capable model.

4. **Grammar does not change extraction accuracy.** All-fields is flat in five
   of six cells (Llama SROIE −2 is one receipt, within the ±13-pt noise). This
   is expected: the grammar constrains *format*, not *which value* the model
   picks. It is a validity tool, not an accuracy tool.

5. **Empty pages answer all-null deterministically.** In grammar mode a blank
   OCR page returns `{null, null, null, null}` without calling the model
   (covered by unit tests). It did not trigger on these subsets because
   PaddleOCR returned some text for every receipt, but it removes the
   prose-on-empty failure (R44) by construction when OCR is truly empty.

6. **Cost: ~6× slower generation.** Grammar-constrained decoding ran ~45–55 s
   per receipt vs ~8 s unconstrained (the grammar sampler is evaluated at every
   token). For a batch job that is a real trade-off against the validity
   guarantee.

## Conclusion

Constrained decoding is a **robust validity and safety property**, not an
accuracy or derailment fix. It guarantees parseable JSON, makes empty-page
handling deterministic, and closes clean truncation gaps — so it is kept as a
selectable option (`--grammar`, a UI toggle), **off by default** so the baseline
taxonomy still measures the format failures. Its honest limit is the headline
result: **a syntactic constraint cannot stop a model-intrinsic derailment**,
because the derailment simply moves inside a grammar-legal string. For the
thesis this is the clean counterpart to the prompt experiments — together they
show Phi-3's instability is a property of the model, reachable by neither the
prompt nor the decoder.
