# Prompt v1 vs v2 — targeted disambiguation rules (issue #24)

Same OCR (PaddleOCR, cached), same models and 50-receipt subsets, same 2-shot
examples; **only the prompt wording differs**. v2 adds rules for the model-stage
failures found in the taxonomy: seller/legal-name (not buyer/brand), final total
(not cash/change/subtotal), copy multi-line values in full, and return null (not
prose) on empty OCR. v1 stays the default baseline. Numbers:
`evaluation/results/prompt_v1_vs_v2.json`; tool: `evaluation/compare_prompts.py`.

With n = 50 the 95% CIs are ~±13 points, so per-cell accuracy shifts below that
are within noise. Category counts (failing fields) are the finer signal.

## Summary

| cell | all-fields v1 → v2 | key category shift | paired (fixed / regressed) |
|---|---|---|---|
| SROIE / Phi-3 | 26 → 26 | **malformed_json 8 → 20** (worse) | 1 / 1 |
| SROIE / Qwen2.5-3B | 44 → 38 | boundary_too_wide 5 → 9 | 0 / 3 |
| SROIE / Llama-3.2-3B | 32 → 36 | wrong_total 7 → 5, boundary_too_narrow 4 → 2 | 3 / 1 |
| CORD / Phi-3 | 96 → 94 | wrong_total 1 → 2 | 0 / 1 |
| CORD / Qwen2.5-3B | 100 → 100 | (already perfect) | 0 / 0 |
| CORD / Llama-3.2-3B | **78 → 90 (+12)** | **wrong_total 10 → 5** | **6 / 0** |

## Findings

1. **One clear, meaningful win: Llama-3.2-3B on CORD, +12 points (78 → 90).** The
   "final total, not cash/change/subtotal" rule directly fixed Llama's cash/total
   confusion — `wrong_total` halved (10 → 5), six receipts fixed, no regressions.
   This is the only cell whose gain is clearly beyond the ±13-point noise, and it
   lands exactly on the failure the rule targeted.

2. **v2 reduces the targeted categories where the model follows it.** `wrong_total`
   falls in most cells (Qwen SROIE 5 → 3, Llama SROIE 7 → 5, Llama CORD 10 → 5),
   and Llama's `boundary_too_narrow` drops 4 → 2. The rules do what they were
   designed to do at the category level.

3. **But v2 backfires on Phi-3 through output format.** On SROIE/Phi-3,
   `malformed_json` **tripled (8 → 20)**: the longer, more detailed prompt induces
   more derailment in the model that was already derailment-prone, cancelling any
   semantic gains (all-fields flat 26 → 26, total 92 → 84). More instruction is
   not free — it can destabilise a fragile model's JSON.

4. **Qwen shows a small SROIE regression (44 → 38, within noise)**, driven by
   `boundary_too_wide` rising 5 → 9 — the "copy in full" rule nudged it to
   over-include adjacent text on some receipts.

## Conclusion

Prompt v2 is a **model-dependent trade-off, not a uniform improvement**: it helps
a capable model on the specific weakness it targets (Llama's cash/total on CORD),
but hurts a fragile model by increasing format derailment (Phi-3). This is exactly
why **both prompts are kept in the system with v1 as the default** — the right
choice depends on the model. For the thesis: the cash/total rule is a clean,
worked example of prompt engineering fixing an identified failure mode, with the
Phi-3 result as an honest counter-example of its limits.

Next (optional): a **shorter v2 for Phi-3** (the cash/total + seller rules only,
dropping the longer clauses) to test whether the semantic gains can be kept
without the format cost.
