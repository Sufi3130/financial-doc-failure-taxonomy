# Few-shot ablation — 2-shot vs 0-shot (issue #24)

Same OCR (PaddleOCR, cached), same models and 50-receipt subsets, same v1 prompt;
**only the few-shot examples differ** (2 worked examples → 0). This isolates what
the examples contribute. Numbers: `evaluation/results/fewshot_2_vs_0.json`; tool:
`evaluation/compare_fewshot.py`.

With n = 50 the 95% CIs are ~±13 points, so a single cell's accuracy shift below
that is within noise — but the direction is the signal: **all six cells drop**,
and the paired counts show far more receipts broken than fixed in every cell.

## Summary

| cell | all-fields 2-shot → 0-shot | paired (fixed / broke) | key category shift |
|---|---|---|---|
| SROIE / Phi-3 | 26 → 18 (−8) | 1 / 5 | boundary_too_narrow 4 → **15** |
| SROIE / Qwen2.5-3B | 44 → 28 (−16) | 1 / 9 | wrong_total 5 → 9 |
| SROIE / Llama-3.2-3B | 32 → 22 (−10) | 2 / 7 | wrong_total 7 → **13** |
| CORD / Phi-3 | 96 → 82 (−14) | 2 / 9 | wrong_total 1 → **9**, schema_valid 98 → 84 |
| CORD / Qwen2.5-3B | 100 → 76 (−24) | 0 / 12 | wrong_total 0 → **11** |
| CORD / Llama-3.2-3B | 78 → 72 (−6) | 2 / 5 | wrong_total 10 → 11 |

## Findings

1. **Removing the examples hurts every model on every subset.** The drop ranges
   from −6 (Llama CORD) to −24 points (Qwen CORD), and in every cell the paired
   count breaks more receipts than it fixes (worst: Qwen CORD, 0 fixed / 12 broke).
   There is no cell where 0-shot is better. The two examples are doing real work.

2. **The examples mainly teach field *semantics*, and that is where 0-shot fails.**
   `wrong_total` rises almost everywhere without them — most sharply on CORD, where
   it goes 0 → 11 (Qwen) and 1 → 9 (Phi-3). Without an example showing that the
   target is the final payable total (not cash, change, or subtotal), the models
   fall back to grabbing the wrong line. This is the same failure the v2 *prompt*
   rule targeted — here the *examples* were silently doing that job already.

3. **On SROIE the examples also teach output shape.** Phi-3's
   `boundary_too_narrow` explodes 4 → 15 with no example to show a full multi-line
   address copied in full, so it under-copies. Llama picks up `malformed_json`
   0 → 4 and `boundary_too_wide` 0 → 4. The worked example anchors the structure,
   not just the content.

4. **The feared downside of few-shot — example copying — does not occur here.**
   The `fewshot_copy` tag (model echoing an example's answer instead of reading the
   receipt) is **0 in both the 2-shot and the 0-shot runs**, across all six cells.
   So in this pipeline the examples give the accuracy gain in finding 1 *without*
   inducing copying — there is no offsetting cost to weigh against them.

5. **Format stability degrades too.** `schema_valid_rate` drops on CORD/Phi-3
   (98 → 84) and slightly on the Llama/SROIE cells — the examples help the smaller,
   derailment-prone models hold the JSON shape, not only the field values.

## Conclusion

The 2-shot examples are **uniformly beneficial and carry no measured copying
cost** in this pipeline: every cell is worse without them, driven by the models
losing the "which total / full value" semantics and (for the fragile models) the
output format. This validates keeping 2-shot as the default configuration.

It also reframes the v2 prompt result: the cash/total disambiguation that v2 added
in *words* was already partly supplied by the *examples* — which is why 0-shot
regresses on exactly `wrong_total`, the same category v2 improved. For the thesis,
the two experiments together make the point cleanly: for these small models,
**concrete worked examples are a stronger lever than longer instructions**, and
they do not trigger the copying failure that few-shot prompting is often warned
about.
