# Cross-Paper Synthesis

Notes comparing the four papers above, written after finishing all four.

## Failure-category overlap

| Category | OCBC (multi-stage) | Fiscal LLM (Karnataka) | VAREX | SOB |
|---|---|---|---|---|
| Parse / invalid JSON | | | (implied) | Parse failures |
| Wrong structure/schema | | Structural mismatch (TEDS) | Schema echo | Schema violations |
| Wrong value, right shape | | Numerical consistency failure | (not separately named) | Value errors — called the dominant gap |
| Missing/omitted field | | | Under-extraction | Missing paths |
| Right value, wrong type | | | | Type mismatches |
| Domain terminology ambiguity | Inconsistent terminology | | | |
| Numeric scale/unit ambiguity | Currency unit ambiguity | | | |
| OCR keyword omission | OCR errors | (image-only, no OCR used) | | (out of scope, text-only) |

"Missing field," "wrong value," and "wrong structure" recur independently
across three of the four papers, which suggests these three are the safest
core categories to build a taxonomy around. Schema echo and under-extraction
are more specific to small models, which matters since I'm testing small
models specifically. The two domain-specific rows (terminology, currency
units) only show up in the one paper actually working with real financial
documents — the two general-purpose benchmarks never encounter this kind of
ambiguity because their test data doesn't have it. That's a fairly direct
argument for why a financial-specific taxonomy adds something the existing
general benchmarks don't cover.

## Architecture comparison against my own proposal

| | OCBC (multi-stage) | Fiscal LLM (Karnataka) | My proposed design |
|---|---|---|---|
| Vision/OCR stage | Pre-processing + PaddleOCR v3 | None — image fed directly to LLM | PaddleOCR/TrOCR, + PaddleOCR-VL as ablation |
| Retrieval/filtering | BM25 keyword retrieval | Sequential context carry-forward, no filtering | None currently |
| Extraction model | Compact VLM (8B, GPU) | Frontier LLM via API (Gemini 2.5 Pro) | Small quantized local LLM (Phi-3 Mini / Mistral 7B), CPU |
| Output format | Structured fields | 5 CSV schema types | JSON |
| Failure handling | Qualitative, 3 categories, no counts | No taxonomy — internal consistency validation | Structured, quantified taxonomy |
| GPU requirement | Yes | Yes (cloud API) | No |

Neither of the two closest financial-extraction papers actually meets a
no-GPU, no-cloud constraint — the OCBC paper needs an A100, the Karnataka
paper depends on a hosted frontier model. Worth stating plainly in the
thesis that this deployment constraint isn't demonstrated yet in the
closest existing financial-document extraction work I could find.
