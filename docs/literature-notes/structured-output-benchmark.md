# The Structured Output Benchmark (SOB)

**Authors:** Singh, Khurdula, Khemlani, Agarwal (JigsawStack Inc.)
**Venue:** arXiv preprint, Apr 2026
**Link:** https://arxiv.org/pdf/2604.25359

## Summary

Like VAREX, this is a benchmark rather than a document pipeline, but it
covers three source modalities — native text, OCR-processed document
images, and audio meeting transcripts — with all three converted to plain
text before reaching the model. This is a deliberate choice: by removing
raw images/audio from the model's input, they isolate "can this model
extract correct values from text" from "can this model see/hear well,"
so a model's vision or speech quality can't confound its structured-output
score.

They evaluate 21 models on seven metrics per response, not just accuracy:
JSON Pass Rate, Value Accuracy (exact match), Faithfulness (partial-credit
token overlap), Path Recall, Structure Coverage, Type Safety, and Perfect
Response Rate.

## Error taxonomy (five categories, ranked by production severity)

1. **Parse failures** — invalid JSON syntax. Rare for capable models (most
   under 2%), but not negligible — one 20B model hit 13.2% on their text
   set.
2. **Schema violations** — valid JSON, but missing required fields or wrong
   nesting.
3. **Value errors** — correct structure, wrong content. This is described
   as the dominant gap: even models above 97% schema compliance still get
   17–31% of individual field values wrong. Structural correctness on its
   own says almost nothing about whether the actual data is right.
4. **Missing paths** — fields omitted entirely (as opposed to a wrong value
   being present).
5. **Type mismatches** — right value, wrong JSON type (e.g. the number 42
   returned as the string "42"). Kept separate from value errors because
   this is a formatting slip a downstream system can often auto-correct,
   unlike a genuinely wrong number.

Their single biggest finding: JSON Pass Rate stays uniformly high across
essentially every model tested, while Value Accuracy trails it by 15–25
points consistently. A response can look completely valid and still be
wrong where it actually matters.

## Relevance to my thesis

This is the cleanest, most citable five-category skeleton of everything I
read, and I plan to use a version of it (possibly with domain-specific
categories added, similar to the terminology/currency ambiguities the OCBC
paper reports) as the starting structure for my own taxonomy rather than
inventing categories from scratch.

None of their 209 image records are financial documents specifically, and
none of the 21 models they test are in the sub-4B, quantized, CPU-only
range I'm targeting — their finding that model size does not predict
structured-output quality (a 14B model beats GPT-5 on their Value Accuracy
metric) is a useful citation for justifying why testing small models
specifically is worthwhile rather than just assuming smaller is worse.

## Citations worth following up
- Ferguson et al. 2026, ExtractBench — closest prior work; reports frontier
  models achieving only 4.6% field-level pass rate on PDF-to-JSON
  extraction, a striking number worth citing directly.
- Tenckhoff et al. 2026, LLMStructBench — finds prompting strategy matters
  more than model size; relevant for justifying testing multiple prompt
  formats rather than locking in one.
- Wang et al. 2025, STED — Semantic Tree Edit Distance metric giving partial
  credit for semantically equivalent values (e.g. "USA" vs "United
  States"); worth considering alongside strict exact-match scoring, since
  financial fields like dates and amounts often have more than one valid
  textual form.
