# Paper: The Structured Output Benchmark (SOB)
**Link:** https://arxiv.org/pdf/2604.25359
**Authors:** Singh et al., JigsawStack Inc.
**Pass completed:** 3 (full)

### Architecture
Like VAREX, this is a benchmark/evaluation methodology, not a document pipeline. Structure:
- Three source modalities tested: native text, OCR-processed images, and audio transcripts — but **all three are converted to plain text before reaching the model**. *Why this design choice matters for you:* they deliberately don't give models raw images or audio, specifically so that a model's extraction score isn't confounded by how good its vision/speech processing is — they're isolating "can it extract correct values from text" as a separate skill from "can it read an image." This is a genuinely useful idea for your own evaluation design: if you want to know whether Phi-3 Mini's failures come from bad OCR text vs. bad reasoning over correct text, feeding it clean ground-truth text (bypassing OCR entirely) as a control condition would isolate the same distinction.
- Seven metrics are computed per response (not just accuracy): JSON Pass Rate, Value Accuracy, Faithfulness (partial-credit token overlap), Path Recall, Structure Coverage, Type Safety, Perfect Response Rate.

### Failure categories named
This paper gives you the cleanest, most citable five-category taxonomy of the four papers, ordered explicitly by production severity:
1. **Parse failures** — output isn't even valid JSON syntax. Rare for capable models (most under 2%) but not zero — GPT-OSS 20B hit 13.2% failures on their text set, showing even a fairly large model can fail at pure syntax.
2. **Schema violations** — valid JSON, but missing required fields or wrong nesting/structure. *Why distinct from parse failures:* the JSON itself is syntactically fine — it just doesn't match the shape the schema demands.
3. **Value errors** — correct structure, wrong content. *Why this is called "the dominant gap":* even models with over 97% schema compliance still get 17–31% of individual field values wrong — meaning structural correctness tells you almost nothing about whether the actual data is right. This is their single most important finding and the one most directly relevant to your thesis: a pipeline can look like it's "working" (valid, well-formed JSON) while still being unreliable where it actually counts.
4. **Missing paths** — fields the model just omits entirely, distinct from value errors because there's no wrong value, there's no value at all.
5. **Type mismatches** — right value, wrong JSON type (e.g. the number 42 returned as the string "42"). *Why kept separate from value errors:* this is a formatting/serialization slip rather than a content mistake — a downstream system can often auto-coerce "42" to 42, but can't auto-correct a genuinely wrong number, so the two need different remediation and shouldn't be double-counted as the same severity.

### What I'd do differently
Their five categories are for structured-output tasks in general (QA, meeting transcripts, OCR'd PDFs of miscellaneous types) — none of their 209 image records are specifically financial documents. Your contribution is applying essentially this same five-category skeleton (or a refined financial-specific variant of it) to a domain they didn't test, and at model sizes (Phi-3 Mini ~3.8B, Mistral 7B) below the sub-set they call out as size-independent — worth explicitly citing their finding that model size does NOT predict structured-output quality (Phi-4 14B beats GPT-5 on their Value Accuracy metric) as motivation for why testing small models specifically, rather than assuming bigger is safer, is a meaningful thing to do.

### Relevant citations to chase (max 3)
- ExtractBench (Ferguson et al. 2026) — described as the closest prior work; they report frontier models achieving only 4.6% field-level pass rate on PDF-to-JSON extraction, a striking number worth citing directly as evidence extraction is genuinely hard even for large models.
- LLMStructBench (Tenckhoff et al. 2026) — finds prompting strategy matters more than model size; directly useful for your methodology section when you justify testing multiple prompting approaches rather than locking in one.
- STED (Wang et al. 2025) — Semantic Tree Edit Distance, a similarity metric that gives partial credit for semantically-equivalent-but-not-identical values (e.g. "USA" vs "United States") — worth considering as a metric alongside strict exact-match, since financial fields (dates, amounts) may have multiple valid textual representations.
