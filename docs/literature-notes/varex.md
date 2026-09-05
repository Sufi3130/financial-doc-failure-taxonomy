# VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents

**Authors:** Barzelay, Azulai, Shapira, Friedman, Abo Dahood, Lee, Daniels
(IBM Research)
**Venue:** arXiv preprint, 2026
**Link:** https://arxiv.org/pdf/2603.15118

## Summary

VAREX is a benchmark, not an extraction pipeline. Its contribution is a
method for generating documents with guaranteed-correct ground truth
("Reverse Annotation"): start from blank fillable government PDF forms,
fill them with placeholder IDs, have an LLM infer a JSON schema from the
filled form, then swap the placeholders for realistic synthetic values.
Because every value is programmatically written into a known form field,
the ground truth is correct by construction rather than manually labeled.

They evaluate 20 models (800M to frontier scale) on four input
representations of the same documents: plain text, layout-preserving text,
document image, and text+image combined.

## Failure taxonomy (this is the main reason I read it)

- **Schema echo** — the model returns the schema definition itself instead
  of extracted values. Two forms: full schema reproduction (verbatim
  definitions, no values at all — dominant in InternVL3.5 1B), and
  schema-wrapped extraction (correct values extracted but left nested
  inside schema metadata like `"type": "object", "properties": {...}` —
  dominant in Qwen3-VL 2B). Both forms are triggered by the same underlying
  cause: they show that removing `$defs`/`$ref` references from the JSON
  Schema (a common feature of auto-generated schemas, e.g. from Pydantic)
  fixes both failure forms at once, with one model jumping from 27% to 92%
  compliance.
- **Under-extraction** — the model understands the task and produces
  correctly-shaped JSON, but leaves most fields empty. Distinguished from
  schema echo by a specific signature: accuracy on the first few requested
  fields is much higher than on the last ones (2.1x gap at 800M parameters),
  suggesting the model runs out of generation capacity partway through
  rather than failing to understand the schema.
- **Instruction-following threshold, roughly 2–4B parameters** — below this
  size, failures are dominated by the two categories above (format/
  compliance failures, not genuine extraction mistakes). Above it, failures
  shift to being actual extraction errors (OCR mistakes, hallucinated
  values, missed fields).

One notable result: a 2B model fine-tuned specifically for extraction
(NuExtract 2.0) gained about 81 percentage points over its untrained 2B
base and showed zero schema echo, meaning structured-output compliance is a
learnable, addressable skill rather than something that requires more
parameters.

## Relevance to my thesis

Phi-3 Mini sits right around the 2–4B threshold they identify, which means
its failure profile could plausibly look quite different depending on
exactly where it falls relative to that boundary — worth checking
explicitly rather than assuming its failures will look like a "small model"
or a "capable model" a priori.

Their evaluation design — testing the same document across multiple input
representations to isolate whether a failure comes from perception (vision)
or from formatting/reasoning — is directly reusable for my OCR+LLM vs.
OCR-free-VLM comparison. I could similarly test feeding the LLM stage clean
ground-truth text (bypassing OCR) as a control, to separate "OCR was wrong"
from "the LLM reasoned incorrectly over correct text."

## Citations worth following up
- Feng et al. 2025, SO-Bench, and Ferguson et al. 2026, ExtractBench — the
  two benchmarks VAREX positions itself against; both single-modality.
- Geng et al. 2025, JSONSchemaBench — evaluates structured-output compliance
  broadly, useful for separating "produces valid JSON" from "produces
  correct JSON," a distinction I want to preserve in my own taxonomy.
- Livathinos et al. 2025, Docling — open-source layout-aware parsing
  toolkit; its output is similar to VAREX's "layout-preserving text"
  modality and could be a useful OCR post-processing step in my own
  pipeline instead of raw OCR output.
