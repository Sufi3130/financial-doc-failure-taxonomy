# Failure Modes of Small Quantized LLMs in Financial Document Information Extraction: A Local, GDPR-Compliant Pipeline

BSc Thesis, ELTE Faculty of Informatics — Final Examination January 2027
Supervisor: Arafat

## Overview

A fully local, GPU-free pipeline for extracting structured data (JSON) from
financial documents (invoices, bank statements, payslips), using OCR engines
(PaddleOCR, TrOCR) feeding a small quantized LLM (Phi-3 Mini / Mistral 7B GGUF).
Benchmarked against a compact OCR-free vision-language model (PaddleOCR-VL) as
an ablation. The primary contribution is a structured failure taxonomy for
financial document extraction using small local models.

## Repo structure

- `docs/` — proposal, design diagrams, literature notes, milestone tracking
- `src/ocr/` — OCR engine wrappers (PaddleOCR, TrOCR)
- `src/extraction/` — LLM prompt templates, JSON schema, extraction logic
- `src/vlm_ablation/` — OCR-free VLM comparison pipeline
- `evaluation/` — dataset prep, metrics, failure taxonomy annotations
- `app/` — Streamlit demo application
- `tests/` — unit/integration tests

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Results

_Not yet available — to be filled in once evaluation on SROIE/CORD is
complete (target: Milestone 2)._

| Metric | OCR+LLM pipeline | OCR-free VLM ablation |
|---|---|---|
| Field-level accuracy | TBD | TBD |
| Schema compliance rate | TBD | TBD |
| Avg. latency / page (CPU) | TBD | TBD |

## Status

See [docs/milestones.md](docs/milestones.md) and the repo's Issues/Milestones tabs for current progress.
