# Milestone Tracking

Mirrors supervisor deadlines. Each milestone below should have a matching
GitHub Milestone with the same due date, and each bullet a corresponding Issue.

## M1 — August 31, 2026 (Pre-Semester Preparation) ✅
- [x] Finalize thesis topic
- [x] Background research (datasets, algorithms, existing solutions)
- [x] System design diagram
- [x] GitHub repo setup + milestones defined
- [x] Initial prototype / project skeleton

## M2 — September 30, 2026 (Early Autumn Semester) ✅
- [x] Implement core OCR + LLM extraction pipeline (PaddleOCR / Tesseract → Phi-3 Mini Q4)
- [x] Integrate backend components, basic end-to-end functionality (`python -m src.run`, batch mode, run log)
- [x] Complete dataset collection and preprocessing (SROIE, CORD)
- [x] Preliminary tests, document architecture ([architecture.md](architecture.md); baseline results in the README)

## M3 — October 31, 2026 (Mid Autumn Semester)
- [x] Develop and integrate Streamlit UI
- [x] Improve model/system performance (prompt v1/v2 + few-shot 2/0 experiments, issue #24; opt-in grammar-constrained JSON decoding, issue #25 — see [results.md](results.md))
- [x] Solid test coverage (done early in M2: 111 tests with synthetic fixtures, GitHub Actions on Ubuntu + Windows)
- [ ] Draft methodology and results sections
- [ ] Register for Final Examination in Neptun (by Nov 1, 2026)

## M4 — November 15, 2026 (Final Sprint)
- [ ] Finalize implementation, testing, documentation
- [ ] User guide + developer documentation
- [ ] Complete failure taxonomy analysis
- [ ] Submit complete thesis draft for feedback

## Optional (contingent on timeline)
- [ ] OCR-free VLM (PaddleOCR-VL) ablation benchmark
- [ ] Hungarian financial document dataset collection + evaluation
