"""Local, GDPR-compliant UI for the receipt-extraction pipeline.

Two tabs:
  1. Extract & Explain   one receipt (upload or dataset) -> OCR boxes, text,
                         raw output, validated JSON, failure tags, timings, and
                         a per-field comparison when a label is available.
  2. Dataset Evaluation  a built-in subset or your own labelled ZIP -> accuracy
                         with confidence intervals, failure-taxonomy counts, and
                         a per-receipt table you can filter and drill into.

Run from the repo root:   streamlit run app/app.py
Everything runs on this machine: no image or label leaves it (see README
"Run the UI" and .streamlit/config.toml, which bind the server to localhost
and disable Streamlit's telemetry).
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.service as service  # noqa: E402
from src.extraction import FIELDS  # noqa: E402

st.set_page_config(page_title="Receipt extraction — failure taxonomy", page_icon="🧾", layout="wide")

MODEL_LABELS = {
    "phi3": "Phi-3-mini (text, 2-stage pipeline)",
    "qwen2.5-3b": "Qwen2.5-3B (text, 2-stage pipeline)",
    "llama3.2-3b": "Llama-3.2-3B (text, 2-stage pipeline)",
    service.VLM_MODEL: "Qwen2.5-VL-3B (image, OCR-free)",
}


# ------------------------------------------------------------------ cached heavy objects
@st.cache_resource(show_spinner=False)
def get_llm(model_name):
    return service.load_llm(model_name)


@st.cache_resource(show_spinner=False)
def get_engine(name):
    return service.load_engine(name)


@st.cache_resource(show_spinner=False)
def get_vlm(model_name):
    return service.load_vlm(model_name)


@st.cache_resource(show_spinner=False)
def session_tempdir():
    return tempfile.mkdtemp(prefix="receipt_ui_")


# ------------------------------------------------------------------ rendering helpers
def show_outcome(record, row):
    """Validated JSON + failure tags + timings + label comparison for one row."""
    outcome = row.get("outcome") or {}
    if row.get("error"):
        st.error(f"Pipeline error at the {row['error']['stage']} stage: "
                 f"{row['error']['type']} — {row['error']['message']}")

    cols = st.columns(3)
    valid = "✅ strict" if outcome.get("strict_valid") else (
        "🟡 schema-valid (wrapped/repaired)" if outcome.get("schema_valid") else "❌ invalid")
    cols[0].metric("JSON output", valid)
    t = row.get("timings") or {}
    if t.get("ocr_s") is not None:
        cols[1].metric("OCR time", f"{t['ocr_s']:.1f} s")
    cols[2].metric("Total time", f"{t.get('total_s', t.get('llm_s', 0)):.1f} s")

    st.markdown("**Extracted fields (validated JSON)**")
    st.json(outcome.get("lenient") or {})

    failures = outcome.get("failures") or []
    if failures:
        st.markdown("**Format failures:** " + ", ".join(
            f"`{f['type']}`" + (f" ({f['field']})" if f.get("field") else "") for f in failures))

    if service.has_labels(record):
        _show_comparison(record, row)
    else:
        _show_labelfree_checks(record, row)


def _show_comparison(record, row):
    """Per-field correct/wrong and which stage failed (needs a label)."""
    result = service.score(record, row)
    instances = {i["field"]: i for i in service.tags(record, row)}
    rows = []
    for field in FIELDS:
        entry = result["fields"].get(field)
        if not entry:
            continue
        inst = instances.get(field)
        rows.append({
            "field": field,
            "predicted": entry["pred"],
            "label": entry["gold"],
            "correct": "✅" if entry["normalized"] else "❌",
            "in OCR text": "yes" if entry["in_ocr"] else "no",
            "failure category": "" if entry["normalized"] else (inst["primary"] if inst else ""),
            "stage": "" if entry["normalized"] else (inst["stage"] if inst else ""),
        })
    st.markdown("**Comparison with the label**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("`in OCR text = no` ⇒ the OCR stage lost the value (OCR miss); "
               "`yes` with a wrong value ⇒ the model mis-extracted it (LLM miss).")


def _show_labelfree_checks(record, row):
    """Checks that need no label: schema validity + a hallucination flag."""
    outcome = row.get("outcome") or {}
    pred = outcome.get("lenient") or {}
    ocr_text = (row.get("ocr") or {}).get("text", "")
    end_to_end = (row.get("ocr") or {}).get("end_to_end")
    st.markdown("**Automatic checks (no label provided)**")
    if end_to_end:
        st.caption("OCR-free VLM: there is no OCR text, so the hallucination check below is not available.")
        return
    rows = []
    for field, value in pred.items():
        if not value:
            continue
        from evaluation.evaluate import squash
        present = squash(value) in squash(ocr_text)
        rows.append({"field": field, "predicted": value,
                     "in OCR text": "yes" if present else "⚠️ no (possible hallucination)"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("A value not found in the OCR text was likely invented by the model. "
               "Enter the correct values above to get the full per-field verdict.")


def show_ocr(record, row):
    """Image with OCR boxes + reading-order text + raw model output."""
    left, right = st.columns([3, 2])
    image_path = REPO_ROOT / record["image"] if not Path(record["image"]).is_absolute() else Path(record["image"])
    ocr = row.get("ocr") or {}
    with left:
        if ocr.get("lines"):
            st.image(service.draw_boxes(image_path, ocr["lines"]),
                     caption="OCR pieces (green = high confidence, amber = medium, red = low)",
                     use_container_width=True)
        else:
            st.image(str(image_path), caption="receipt image", use_container_width=True)
    with right:
        if ocr.get("text"):
            with st.expander("Reading-order OCR text", expanded=True):
                st.text(ocr["text"])
        raw = (row.get("llm") or {}).get("raw_output")
        if raw is not None:
            with st.expander("Raw model output", expanded=not row.get("outcome", {}).get("strict_valid")):
                st.code(raw, language="json")


# ------------------------------------------------------------------ Tab 1: Extract & Explain
def tab_extract():
    st.subheader("Extract & Explain")
    st.caption("Run one receipt through the pipeline and see every stage: OCR, the model's "
               "raw output, the validated JSON, the failure tags, and — with a label — which stage failed.")

    source = st.radio("Receipt source", ["Upload an image", "Pick from a dataset"], horizontal=True)
    record, image_path = None, None

    if source == "Upload an image":
        up = st.file_uploader("Receipt image (PNG or JPG)", type=["png", "jpg", "jpeg"])
        locale = st.selectbox("Currency / locale", list(service.LOCALES),
                              format_func=lambda k: service.LOCALES[k])
        if up is not None:
            tmp = Path(session_tempdir()) / up.name
            tmp.write_bytes(up.getbuffer())
            image_path = tmp
            with st.expander("Optional: enter the correct values to get the full failure taxonomy"):
                gold = {f: st.text_input(f, key=f"gold_{f}") for f in FIELDS}
            record = service.make_record(image_path, locale=locale, gold=gold)
    else:
        manifests = service.builtin_manifests()
        if not manifests:
            st.info("No dataset subsets found. Run the dataset prepare scripts first (see README).")
            return
        name = st.selectbox("Dataset", list(manifests))
        recs = service.load_manifest(manifests[name])
        rid = st.selectbox("Receipt", [r["id"] for r in recs])
        record = next(r for r in recs if r["id"] == rid)
        image_path = REPO_ROOT / record["image"]

    engine_choice = _engine_model_controls()

    if record and st.button("Run extraction", type="primary"):
        row = _run_single(record, engine_choice)
        if row:
            st.session_state["last_row"] = (record, row)

    if "last_row" in st.session_state:
        record, row = st.session_state["last_row"]
        st.divider()
        show_ocr(record, row)
        st.divider()
        show_outcome(record, row)


def _engine_model_controls():
    """Shared OCR-engine + model (or VLM) picker. Returns a dict describing the choice."""
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Mode", ["2-stage pipeline (OCR → LLM)", "OCR-free VLM (image → JSON)"])
    if mode.startswith("OCR-free"):
        return {"vlm": True, "model": service.VLM_MODEL}
    engine = c2.selectbox("OCR engine", service.OCR_ENGINES)
    model = c3.selectbox("LLM", service.TEXT_MODELS, format_func=lambda m: MODEL_LABELS.get(m, m))
    shots = c3.radio("Few-shot", [2, 0], format_func=lambda k: f"{k}-shot", horizontal=True)
    return {"vlm": False, "engine": engine, "model": model, "shots": shots}


def _run_single(record, choice):
    try:
        if choice["vlm"]:
            with st.spinner("Loading the vision-language model and reading the image (slow on CPU)…"):
                vlm = get_vlm(choice["model"])
                return service.extract_vlm(record, vlm)
        with st.spinner(f"Running {choice['engine']} + {choice['model']} (first load is slow)…"):
            engine = get_engine(choice["engine"])
            llm = get_llm(choice["model"])
            ids, examples = service.fewshot(engine, shots=choice["shots"])
            return service.extract(record, engine, llm, examples, ids)
    except Exception as exc:  # keep the UI alive; surface the cause
        st.error(f"Could not run extraction: {type(exc).__name__}: {exc}")
        return None


# ------------------------------------------------------------------ Tab 2: Dataset Evaluation
def tab_evaluate():
    st.subheader("Dataset Evaluation")
    st.caption("Run the pipeline over a whole dataset and see accuracy with confidence intervals and "
               "the failure-taxonomy counts. Use a built-in subset or upload your own labelled receipts.")

    source = st.radio("Dataset", ["Built-in subset", "Upload your own (ZIP)"], horizontal=True)
    records = None

    if source == "Built-in subset":
        manifests = service.builtin_manifests()
        if not manifests:
            st.info("No dataset subsets found. Run the dataset prepare scripts first (see README).")
            return
        name = st.selectbox("Subset", list(manifests))
        records = service.load_manifest(manifests[name])
        st.caption(f"{len(records)} receipts.")
    else:
        st.markdown("Upload a **ZIP** containing your images and a `labels.csv` with columns "
                    "`id, image, company, date, address, total` (any field may be blank; only "
                    "filled fields are scored).")
        locale = st.selectbox("Currency / locale", list(service.LOCALES),
                              format_func=lambda k: service.LOCALES[k], key="eval_locale")
        up = st.file_uploader("Dataset ZIP", type=["zip"])
        if up is not None:
            dest = Path(session_tempdir()) / "user_dataset"
            records, messages = service.ingest_zip(up.getbuffer().tobytes(), dest, locale=locale)
            for m in messages:
                st.warning(m)
            if records:
                st.success(f"Loaded {len(records)} receipts.")

    c1, c2 = st.columns(2)
    engine = c1.selectbox("OCR engine", service.OCR_ENGINES, key="eval_engine")
    model = c2.selectbox("LLM", service.TEXT_MODELS, format_func=lambda m: MODEL_LABELS.get(m, m), key="eval_model")

    if records and st.button("Run evaluation", type="primary"):
        n = len(records)
        est_min = n * (1.7 if engine == "paddleocr" else 0.9)
        st.info(f"Running {n} receipts through {engine} + {model}. "
                f"Uncached images take ~1–4 min each on CPU (rough estimate: {est_min:.0f} min).")
        _run_batch(records, engine, model)

    if "eval_result" in st.session_state:
        _show_eval_result(*st.session_state["eval_result"])


def _run_batch(records, engine, model):
    progress = st.progress(0.0, text="starting…")
    done = {"n": 0}
    total = len(records)

    def log(msg):
        if str(msg).startswith("["):            # per-receipt line from src.run
            done["n"] += 1
            progress.progress(min(done["n"] / total, 1.0), text=f"{done['n']}/{total}  {msg}")

    try:
        with st.spinner("Loading the model…"):
            llm = get_llm(model)
        run_dir = service.new_run_dir()
        run_jsonl, manifest_path = service.run_batch(records, engine, llm, run_dir, log=log)
        with st.spinner("Scoring and tagging…"):
            result = service.evaluate_run(run_dir, manifest_path)
        st.session_state["eval_result"] = (records, *result)
        progress.progress(1.0, text="done")
    except Exception as exc:
        st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")


def _show_eval_result(records, _records2, per_receipt, summaries, instances, counts):
    for engine, s in summaries.items():
        st.markdown(f"### Results — `{engine}`")
        c = st.columns(4)
        allf = s.get("all_fields_normalized_acc")
        ci = s.get("all_fields_normalized_acc_ci95")
        c[0].metric("All fields correct", f"{allf}%" if allf is not None else "—",
                    help=f"95% CI {ci}" if ci else None)
        c[1].metric("Receipts", s["receipts"])
        c[2].metric("Errors", s["errors"])
        sc = s["schema"]
        c[3].metric("Schema-valid JSON", f"{sc['schema_valid_rate']}%")

        field_rows = []
        for f, x in s["fields"].items():
            field_rows.append({"field": f, "normalised acc %": x["normalized_acc"],
                               "95% CI": f"{x['normalized_acc_ci95']}" if x["normalized_acc_ci95"] else "",
                               "strict %": x["strict_acc"], "in OCR %": x["in_ocr"],
                               "OCR miss": x["ocr_miss"], "LLM miss": x["llm_miss_given_in_ocr"]})
        st.dataframe(pd.DataFrame(field_rows), hide_index=True, use_container_width=True)

    st.markdown("### Failure taxonomy")
    prim = counts.get("primary", {})
    if prim:
        tax = pd.DataFrame(prim).fillna(0).astype(int)
        st.dataframe(tax, use_container_width=True)
    else:
        st.caption("No failures tagged (every scored field was correct).")

    _show_receipt_explorer(per_receipt, instances)


def _show_receipt_explorer(per_receipt, instances):
    st.markdown("### Per-receipt explorer")
    inst_by_receipt = {}
    for i in instances:
        inst_by_receipt.setdefault((i["engine"], i["receipt_id"]), []).append(i)

    rows = []
    for engine, results in per_receipt.items():
        for r in results:
            fields = r["fields"]
            wrong = [f for f, x in fields.items() if not x["normalized"]]
            cats = sorted({i["primary"] for i in inst_by_receipt.get((engine, r["receipt_id"]), [])})
            rows.append({"engine": engine, "receipt": r["receipt_id"],
                         "all correct": "✅" if fields and not wrong else "❌",
                         "wrong fields": ", ".join(wrong), "failure categories": ", ".join(cats),
                         "error": r["error"] or ""})
    df = pd.DataFrame(rows)
    all_cats = sorted({c for r in rows for c in (r["failure categories"].split(", ") if r["failure categories"] else [])})
    chosen = st.multiselect("Filter by failure category", all_cats)
    if chosen:
        df = df[df["failure categories"].apply(lambda s: any(c in s for c in chosen))]
    st.dataframe(df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ main
def main():
    st.title("🧾 Receipt extraction — local failure-mode explorer")
    st.caption("Small quantized LLMs for financial-document extraction, run entirely on this machine. "
               "BSc thesis, ELTE.")
    tab1, tab2 = st.tabs(["Extract & Explain", "Dataset Evaluation"])
    with tab1:
        tab_extract()
    with tab2:
        tab_evaluate()


main()
