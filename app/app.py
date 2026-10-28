"""Local, GDPR-compliant UI for the receipt-extraction pipeline.

Tabs:
  1. Extract & Explain   one receipt -> OCR boxes, text, raw output, validated
                         JSON, failure tags, timings, per-field label comparison.
  2. Compare models      one receipt -> Phi-3 / Qwen2.5-3B / Llama-3.2-3B / the
                         OCR-free VLM side by side (output, correctness, timing).
  3. Dataset Evaluation  one image, a built-in subset, or your own labelled ZIP
                         -> accuracy with CIs and the failure-taxonomy counts;
                         the OCR-free VLM is available as an engine too.

Run from the repo root:   streamlit run app/app.py
Everything runs on this machine: no image or label leaves it (see README
"Run the UI" and .streamlit/config.toml — localhost only, telemetry off).
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Streamlit runs this file as a script with app/ on sys.path, so the name `app`
# resolves to this module, not the package: import the sibling directly. (Under
# pytest the package import `app.service` is used instead; both reach the same file.)
try:
    import service  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - imported as a package (tests)
    from app import service  # noqa: E402
from src.extraction import FIELDS  # noqa: E402

st.set_page_config(page_title="Receipt extraction — failure taxonomy", page_icon="🧾", layout="wide")

MODEL_LABELS = {
    "phi3": "Phi-3-mini",
    "qwen2.5-3b": "Qwen2.5-3B",
    "llama3.2-3b": "Llama-3.2-3B",
    service.VLM_MODEL: "Qwen2.5-VL-3B (image)",
}
ALL_MODELS = service.TEXT_MODELS + [service.VLM_MODEL]
_PROMPT_DESC = {"v1": "baseline", "v2": "disambiguation rules"}

_CSS = """
<style>
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none !important;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}

/* sidebar */
[data-testid="stSidebar"] {background: #080a0f; border-right: 1px solid rgba(255,255,255,0.06);}
[data-testid="stSidebar"] .block-container {padding-top: 1.1rem;}
.logo {display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.2rem; margin-bottom:10px;}
.logo .mark {background:linear-gradient(135deg,#6366f1,#22d3ee); width:34px; height:34px;
  border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.05rem;}
.badge {display:inline-flex; align-items:center; gap:6px; background:rgba(34,197,94,0.12);
  color:#4ade80; border:1px solid rgba(34,197,94,0.3); padding:3px 10px; border-radius:999px; font-size:0.78rem;}
[data-testid="stSidebar"] .stButton button {text-align:left; justify-content:flex-start; border:none;
  background:transparent; font-weight:600; border-radius:10px; padding:8px 12px;}
[data-testid="stSidebar"] .stButton button:hover {background:rgba(255,255,255,0.05);}
[data-testid="stSidebar"] .stButton button[kind="primary"] {background:rgba(99,102,241,0.18); color:#c7d2fe;}

/* metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(99,102,241,0.08), rgba(99,102,241,0.02));
  border: 1px solid rgba(99,102,241,0.18); border-radius: 14px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {opacity:0.7; font-size:0.74rem; text-transform:uppercase; letter-spacing:0.05em;}

/* bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background:#121722; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:8px 6px;
}

/* buttons, inputs */
.stButton button {border-radius:10px; font-weight:600;}
.stButton button[kind="primary"] {background:#6366f1; border:none;}
[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}

h1 {font-weight:800; letter-spacing:-0.02em;}
h2,h3 {letter-spacing:-0.01em;}
.hero-sub {opacity:0.68; font-size:1.0rem; margin-top:-8px;}

/* pipeline diagram */
.pipeline {display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; margin:10px 0 4px;}
.pipeline .stage {background:#121722; border:1px solid rgba(255,255,255,0.08); border-radius:14px;
  padding:14px 16px; min-width:118px; flex:1;}
.pipeline .stage .ic {font-size:1.45rem;}
.pipeline .stage .nm {font-weight:700; margin-top:6px;}
.pipeline .stage small {opacity:0.6;}
.pipeline .arrow {display:flex; align-items:center; color:#6366f1; font-size:1.3rem; padding:0 2px;}
.feature h4 {margin:2px 0 4px;}
</style>
"""


# ------------------------------------------------------------------ cached heavy objects
@st.cache_resource(show_spinner=False)
def get_llm(model_name, grammar=False):
    return service.load_llm(model_name, grammar=grammar)


@st.cache_resource(show_spinner=False)
def get_engine(name):
    return service.load_engine(name)


@st.cache_resource(show_spinner=False)
def get_vlm(model_name):
    return service.load_vlm(model_name)


@st.cache_resource(show_spinner=False)
def session_tempdir():
    return tempfile.mkdtemp(prefix="receipt_ui_")


# ------------------------------------------------------------------ small helpers
def image_path_of(record):
    p = Path(record["image"])
    return p if p.is_absolute() else REPO_ROOT / p


def run_model(record, model_name, engine_name="paddleocr", shots=2, prompt_version=service.DEFAULT_PROMPT,
              grammar=False):
    """Run one receipt with one model (text pipeline or VLM). Returns a run row."""
    if service.is_vlm_engine(model_name):
        vlm = get_vlm(model_name)
        return service.extract_vlm(record, vlm)
    engine = get_engine(engine_name)
    llm = get_llm(model_name, grammar=grammar)
    ids, examples = service.fewshot(engine, shots=shots)
    return service.extract(record, engine, llm, examples, ids, prompt_version=prompt_version)


_STAGE_PCT = {"ocr": 0.55, "llm": 0.72, "parse": 0.95}
_STAGE_TXT = {"ocr": "Reading the receipt (OCR)…", "llm": "Running the model…", "parse": "Parsing the output…"}


def _run_with_progress(record, choice, bar):
    """Run one receipt, advancing the progress bar through each phase."""
    if choice["vlm"]:
        bar.progress(0.2, text="Loading the vision-language model (slow on CPU)…")
        vlm = get_vlm(service.VLM_MODEL)
        bar.progress(0.5, text="Reading the image end-to-end (this takes a few minutes)…")
        row = service.extract_vlm(record, vlm)
        bar.progress(1.0, text="Done")
        return row
    bar.progress(0.1, text="Loading the OCR engine…")
    engine = get_engine(choice["engine"])
    bar.progress(0.3, text="Loading the model (first time is slow)…")
    llm = get_llm(choice["model"], grammar=choice.get("grammar", False))
    bar.progress(0.42, text="Preparing few-shot examples…")
    ids, examples = service.fewshot(engine, shots=choice["shots"])
    row = service.extract(record, engine, llm, examples, ids,
                          on_stage=lambda s: bar.progress(_STAGE_PCT.get(s, 0.6),
                                                          text=_STAGE_TXT.get(s, "Working…")),
                          prompt_version=choice.get("prompt", service.DEFAULT_PROMPT))
    bar.progress(1.0, text="Done")
    return row


def valid_badge(outcome):
    if outcome.get("strict_valid"):
        return "✅ strict JSON"
    if outcome.get("schema_valid"):
        return "🟡 schema-valid"
    return "❌ invalid"


def pick_receipt(key, allow_upload=True, label_inputs=False):
    """Shared receipt chooser. Returns a manifest record (with image) or None."""
    options = (["Upload an image"] if allow_upload else []) + ["Pick from a dataset"]
    source = st.radio("Receipt source", options, horizontal=True, key=f"{key}_src")
    if source == "Upload an image":
        up = st.file_uploader("Receipt image (PNG or JPG)", type=["png", "jpg", "jpeg"], key=f"{key}_up")
        locale = st.selectbox("Currency / locale", list(service.LOCALES),
                              format_func=lambda k: service.LOCALES[k], key=f"{key}_loc")
        if up is None:
            return None
        tmp = Path(session_tempdir()) / up.name
        tmp.write_bytes(up.getbuffer())
        gold = None
        if label_inputs:
            with st.expander("Optional: enter the correct values for a full comparison"):
                cols = st.columns(2)
                gold = {f: cols[i % 2].text_input(f, key=f"{key}_gold_{f}") for i, f in enumerate(FIELDS)}
        return service.make_record(tmp, locale=locale, gold=gold)
    scope = st.radio("Dataset scope", ["50-receipt subset", "Full dataset"],
                     horizontal=True, key=f"{key}_scope")
    manifests = service.builtin_manifests() if scope == "50-receipt subset" else service.full_manifests()
    if not manifests:
        st.info("No dataset manifests found. Run the dataset prepare scripts first (see README).")
        return None
    name = st.selectbox("Dataset", list(manifests), key=f"{key}_ds")
    recs = service.load_manifest(manifests[name])
    if scope == "Full dataset":
        recs = service.evaluatable_records(recs)   # drop the train / few-shot split
        st.caption(f"{len(recs)} receipts in the full set. Anything outside the 50-receipt subset "
                   "has not been OCR'd yet, so the first run on it is slower.")
    rid = st.selectbox("Receipt", [r["id"] for r in recs], key=f"{key}_rid")
    return next(r for r in recs if r["id"] == rid)


# ------------------------------------------------------------------ outcome rendering
def show_ocr(record, row):
    left, right = st.columns([3, 2])
    ocr = row.get("ocr") or {}
    with left:
        if ocr.get("lines"):
            st.image(service.draw_boxes(image_path_of(record), ocr["lines"]),
                     caption="OCR pieces — green = high confidence, amber = medium, red = low",
                     use_container_width=True)
        else:
            st.image(str(image_path_of(record)), caption="receipt image", use_container_width=True)
    with right:
        if ocr.get("text"):
            with st.expander("Reading-order OCR text", expanded=True):
                st.text(ocr["text"])
        raw = (row.get("llm") or {}).get("raw_output")
        if raw is not None:
            with st.expander("Raw model output", expanded=not (row.get("outcome") or {}).get("strict_valid")):
                st.code(raw, language="json")


def show_outcome(record, row):
    outcome = row.get("outcome") or {}
    if row.get("error"):
        st.error(f"Pipeline error at the {row['error']['stage']} stage: "
                 f"{row['error']['type']} — {row['error']['message']}")
    c = st.columns(3)
    c[0].metric("JSON output", valid_badge(outcome))
    t = row.get("timings") or {}
    if t.get("ocr_s") is not None:
        c[1].metric("OCR time", f"{t['ocr_s']:.1f} s")
    c[2].metric("Total time", f"{t.get('total_s', t.get('llm_s', 0)):.1f} s")

    st.markdown("**Extracted fields (validated JSON)**")
    st.json(outcome.get("lenient") or {})

    failures = outcome.get("failures") or []
    real = [f for f in failures if f["type"] != "null_field"]
    nulls = [f["field"] for f in failures if f["type"] == "null_field" and f.get("field")]
    if real:
        st.markdown("**JSON format problems:** " + ", ".join(
            f"`{f['type']}`" + (f" ({f['field']})" if f.get("field") else "") for f in real))
    if nulls:
        st.caption("Returned **null** (no value) for: " + ", ".join(nulls) +
                   " — `null` means *not on the receipt*. For CORD receipts that is the expected answer, "
                   "since CORD only labels the total.")

    if service.has_labels(record):
        _show_comparison(record, row)
    else:
        _show_labelfree_checks(record, row)


def _show_comparison(record, row):
    result = service.score(record, row)
    instances = {i["field"]: i for i in service.tags(record, row)}
    rows = []
    for field in FIELDS:
        entry = result["fields"].get(field)
        if not entry:
            continue
        inst = instances.get(field)
        rows.append({
            "field": field, "predicted": entry["pred"], "label": entry["gold"],
            "correct": "✅" if entry["normalized"] else "❌",
            "in OCR text": "yes" if entry["in_ocr"] else "no",
            "failure category": "" if entry["normalized"] else (inst["primary"] if inst else ""),
            "stage": "" if entry["normalized"] else (inst["stage"] if inst else ""),
        })
    st.markdown("**Comparison with the label**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    end_to_end = bool((row.get("ocr") or {}).get("end_to_end"))
    wrong = [(f, instances.get(f)) for f in FIELDS
             if f in result["fields"] and not result["fields"][f]["normalized"]]
    if wrong:
        st.markdown("**Why these fields are wrong**")
        for field, inst in wrong:
            if end_to_end:
                st.markdown(f"- **{field}** — the VLM read the image directly; with no OCR text this can't be "
                            f"split into a reading vs an extraction error.")
            elif inst is not None:
                st.markdown(f"- **{field}** — {service.explain(inst)}")
            else:
                st.markdown(f"- **{field}** — wrong value.")
    if not end_to_end:
        st.caption("Reading the table: `in OCR text = no` ⇒ the OCR stage lost the value (OCR miss); "
                   "`yes` with a wrong value ⇒ the model mis-extracted it (LLM miss).")
    else:
        st.caption("OCR-free VLM: there is no OCR stage, so a wrong field cannot be split into "
                   "reading vs extraction.")


def _show_labelfree_checks(record, row):
    outcome = row.get("outcome") or {}
    pred = outcome.get("lenient") or {}
    ocr_text = (row.get("ocr") or {}).get("text", "")
    st.markdown("**Automatic checks (no label provided)**")
    if (row.get("ocr") or {}).get("end_to_end"):
        st.caption("OCR-free VLM: there is no OCR text, so the hallucination check is not available. "
                   "Enter the correct values to get a full per-field verdict.")
        return
    from evaluation.evaluate import squash
    rows = [{"field": f, "predicted": v,
             "in OCR text": "yes" if squash(v) in squash(ocr_text) else "⚠️ no (possible hallucination)"}
            for f, v in pred.items() if v]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("A value not found in the OCR text was likely invented by the model. "
               "Enter the correct values above to get the full per-field verdict.")


# ------------------------------------------------------------------ Tab 1: Extract & Explain
def tab_extract():
    hero("Extract &amp; Explain",
         "Run one receipt through the pipeline and see every stage: OCR, the model's raw output, the "
         "validated JSON, the failure tags, and — with a label — which stage failed.")

    record = pick_receipt("ex", allow_upload=True, label_inputs=True)
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Mode", ["2-stage pipeline (OCR → LLM)", "OCR-free VLM (image → JSON)"])
    if mode.startswith("OCR-free"):
        choice = {"vlm": True, "model": service.VLM_MODEL}
    else:
        engine = c2.selectbox("OCR engine", service.OCR_ENGINES)
        prompt_v = c2.selectbox("Prompt", service.PROMPTS, key="ex_prompt",
                                format_func=lambda v: f"{v} — {_PROMPT_DESC[v]}")
        model = c3.selectbox("LLM", service.TEXT_MODELS, format_func=lambda m: MODEL_LABELS.get(m, m))
        shots = c3.radio("Few-shot", [2, 0], format_func=lambda k: f"{k}-shot", horizontal=True)
        grammar_on = st.checkbox(
            "Constrain output to the JSON schema (grammar)", key="ex_grammar",
            help="Issue #25: force valid Receipt JSON via a GBNF grammar (eliminates malformed_json; "
                 "empty pages answer all-null). Off by default so the baseline still shows format failures.")
        choice = {"vlm": False, "engine": engine, "model": model, "shots": shots,
                  "prompt": prompt_v, "grammar": grammar_on}

    if record and not choice["vlm"] and not service.ocr_cached(choice["engine"], record):
        st.info("🆕 This receipt has not been read before, so the OCR step will run fresh — "
                "the first run on a new receipt is slower.")

    if record and st.button("Run extraction", type="primary", key="ex_run"):
        bar = st.progress(0.0, text="starting…")
        try:
            row = _run_with_progress(record, choice, bar)
            st.session_state["ex_row"] = (record, row)
        except Exception as exc:
            st.error(f"Could not run extraction: {type(exc).__name__}: {exc}")
        finally:
            bar.empty()

    if "ex_row" in st.session_state:
        record, row = st.session_state["ex_row"]
        st.divider()
        show_ocr(record, row)
        st.divider()
        show_outcome(record, row)


# ------------------------------------------------------------------ Tab 2: Compare models
def tab_compare():
    hero("Compare models",
         "Run one receipt through several models at once and compare the output, correctness, failure "
         "category and speed. The VLM reads the image directly (no OCR) and is slow on CPU.")

    record = pick_receipt("cmp", allow_upload=True, label_inputs=True)
    chosen = st.multiselect("Models", ALL_MODELS, default=service.TEXT_MODELS,
                            format_func=lambda m: MODEL_LABELS.get(m, m))
    c1, c2 = st.columns(2)
    engine = c1.selectbox("OCR engine (for the text models)", service.OCR_ENGINES, key="cmp_engine")
    prompt_v = c2.selectbox("Prompt (for the text models)", service.PROMPTS, key="cmp_prompt",
                            format_func=lambda v: f"{v} — {_PROMPT_DESC[v]}")
    grammar_on = st.checkbox(
        "Constrain output to the JSON schema (grammar)", key="cmp_grammar",
        help="Issue #25: force valid Receipt JSON via a GBNF grammar. Off by default.")

    if record and not service.ocr_cached(engine, record):
        st.info("🆕 This receipt has not been read before, so the OCR step will run fresh — "
                "slower the first time.")

    if record and chosen and st.button("Run comparison", type="primary", key="cmp_run"):
        rows = {}
        bar = st.progress(0.0, text="starting…")
        for i, model in enumerate(chosen, 1):
            slow = " (VLM — slow)" if service.is_vlm_engine(model) else ""
            bar.progress((i - 1) / len(chosen), text=f"{MODEL_LABELS.get(model, model)}{slow}…")
            try:
                rows[model] = run_model(record, model, engine, prompt_version=prompt_v, grammar=grammar_on)
            except Exception as exc:
                st.error(f"{MODEL_LABELS.get(model, model)} failed: {type(exc).__name__}: {exc}")
        bar.progress(1.0, text="done")
        st.session_state["cmp_rows"] = (record, rows)

    if "cmp_rows" in st.session_state:
        record, rows = st.session_state["cmp_rows"]
        if rows:
            st.divider()
            _show_compare(record, rows)


def _show_compare(record, rows):
    labelled = service.has_labels(record)
    scores = {m: service.score(record, row) for m, row in rows.items()} if labelled else {}
    tags = {m: {i["field"]: i for i in service.tags(record, row)} for m, row in rows.items()} if labelled else {}

    table = []
    for model, row in rows.items():
        outcome = row.get("outcome") or {}
        pred = outcome.get("lenient") or {}
        rec = {"model": MODEL_LABELS.get(model, model)}
        all_ok = labelled
        for f in FIELDS:
            value = pred.get(f)
            if labelled and f in scores[model]["fields"]:
                ok = scores[model]["fields"][f]["normalized"]
                all_ok = all_ok and ok
                rec[f] = f"{'✅' if ok else '❌'} {value if value is not None else '—'}"
            else:
                rec[f] = value if value is not None else "—"
        if labelled:
            rec["all correct"] = "✅" if all_ok else "❌"
        rec["JSON"] = valid_badge(outcome)
        t = row.get("timings") or {}
        rec["time (s)"] = round(t.get("total_s", t.get("llm_s", 0)), 1)
        table.append(rec)

    st.markdown("**Side-by-side**")
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    if labelled:
        st.caption("✅/❌ is the normalised match against the label. "
                   "The VLM has no OCR stage, so its failures can't be split into OCR vs LLM.")

    with st.expander("Per-model detail (raw output + failure categories)"):
        for model, row in rows.items():
            st.markdown(f"**{MODEL_LABELS.get(model, model)}**")
            cats = [f"`{i['field']}: {i['primary']}`" for i in (tags.get(model, {}).values())
                    if not (scores.get(model, {}).get("fields", {}).get(i["field"], {}) or {}).get("normalized")]
            if labelled and cats:
                st.markdown("failures: " + ", ".join(cats))
            raw = (row.get("llm") or {}).get("raw_output")
            if raw is not None:
                st.code(raw, language="json")


# ------------------------------------------------------------------ Tab 3: Dataset Evaluation
def tab_evaluate():
    hero("Dataset Evaluation",
         "Score the pipeline over one image, a built-in subset, or your own labelled dataset: accuracy "
         "with 95% confidence intervals and the failure-taxonomy counts.")

    source = st.radio("What to evaluate",
                      ["Built-in subset", "Single receipt", "Upload (image or ZIP)"], horizontal=True)
    records = None

    if source == "Built-in subset":
        manifests = service.builtin_manifests()
        if not manifests:
            st.info("No dataset subsets found. Run the dataset prepare scripts first (see README).")
            return
        name = st.selectbox("Subset", list(manifests), key="eval_subset")
        records = service.load_manifest(manifests[name])
        st.caption(f"{len(records)} receipts.")
    elif source == "Single receipt":
        rec = pick_receipt("eval1", allow_upload=False)
        if rec is not None:
            records = [rec]
            st.caption("One receipt — accuracy is a single pass, so confidence intervals are not meaningful.")
    else:
        records = _upload_for_eval()

    c1, c2 = st.columns(2)
    engine = c1.selectbox("Engine", service.OCR_ENGINES + [service.VLM_MODEL],
                          format_func=lambda e: MODEL_LABELS.get(e, e) if service.is_vlm_engine(e) else e,
                          key="eval_engine")
    vlm = service.is_vlm_engine(engine)
    model = None if vlm else c2.selectbox("LLM", service.TEXT_MODELS,
                                          format_func=lambda m: MODEL_LABELS.get(m, m), key="eval_model")

    if records and st.button("Run evaluation", type="primary", key="eval_run"):
        _run_batch(records, engine, model)

    if "eval_result" in st.session_state:
        _show_eval_result(*st.session_state["eval_result"])


def _upload_for_eval():
    locale = st.selectbox("Currency / locale", list(service.LOCALES),
                          format_func=lambda k: service.LOCALES[k], key="eval_locale")
    up = st.file_uploader("A single image, or a ZIP of images + labels.csv", type=["zip", "png", "jpg", "jpeg"],
                          key="eval_up")
    if up is None:
        st.markdown("ZIP layout: `images/` + `labels.csv` with columns "
                    "`id, image, company, date, address, total` (any field may be blank).")
        return None
    if up.name.lower().endswith(".zip"):
        dest = Path(session_tempdir()) / "user_dataset"
        records, messages = service.ingest_zip(up.getbuffer().tobytes(), dest, locale=locale)
        for m in messages:
            st.warning(m)
        if records:
            st.success(f"Loaded {len(records)} receipts.")
        return records
    tmp = Path(session_tempdir()) / up.name
    tmp.write_bytes(up.getbuffer())
    with st.expander("Enter the correct values (so the single image can be scored)", expanded=True):
        cols = st.columns(2)
        gold = {f: cols[i % 2].text_input(f, key=f"eval_gold_{f}") for i, f in enumerate(FIELDS)}
    return [service.make_record(tmp, locale=locale, gold=gold)]


def _run_batch(records, engine, model):
    n = len(records)
    if not service.is_vlm_engine(engine):
        est = n * (1.7 if engine == "paddleocr" else 0.9)
        st.info(f"Running {n} receipt(s) through {engine} + {model}. "
                f"Uncached images take ~1–2 min each on CPU (~{est:.0f} min total).")
    else:
        st.info(f"Running {n} receipt(s) through the OCR-free VLM. ~3–4 min each on CPU "
                f"(~{n * 3.5:.0f} min total).")
    progress = st.progress(0.0, text="starting…")
    done, total = {"n": 0}, n

    def log(msg):
        if str(msg).startswith("["):
            done["n"] += 1
            progress.progress(min(done["n"] / total, 1.0), text=f"{done['n']}/{total}  {msg}")

    try:
        with st.spinner("Loading the model…"):
            model_obj = get_vlm(service.VLM_MODEL) if service.is_vlm_engine(engine) else get_llm(model)
        run_dir = service.new_run_dir()
        run_jsonl, manifest_path = service.run_batch(records, engine, model_obj, run_dir, log=log)
        with st.spinner("Scoring and tagging…"):
            result = service.evaluate_run(run_dir, manifest_path)
        st.session_state["eval_result"] = result
        progress.progress(1.0, text="done")
    except Exception as exc:
        st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")


def _show_eval_result(records, per_receipt, summaries, instances, counts):
    for engine, s in summaries.items():
        is_vlm = engine == "vlm"
        st.markdown(f"### Results — {'Qwen2.5-VL-3B (OCR-free)' if is_vlm else f'`{engine}`'}")
        c = st.columns(4)
        allf, ci = s.get("all_fields_normalized_acc"), s.get("all_fields_normalized_acc_ci95")
        c[0].metric("All fields correct", f"{allf}%" if allf is not None else "—",
                    help=f"95% CI {ci}" if ci else None)
        c[1].metric("Receipts", s["receipts"])
        c[2].metric("Errors", s["errors"])
        c[3].metric("Schema-valid JSON", f"{s['schema']['schema_valid_rate']}%")

        field_rows = []
        for f, x in s["fields"].items():
            row = {"field": f, "normalised acc %": x["normalized_acc"],
                   "95% CI": f"{x['normalized_acc_ci95']}" if x["normalized_acc_ci95"] else "",
                   "strict %": x["strict_acc"]}
            if not is_vlm:
                row.update({"in OCR %": x["in_ocr"], "OCR miss": x["ocr_miss"],
                            "LLM miss": x["llm_miss_given_in_ocr"]})
            field_rows.append(row)
        st.dataframe(pd.DataFrame(field_rows), hide_index=True, use_container_width=True)

    if any(e == "vlm" for e in summaries):
        st.caption("The OCR-free VLM has no OCR stage, so OCR-vs-LLM attribution and the "
                   "stage-based taxonomy below do not apply to it.")
    else:
        st.markdown("### Failure taxonomy")
        prim = counts.get("primary", {})
        if prim:
            st.dataframe(pd.DataFrame(prim).fillna(0).astype(int), use_container_width=True)
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
            wrong = [f for f, x in r["fields"].items() if not x["normalized"]]
            cats = sorted({i["primary"] for i in inst_by_receipt.get((engine, r["receipt_id"]), [])})
            rows.append({"engine": engine, "receipt": r["receipt_id"],
                         "all correct": "✅" if r["fields"] and not wrong else "❌",
                         "wrong fields": ", ".join(wrong), "failure categories": ", ".join(cats),
                         "error": r["error"] or ""})
    df = pd.DataFrame(rows)
    all_cats = sorted({c for r in rows for c in (r["failure categories"].split(", ") if r["failure categories"] else [])})
    chosen = st.multiselect("Filter by failure category", all_cats)
    if chosen:
        df = df[df["failure categories"].apply(lambda s: any(c in s for c in chosen))]
    st.dataframe(df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ chrome: hero, nav, home
def hero(title, subtitle):
    st.markdown(f"## {title}")
    st.markdown(f'<p class="hero-sub">{subtitle}</p>', unsafe_allow_html=True)
    st.write("")


NAV = [("Home", "🏠"), ("Extract", "🔍"), ("Compare", "⚖️"), ("Evaluate", "📊")]


def current_page():
    """The page is kept in the URL (?page=...) so a refresh stays put and the
    browser Back/Forward buttons move between pages."""
    valid = {name for name, _ in NAV}
    page = st.query_params.get("page", "Home")
    return page if page in valid else "Home"


def go(page):
    st.query_params["page"] = page
    st.rerun()


def sidebar():
    with st.sidebar:
        st.markdown('<div class="logo"><div class="mark">🧾</div> ReceiptIE</div>', unsafe_allow_html=True)
        st.markdown('<span class="badge">● Runs locally</span>', unsafe_allow_html=True)
        st.write("")
        current = current_page()
        for name, icon in NAV:
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                         type="primary" if current == name else "secondary"):
                go(name)
        st.divider()
        st.caption("**Privacy:** the server binds to `localhost`, telemetry is off, and uploads are "
                   "processed in a temporary folder — nothing leaves this machine.")
        st.caption("BSc thesis · ELTE · models run on CPU (first load and the VLM are slow).")


def home():
    st.markdown("# 🧾 Receipt extraction")
    st.markdown('<p class="hero-sub">Small quantized LLMs for financial-document information extraction '
                '— a local, GDPR-compliant pipeline with a failure taxonomy.</p>', unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="pipeline">
      <div class="stage"><div class="ic">🧾</div><div class="nm">Receipt image</div><small>SROIE / CORD / upload</small></div>
      <div class="arrow">→</div>
      <div class="stage"><div class="ic">🔎</div><div class="nm">OCR</div><small>PaddleOCR / Tesseract</small></div>
      <div class="arrow">→</div>
      <div class="stage"><div class="ic">🧠</div><div class="nm">Small LLM</div><small>Phi-3 / Qwen / Llama (Q4)</small></div>
      <div class="arrow">→</div>
      <div class="stage"><div class="ic">✅</div><div class="nm">Validated JSON</div><small>4 fields, schema-checked</small></div>
      <div class="arrow">→</div>
      <div class="stage"><div class="ic">🏷️</div><div class="nm">Failure taxonomy</div><small>which stage failed, and why</small></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("An OCR-free vision-language model (Qwen2.5-VL) can also read the image end-to-end, "
               "skipping the OCR stage.")
    st.write("")

    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "🔍", "Extract & Explain", "Run one receipt and see every stage — OCR boxes, raw output, "
         "validated JSON, failure tags, and which stage failed.", "Extract"),
        (c2, "⚖️", "Compare models", "Run Phi-3, Qwen2.5-3B, Llama-3.2-3B and the OCR-free VLM side by "
         "side on one receipt: output, correctness and speed.", "Compare"),
        (c3, "📊", "Dataset Evaluation", "Score a single image, a built-in subset, or your own labelled "
         "ZIP — accuracy with confidence intervals and taxonomy counts.", "Evaluate"),
    ]
    for col, icon, title, body, target in cards:
        with col, st.container(border=True):
            st.markdown(f'<div class="feature"><h4>{icon} {title}</h4></div>', unsafe_allow_html=True)
            st.caption(body)
            if st.button(f"Open {title.split()[0]}", key=f"home_{target}", use_container_width=True):
                go(target)

    st.write("")
    with st.container(border=True):
        st.markdown("#### Why this exists")
        st.markdown(
            "Cloud document-AI sends financial receipts to third-party servers. This project keeps the "
            "whole pipeline **on one machine** with small **quantized** models, and studies *where and "
            "why* they fail — a **failure taxonomy** that attributes each error to the OCR stage or the "
            "LLM. The two-stage pipeline is interpretable; the OCR-free VLM is faster to reason about but "
            "a black box. Open **Evaluate** for the measured results, or **Extract** for a live run.")


def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    sidebar()
    page = current_page()
    if page == "Home":
        home()
    elif page == "Extract":
        tab_extract()
    elif page == "Compare":
        tab_compare()
    else:
        tab_evaluate()


main()
