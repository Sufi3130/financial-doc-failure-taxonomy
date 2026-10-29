"""Render the thesis figures from committed result JSONs (no run dirs needed).

    python evaluation/make_figures.py          # writes docs/figures/*.png + *.pdf

Reads evaluation/results/model_accuracy.json (see figure_data.py) and
evaluation/failure_taxonomy/by_model_counts.json. Data-driven: to update after a
new/full-dataset run, refresh those JSONs and re-run this — the figures
regenerate, nothing here is hand-edited.

Palette and chart rules follow the data-viz skill: colour = model (identity),
fixed order, colourblind-safe first-three slots; thin bars; Wilson 95% CI error
bars; direct value labels; recessive hairline grid; legend for the three series.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGDIR = REPO / "docs" / "figures"

# model -> colour (validated categorical slots 1/2/3, light surface; CVD-safe all-pairs)
MODELS = ["Phi-3", "Qwen2.5-3B", "Llama-3.2-3B"]
COLOR = {"Phi-3": "#2a78d6", "Qwen2.5-3B": "#eb6834", "Llama-3.2-3B": "#1baf7a"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"

CAT_LABEL = {"wrong_entity": "Wrong\nentity", "wrong_total": "Wrong\ntotal",
             "boundary_too_wide": "Boundary\ntoo wide", "boundary_too_narrow": "Boundary\ntoo narrow",
             "partial_value": "Partial\nvalue", "malformed_json": "Malformed\nJSON",
             "wrong_value": "Wrong\nvalue", "wrong_date_format": "Wrong date\nformat",
             "copy_corruption": "Copy\ncorruption"}
OCR_STAGE = {"value_not_in_ocr"}  # OCR-stage, model-independent — excluded from the model figure


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASELINE,
    })


def _bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.tick_params(length=0)


def _label(ax, x, top, text, dy=1.5):
    ax.text(x, top + dy, text, ha="center", va="bottom", fontsize=8, color=INK)


def grouped(ax, groups, series, colors, err=None, fmt="{:.0f}", width=0.26, gap=0.02):
    """series: {name: [values per group]}; err: {name: [(lo,hi) per group]} or None."""
    n = len(series)
    base = np.arange(len(groups))
    offs = (np.arange(n) - (n - 1) / 2) * (width + gap)
    for i, (name, vals) in enumerate(series.items()):
        xs = base + offs[i]
        bars = ax.bar(xs, vals, width=width, color=colors[name], label=name,
                      edgecolor=SURFACE, linewidth=1.2, zorder=3)
        tops = list(vals)
        if err is not None:
            lo = [v - e[0] for v, e in zip(vals, err[name])]
            hi = [e[1] - v for v, e in zip(vals, err[name])]
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2,
                        elinewidth=1.0, capsize=2.5, zorder=4)
            tops = [e[1] for e in err[name]]
        for x, v, t in zip(xs, vals, tops):
            if v > 0:
                _label(ax, x, t, fmt.format(v))
    ax.set_xticks(base)


def fig_model_accuracy(acc):
    style()
    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1.25], wspace=0.18)
    axS, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # SROIE: four fields + all-4
    fields = ["company", "date", "address", "total"]
    groups = ["Company", "Date", "Address", "Total", "All 4\nfields"]
    series, err = {}, {}
    for m in MODELS:
        d = acc["SROIE"][m]
        series[m] = [d["fields"][f]["acc"] for f in fields] + [d["all_fields"]["acc"]]
        err[m] = [tuple(d["fields"][f]["ci"]) for f in fields] + [tuple(d["all_fields"]["ci"])]
    grouped(axS, groups, series, COLOR, err=err)
    axS.set_xticklabels(groups)
    axS.set_ylim(0, 108); axS.set_ylabel("Normalised match (%)")
    axS.set_title("SROIE Task 3", fontsize=11, color=INK, loc="left", pad=8)

    # CORD: total only
    cs, ce = {}, {}
    for m in MODELS:
        d = acc["CORD"][m]
        cs[m] = [d["all_fields"]["acc"]]; ce[m] = [tuple(d["all_fields"]["ci"])]
    grouped(axC, ["Total"], cs, COLOR, err=ce, width=0.22, gap=0.02)
    axC.set_xticklabels(["Total"])
    axC.set_ylim(0, 108); axC.set_yticklabels([])
    axC.set_title("CORD", fontsize=11, color=INK, loc="left", pad=8)
    for ax in (axS, axC):
        _bare(ax)

    axS.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.04),
               fontsize=9, handlelength=1.1, columnspacing=1.4)
    fig.suptitle("Field-level extraction accuracy by model", x=0.5, y=0.99,
                 fontsize=12.5, color=INK, ha="center")
    fig.text(0.5, 0.005,
             "PaddleOCR · v1 prompt · 2-shot · n = 50 per cell · error bars: Wilson 95% CI",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    _save(fig, "model_accuracy")


def fig_taxonomy(counts):
    style()
    cell = counts["SROIE"]["paddleocr"]
    cats = {}
    for m in MODELS:
        for c, n in cell[m]["by_category"].items():
            if c not in OCR_STAGE:
                cats[c] = cats.get(c, 0) + n
    order = sorted(cats, key=lambda c: cats[c], reverse=True)
    groups = [CAT_LABEL.get(c, c.replace("_", "\n")) for c in order]
    series = {m: [cell[m]["by_category"].get(c, 0) for c in order] for m in MODELS}

    fig, ax = plt.subplots(figsize=(10, 4.4))
    grouped(ax, groups, series, COLOR, width=0.26, gap=0.02)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, max(cats.values()) * 1.2 if cats else 1)
    ax.set_ylabel("Failing fields (count)")
    _bare(ax)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.04),
              fontsize=9, handlelength=1.1, columnspacing=1.4)
    fig.suptitle("Model-stage failures on SROIE", x=0.5, y=0.99, fontsize=12.5, color=INK)
    fig.text(0.5, 0.005,
             "PaddleOCR · counts are failing field-instances across 50 receipts × 4 fields · "
             "excludes OCR-stage value_not_in_ocr (33/33/32, model-independent)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    _save(fig, "taxonomy_sroie")


def _hdeltas(ax, rows, colors, xlim):
    """Horizontal signed-delta bars. rows: [(label, delta, colour)], top row first."""
    ys = np.arange(len(rows))[::-1]  # first row at the top
    for y, (label, d, col) in zip(ys, rows):
        if abs(d) < 0.5:
            ax.plot(0, y, "o", ms=4, color=col, zorder=4)
            ax.text(0.6, y, "0", ha="left", va="center", fontsize=7.5, color=MUTED)
        else:
            ax.barh(y, d, height=0.62, color=col, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            ha = "left" if d > 0 else "right"
            dx = 0.6 if d > 0 else -0.6
            ax.text(d + dx, y, f"{d:+.0f}", ha=ha, va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK2)
    ax.set_xlim(*xlim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(length=0)


def fig_experiment_deltas(prompt, fewshot, grammar):
    """Three interventions, each as the change in all-fields accuracy per cell."""
    style()
    order = [("phi3", "Phi-3"), ("qwen2.5-3b", "Qwen2.5-3B"), ("llama3.2-3b", "Llama-3.2-3B")]
    cells = [(f"{disp} · {ds}", f"{ds}/{reg}", disp)
             for reg, disp in order for ds in ("SROIE", "CORD")]

    def rows(data, lo, hi):
        out = []
        for label, key, disp in cells:
            d = data[key]["all_fields"]
            out.append((label, d[hi] - d[lo], COLOR[disp]))
        return out

    panels = [
        ("Switch prompt v1 → v2", rows(prompt, "v1", "v2")),
        ("Remove few-shot (2 → 0)", rows(fewshot, "2shot", "0shot")),
        ("Enable grammar (off → on)", rows(grammar, "off", "on")),
    ]
    xlim = (-28, 16)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharex=True)
    for ax, (title, rws) in zip(axes, panels):
        _hdeltas(ax, rws, COLOR, xlim)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    axes[1].set_xlabel("Change in all-fields accuracy (points)")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.suptitle("Effect of each intervention on extraction accuracy", x=0.5, y=0.99,
                 fontsize=12.5, color=INK)
    fig.text(0.5, 0.005,
             "positive = the change raised accuracy  ·  colour = model  ·  n = 50 per cell  ·  "
             "baseline: v1 prompt, 2-shot, grammar off",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    _save(fig, "experiment_deltas")


def fig_miss_source(counts):
    """SROIE failures split by stage: OCR (value_not_in_ocr) vs model (the rest), per OCR engine."""
    style()
    OCR = "#898781"   # upstream / contextual
    LLM = COLOR["Phi-3"]  # blue = the stage this thesis studies
    engines = [("paddleocr", "PaddleOCR"), ("tesseract", "Tesseract")]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    base = np.arange(len(MODELS))
    for ax, (eng, title) in zip(axes, engines):
        cell = counts["SROIE"][eng]
        ocr = [cell[m]["by_category"].get("value_not_in_ocr", 0) for m in MODELS]
        llm = [cell[m]["total"] - o for m, o in zip(MODELS, ocr)]
        b1 = ax.bar(base, ocr, width=0.6, color=OCR, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                    label="OCR stage (text missing)")
        ax.bar(base, llm, width=0.6, bottom=ocr, color=LLM, edgecolor=SURFACE, linewidth=1.2,
               zorder=3, label="Model stage (extraction)")
        for x, o, l in zip(base, ocr, llm):
            if o > 0:
                ax.text(x, o / 2, str(o), ha="center", va="center", fontsize=8, color=INK)
            if l > 0:
                ax.text(x, o + l / 2, str(l), ha="center", va="center", fontsize=8, color="#ffffff")
        ax.set_xticks(base)
        ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        _bare(ax)
    axes[0].set_ylabel("Failing field-instances (count)")

    axes[0].legend(frameon=False, ncol=1, loc="upper left", fontsize=8.5, handlelength=1.1)
    fig.suptitle("Where SROIE failures come from: OCR vs model stage", x=0.5, y=0.99,
                 fontsize=12.5, color=INK)
    fig.text(0.5, 0.005,
             "counts across 50 receipts × 4 fields  ·  weaker OCR (Tesseract) roughly doubles the "
             "upstream losses the model never gets to recover",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    _save(fig, "miss_source_sroie")


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"written: docs/figures/{name}.png + .pdf")


def _results(name):
    return json.loads((HERE / "results" / name).read_text(encoding="utf-8"))


def main():
    acc = _results("model_accuracy.json")
    counts = json.loads((REPO / "evaluation" / "failure_taxonomy" / "by_model_counts.json").read_text(encoding="utf-8"))
    fig_model_accuracy(acc)
    fig_taxonomy(counts)
    fig_experiment_deltas(_results("prompt_v1_vs_v2.json"),
                          _results("fewshot_2_vs_0.json"),
                          _results("grammar_on_vs_off.json"))
    fig_miss_source(counts)


if __name__ == "__main__":
    main()
