#!/usr/bin/env python3
"""Resolution-Pareto figure (site fig. 21): accuracy / latency / energy per tier.

Reads (all committed):
    RESULTS_pareto_int4.json, RESULTS_pareto_int8.json   score_ood.py, reference = t448
    pareto_timing_power_int4.json, pareto_timing_power_int8.json   parse_pareto.py

Three panels, x = frame-resolution tier (448 / 720p / native 1080p), series = INT4 / INT8:
    A: 8-way MC accuracy (Wilson 95% CI), chance line, FP16@448 reference line
    B: E2E latency per request (s)
    C: energy per request (J)
Missing cells (e.g. INT8@1080p if unavailable) are skipped.

Run:  .venv/bin/python alpamayo-edge/eval/accuracy/driving/pareto/pareto_viz.py
Out:  alpamayo-edge/docs/figures/resolution_pareto.png
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from paper_style import apply, bar_kw, ANNOT, C4, C8, CF, WIDTH_FULL  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "..", "docs", "figures")

TIERS = ["448", "720p", "1080p"]
RUNKEY = {"448": "t448", "720p": "t720", "1080p": "t1080"}


def load(tag):
    p = os.path.join(HERE, f"RESULTS_pareto_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_tim(tag):
    p = os.path.join(HERE, f"pareto_timing_power_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


acc = {t: load(t) for t in ("int4", "int8")}
tim = {t: load_tim(t) for t in ("int4", "int8")}

fp16_448 = None
r448 = os.path.join(HERE, "..", "RESULTS.json")
if os.path.exists(r448):
    r = json.load(open(r448))
    if "fp16" in r.get("runs", {}):
        fp16_448 = r["runs"]["fp16"]["overall"]


def wilson_ci(pct, n, z=1.959963985):
    p = pct / 100
    d = 1 + z * z / n
    centre = 100 * (p + z * z / (2 * n)) / d
    half = 100 * z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return pct - (centre - half), (centre + half) - pct


def series_vals(kind):
    """kind in {'acc','e2e','j'} -> {tag: [v or None per tier]}"""
    out = {}
    for tag in ("int4", "int8"):
        vals = []
        for tier in TIERS:
            v = None
            if kind == "acc" and acc[tag] and RUNKEY[tier] in acc[tag]["runs"]:
                v = acc[tag]["runs"][RUNKEY[tier]]["overall"]
            elif kind != "acc" and tim[tag]:
                key = f"{tag}_{'448' if tier=='448' else tier}"
                if key in tim[tag]:
                    v = tim[tag][key]["e2e_mean_s" if kind == "e2e" else "j_per_request"]
            vals.append(v)
        out[tag] = vals
    return out


fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(WIDTH_FULL, 4.0))
x = np.arange(len(TIERS)); w = 0.36
panels = [(a1, series_vals("acc"), "8-way MC accuracy (%)", "Accuracy (n = 184/tier)"),
          (a2, series_vals("e2e"), "E2E latency (s / request)", "Latency per request"),
          (a3, series_vals("j"), "energy (J / request)", "Energy per request")]

for ax, series, ylabel, title in panels:
    for off, tag, c, lab in [(-w / 2, "int4", C4, "INT4 (AWQ)"),
                             (w / 2, "int8", C8, "INT8 (SQ)")]:
        vals = series[tag]
        idx = [i for i, v in enumerate(vals) if v is not None]
        if not idx:
            continue
        vv = [vals[i] for i in idx]
        if ax is a1:
            n = acc[tag]["n"]
            err = np.array([wilson_ci(v, n) for v in vv]).T
            bars = ax.bar(x[idx] + off, vv, w, color=c, label=lab, yerr=err,
                          capsize=2.5, error_kw=dict(elinewidth=0.8, capthick=0.8,
                                                     ecolor="#333333"), **bar_kw())
            for r, v, up in zip(bars, vv, err[1]):
                ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, v + up),
                            ha="center", va="bottom", fontsize=ANNOT,
                            xytext=(0, 2), textcoords="offset points")
        else:
            bars = ax.bar(x[idx] + off, vv, w, color=c, label=lab, **bar_kw())
            fmt = "{:.2f}" if ax is a2 else "{:.0f}"
            for r, v in zip(bars, vv):
                ax.annotate(fmt.format(v), (r.get_x() + r.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=ANNOT,
                            xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(TIERS)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.margins(y=0.18)

if fp16_448 is not None:
    a1.axhline(fp16_448, color=CF, lw=1.0, ls="--", zorder=1)
    a1.text(1.0, fp16_448, f"FP16 @448 {fp16_448:.1f}%", color=CF, fontsize=ANNOT,
            ha="center", va="bottom")
a1.axhline(12.5, color="#94a3b8", lw=1.0, ls=":", zorder=1)
a1.text(2.0, 12.5, "chance 12.5%", color="#64748b", fontsize=ANNOT,
        ha="center", va="bottom")
a1.set_ylim(0, 100)
a1.legend(loc="upper left", fontsize=ANNOT, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "resolution_pareto.png"))
plt.close(fig)
print("wrote", os.path.join(OUT, "resolution_pareto.png"))
