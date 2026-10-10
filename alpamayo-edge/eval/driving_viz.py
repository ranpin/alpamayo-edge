#!/usr/bin/env python3
"""Driving-domain OOD eval figure (n=184, 8-way MC).

Reads eval/accuracy/driving/RESULTS.json (committed, produced by score_ood.py)
so the figure regenerates deterministically from data, not hand-transcribed
numbers. Style follows bench_viz.py / paper_style.py.

Run:  .venv/bin/python alpamayo-edge/eval/driving_viz.py
Out:  alpamayo-edge/docs/figures/driving_accuracy.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply, bar_kw, fig_legend, ANNOT, C4, C8, CF,
                         WIDTH_FULL)

apply()
import matplotlib.pyplot as plt  # noqa: E402  (after apply(), style is set)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
RES = os.path.join(HERE, "accuracy", "driving", "RESULTS.json")

SHORT = {
    "WORK_ZONES_TEMP_TRAFFIC_CONTROL": "work\nzones",
    "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY": "pedes-\ntrians",
    "COMPLEX_INTERSECTION_INTERACTION": "inter-\nsections",
    "EMERGENCY_INCIDENT_SCENE": "emergen-\ncy",
    "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR": "special\nvehicle",
    "ANIMALS_BIRDS_ROADKILL": "animals",
    "ROAD_DEBRIS_OR_SAFETY_TRACES": "road\ndebris",
    "CYCLISTS_AND_MICROMOBILITY_COMPLEX": "cyclists",
}

r = json.load(open(RES))
runs = r["runs"]
n_all = r["n"]
chance = r["chance_pct"]

# order classes by FP16 accuracy (desc), then ALL
per_class = runs["fp16"]["per_class"]
order = sorted(per_class, key=lambda c: -per_class[c]["acc"])
groups = [SHORT[c] for c in order] + ["ALL\n(n=184)"]
NS = [per_class[c]["n"] for c in order] + [n_all]
series = {}
for tag in ("fp16", "int8", "int4"):
    vals = [runs[tag]["per_class"][c]["acc"] for c in order]
    vals.append(runs[tag]["overall"])
    series[tag] = vals

import math  # noqa: E402


def wilson_ci(pct, n, z=1.959963985):
    """Asymmetric Wilson error-bar offsets around pct (same impl as bench_viz)."""
    p = pct / 100
    d = 1 + z * z / n
    centre = 100 * (p + z * z / (2 * n)) / d
    half = 100 * z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return pct - (centre - half), (centre + half) - pct


fig, ax = plt.subplots(figsize=(WIDTH_FULL, 4.4))
x = np.arange(len(groups)); w = 0.26
for off, tag, c, lab in [(-w, "fp16", CF, "FP16 (ref)"),
                         (0, "int8", C8, "INT8 (SmoothQuant)"),
                         (w, "int4", C4, "INT4 (AWQ)")]:
    vals = series[tag]
    err = np.array([wilson_ci(v, n) for v, n in zip(vals, NS)]).T
    bars = ax.bar(x + off, vals, w, color=c, label=lab, yerr=err, capsize=2.5,
                  error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="#333333"),
                  **bar_kw())
    for r_, v, up in zip(bars, vals, err[1]):
        ax.annotate(f"{v:.1f}", (r_.get_x() + r_.get_width() / 2, v + up),
                    ha="center", va="bottom", fontsize=ANNOT,
                    xytext=(0, 2), textcoords="offset points")

ax.axhline(runs["fp16"]["overall"], color=CF, lw=1.0, ls="--", zorder=1)
ax.text(3.5, runs["fp16"]["overall"], f"FP16 overall {runs['fp16']['overall']:.1f}%",
        color=CF, fontsize=ANNOT, ha="center", va="bottom")
ax.axhline(chance, color="#94a3b8", lw=1.0, ls=":", zorder=1)
ax.text(7.6, chance, f"chance {chance}%", color="#64748b", fontsize=ANNOT,
        ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("8-way multiple-choice accuracy (%)")
ax.set_ylim(0, 104)
ax.set_title("Driving-domain OOD scene-classification accuracy, "
             "PhysicalAI-AV (n = 184)")
fig_legend(fig, ax)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT, "driving_accuracy.png"))
plt.close(fig)
print("wrote", os.path.join(OUT, "driving_accuracy.png"))
