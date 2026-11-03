#!/usr/bin/env python3
"""Figure 22 · Alpamayo-R1 backbone INT4 AWQ vs FP16: minADE6 / latency / engine size.

Data (measured 2026-11-03, orin-goat 64GB MAXN; full provenance in
eval/r1/int4_backbone.md §四):
  minADE6  chunk 3116, 18 clips x 6 seeds, compute_minade.py --num_traj_samples 6;
           CI = t(0.975,17)*sd/sqrt(18) recomputed from the per-clip CSVs on goat
           (gated data, CSVs stay on-device; only aggregates leave the machine).
  latency  llm_bench, identical engine params (maxInputLen 3424 / maxKV 4096 / batch 1).
  size     llm.engine file bytes.

Run:  .venv/bin/python eval/r1/int4_viz.py                   -> docs/figures/
      FIG_THEME=dark .venv/bin/python eval/r1/int4_viz.py    -> docs/figures/dark/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paper_style import (ANNOT, C, C4, CF, WIDTH_FULL, apply,  # noqa: E402
                         bar_kw, fig_legend, label_bars, out_dir)

apply()
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

ADE_X = ["minADE6@3s", "minADE6@6.4s"]
FP16_ADE, FP16_CI = [0.17460, 0.80401], [0.05323, 0.31983]
INT4_ADE, INT4_CI = [0.17590, 0.80046], [0.05755, 0.31526]

LAT_X = ["prefill@512", "decode@KV512"]
FP16_LAT = [277.9, 100.3]
INT4_LAT = [290.3, 39.8]

ENG = [15.19, 4.88]

fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 3.1))
x = np.arange(2)
w = 0.36
MUTED = C("#475569")

ax = axes[0]
b1 = ax.bar(x - w / 2, FP16_ADE, w, yerr=FP16_CI, color=CF, label="FP16",
            capsize=2.5, error_kw=dict(lw=0.8, ecolor=MUTED), **bar_kw())
b2 = ax.bar(x + w / 2, INT4_ADE, w, yerr=INT4_CI, color=C4, label="INT4 AWQ",
            capsize=2.5, error_kw=dict(lw=0.8, ecolor=MUTED), **bar_kw())
label_bars(ax, b1, fmt="{:.3f}")
label_bars(ax, b2, fmt="{:.3f}")
ax.annotate("Δ=+0.0013", (x[0], 0.31), ha="center", fontsize=ANNOT, color=MUTED)
ax.annotate("Δ=−0.0036（阈值 0.1）", (x[1], 1.27), ha="center", fontsize=ANNOT, color=MUTED)
ax.set_xticks(x)
ax.set_xticklabels(ADE_X)
ax.set_ylim(0, 1.42)
ax.set_ylabel("minADE6 (m)")
ax.set_title("(a) 轨迹精度（误差棒 = 95% CI）")

ax = axes[1]
b1 = ax.bar(x - w / 2, FP16_LAT, w, color=CF, label="FP16", **bar_kw())
b2 = ax.bar(x + w / 2, INT4_LAT, w, color=C4, label="INT4 AWQ", **bar_kw())
label_bars(ax, b1, fmt="{:.1f}")
label_bars(ax, b2, fmt="{:.1f}")
ax.annotate("0.96×", (x[0] + w / 2, 302), ha="center", fontsize=ANNOT, color=MUTED)
ax.annotate("2.52×", (x[1] + w / 2, 52), ha="center", fontsize=ANNOT, color=MUTED)
ax.set_xticks(x)
ax.set_xticklabels(LAT_X)
ax.set_ylim(0, 348)
ax.set_ylabel("延迟 (ms)")
ax.set_title("(b) llm 延迟（llm_bench）")

ax = axes[2]
b = ax.bar(["FP16", "INT4 AWQ"], ENG, 0.5, color=[CF, C4], **bar_kw())
label_bars(ax, b, fmt="{:.2f}")
ax.annotate("3.11× 小", (1, 6.4), ha="center", fontsize=ANNOT, color=MUTED)
ax.set_ylim(0, 18)
ax.set_ylabel("llm.engine (GB)")
ax.set_title("(c) llm 引擎体积")

fig_legend(fig, axes[0], ncol=2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = os.path.join(out_dir(HERE), "r1_int4_quant.png")
fig.savefig(out)
print("wrote", out)
