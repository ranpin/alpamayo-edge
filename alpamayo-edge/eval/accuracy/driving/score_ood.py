#!/usr/bin/env python3
"""Score the driving-domain OOD eval (n=184, 8-way MC, chance 12.5%).

Companion to ../benchmark/score_mc.py (n=210 robovqa/robofail): same parsing,
Wilson CI and exact McNemar implementations are imported from there so the two
benchmarks stay methodologically identical.

Ground truth: ../../../data/ood_eval/manifest.json (gate-3 artifact; class-balanced
8x23, answer letters balanced A-H x23, human-verified NVIDIA PhysicalAI-AV labels).

Usage:
    .venv/bin/python alpamayo-edge/eval/accuracy/driving/score_ood.py \
        --runs fp16=orin_fp16_outputs_n184.json int8=orin_int8_outputs_n184.json \
               int4=orin_int4_outputs_n184.json \
        [--reference fp16] [--out RESULTS.json]

The --reference run (default fp16) is what McNemar compares the others against.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmark"))
from score_mc import load_run, mcnemar_exact, parse_choice, wilson  # noqa: E402

MANIFEST = os.path.join(HERE, "..", "..", "..", "data", "ood_eval", "manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="tag=file.json ...")
    ap.add_argument("--reference", default="fp16",
                    help="run tag used as McNemar reference")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(MANIFEST))
    items = man["items"]
    n = len(items)
    classes = sorted({e["event_cluster"] for e in items})
    print(f"manifest: n={n}, {len(classes)} classes x {man['per_class']}, "
          f"chance {man['chance_baseline_pct']}%\n")

    runs = {}
    for spec in a.runs:
        tag, path = spec.split("=", 1)
        gens = load_run(os.path.join(HERE, path))
        if len(gens) != n:
            print(f"!! {tag}: {len(gens)} responses vs {n} items -- skipping")
            continue
        choices = [parse_choice(g, e["options"]) for e, g in zip(items, gens)]
        runs[tag] = {
            "correct": [c == e["answer"] for c, e in zip(choices, items)],
            "choices": choices,
            "unparsed": sum(1 for c in choices if c is None),
        }

    if a.reference not in runs:
        print(f"!! reference run '{a.reference}' missing")
        return 1
    ref = runs[a.reference]["correct"]

    results = {"n": n, "chance_pct": man["chance_baseline_pct"],
               "reference": a.reference, "runs": {}}
    for tag, r in runs.items():
        k = sum(r["correct"])
        lo, hi = wilson(k, n)
        b = sum(1 for f, q in zip(ref, r["correct"]) if f and not q)
        c = sum(1 for f, q in zip(ref, r["correct"]) if q and not f)
        p = mcnemar_exact(b, c) if tag != a.reference else None
        letters = collections.Counter(r["choices"])
        print(f"{tag:8} overall {100*k/n:5.1f}%  CI [{lo:.1f},{hi:.1f}]"
              + (f"  vs {a.reference}: {100*(k-sum(ref))/n:+.1f} pts, "
                 f"McNemar p={p:.3f} "
                 f"({'significant' if p < 0.05 else 'NOT significant'})"
                 if tag != a.reference else "")
              + (f"  unparsed={r['unparsed']}" if r["unparsed"] else ""))
        print(f"         letters: "
              + " ".join(f"{L}={letters.get(L,0)}" for L in "ABCDEFGH"))
        per_class = {}
        for cls in classes:
            idx = [i for i, e in enumerate(items) if e["event_cluster"] == cls]
            ks = sum(r["correct"][i] for i in idx)
            per_class[cls] = {"acc": 100 * ks / len(idx), "n": len(idx)}
            print(f"         {cls:42} {100*ks/len(idx):5.1f}%  (n={len(idx)})")
        per_split = {}
        for sp in sorted({e["split"] for e in items}):
            idx = [i for i, e in enumerate(items) if e["split"] == sp]
            ks = sum(r["correct"][i] for i in idx)
            per_split[sp] = {"acc": 100 * ks / len(idx), "n": len(idx)}
            print(f"         split={sp:5} {100*ks/len(idx):5.1f}%  (n={len(idx)})")
        print()
        results["runs"][tag] = {
            "overall": 100 * k / n, "ci": [lo, hi], "unparsed": r["unparsed"],
            "letter_dist": {L: letters.get(L, 0) for L in "ABCDEFGH"},
            "mcnemar_p_vs_ref": p, "discordant_b_c": [b, c],
            "per_class": per_class, "per_split": per_split,
        }

    if a.out:
        json.dump(results, open(os.path.join(HERE, a.out), "w"), indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
