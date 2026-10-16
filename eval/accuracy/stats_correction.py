#!/usr/bin/env python3
"""Y1 statistical correction (2026-10-16): family-wise multiple-comparison
correction + TOST equivalence, recomputed from raw per-item outputs.

Why this exists
---------------
The interview-expert review (finding Y1) flagged two statistical vulnerabilities:

  (a) The resolution-Pareto headline cites the UNCORRECTED p=0.047 (INT8 720p
      vs 448) as the key evidence for "the knee is at 720p", while the Pareto
      README's own pre-registered Bonferroni (alpha=0.0125 over 4 vs-448 tests)
      already shows 720p does NOT survive -- only 1080p (p=0.011) does. The
      top-level README/site present p=0.047 without this caveat: an internal
      contradiction an interviewer catches by reading both documents.

  (b) "Quantization causes no significant accuracy loss" rests on failing to
      reject the null with an underpowered design (MDE ~7 pts). Absence of
      evidence is not evidence of absence: a real 3-4 pt loss would be
      invisible here. The fix is a TOST equivalence test that turns the claim
      into a positive, bounded statement (or honestly reports where equivalence
      within the MDE cannot be established).

What this script does
---------------------
  1. Recomputes every McNemar test from the raw per-item outputs, verifying the
     discordant b/c counts already committed in RESULTS*.json (consistency check).
  2. Groups tests into their PRE-REGISTERED families and applies Bonferroni and
     Benjamini-Hochberg FDR within each family.
  3. Adds TOST equivalence tests for the quantization-no-loss claims, with the
     pre-specified margin delta = the study's own MDE (~7 pts), plus a delta
     sensitivity sweep (5/7/10 pts) so the equivalence boundary is visible
     rather than cherry-picked.

No scipy: exact McNemar (math.comb), paired-difference normal CI, BH-FDR and
TOST are implemented inline, matching score_mc.py's dependency-free style.

Usage:
    .venv/bin/python eval/accuracy/stats_correction.py [--out corrected_stats.json]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "benchmark"))
from score_mc import load_run, mcnemar_exact, parse_choice  # noqa: E402

DRIVING_MANIFEST = os.path.join(HERE, "..", "..", "data", "ood_eval",
                                "manifest.json")
DRIVING_DIR = os.path.join(HERE, "driving")
PARETO_DIR = os.path.join(DRIVING_DIR, "pareto")
BENCH_DIR = os.path.join(HERE, "benchmark")
BENCH_MANIFEST = os.path.join(BENCH_DIR, "cosmos_bench210_manifest.json")

# Pre-specified equivalence margin = the study's own MDE (~7 pts at n=184/210).
# You cannot claim equivalence finer than the smallest effect the design can
# detect; sweeping 5/7/10 below exposes the boundary instead of hiding it.
DELTA_PRESET_PTS = 7.0
DELTA_SWEEP_PTS = (5.0, 7.0, 10.0)
ALPHA = 0.05
Z_90 = 1.6448536269514722   # one-sided 5% => 90% two-sided CI for TOST
Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Correctness vectors from raw outputs
# ---------------------------------------------------------------------------
def driving_correctness():
    """Return {tag: [bool x184]} for the driving-domain runs (incl. Pareto)."""
    man = json.load(open(DRIVING_MANIFEST))
    items = man["items"]
    n = len(items)
    runs = {
        "fp16": os.path.join(DRIVING_DIR, "orin_fp16_outputs_n184.json"),
        "int8": os.path.join(DRIVING_DIR, "orin_int8_outputs_n184.json"),
        "int4": os.path.join(DRIVING_DIR, "orin_int4_outputs_n184.json"),
        "int8_t448": os.path.join(PARETO_DIR, "orin_int8_448_outputs_n184.json"),
        "int8_t720": os.path.join(PARETO_DIR, "orin_int8_720_outputs_n184.json"),
        "int8_t1080": os.path.join(PARETO_DIR, "orin_int8_1080_outputs_n184.json"),
        "int4_t448": os.path.join(PARETO_DIR, "orin_int4_448_outputs_n184.json"),
        "int4_t720": os.path.join(PARETO_DIR, "orin_int4_720_outputs_n184.json"),
        "int4_t1080": os.path.join(PARETO_DIR, "orin_int4_1080_outputs_n184.json"),
    }
    out = {}
    for tag, path in runs.items():
        gens = load_run(path)
        assert len(gens) == n, f"{tag}: {len(gens)} != {n}"
        out[tag] = [parse_choice(g, e["options"]) == e["answer"]
                    for e, g in zip(items, gens)]
    return out, n


def robot_correctness():
    """Return {tag: [bool x210]} for the robot-domain runs (fp16 from manifest)."""
    man = json.load(open(BENCH_MANIFEST))["manifest"]
    n = len(man)
    out = {"fp16": [e.get("fp16_pred") == e["answer"] for e in man]}
    for tag, path in (("int8", "orin_int8_outputs_n210.json"),
                      ("int4", "orin_int4_outputs_n210.json")):
        gens = load_run(os.path.join(BENCH_DIR, path))
        assert len(gens) == n, f"{tag}: {len(gens)} != {n}"
        out[tag] = [parse_choice(g, e.get("options") or {}) == e["answer"]
                    for e, g in zip(man, gens)]
    return out, n


# ---------------------------------------------------------------------------
# Paired comparison statistics
# ---------------------------------------------------------------------------
def paired_stats(ref, test, n):
    """McNemar + paired-difference stats for two correctness vectors.

    b = ref right / test wrong; c = test right / ref wrong.
    diff = acc(test) - acc(ref) = (c - b) / n.
    SE of the paired difference (standard McNemar form).
    """
    b = sum(1 for f, q in zip(ref, test) if f and not q)
    c = sum(1 for q, f in zip(test, ref) if q and not f)
    p = mcnemar_exact(b, c)
    diff = (c - b) / n * 100.0                      # percentage points
    se = math.sqrt(max(0.0, (b + c) - (c - b) ** 2 / n)) / n * 100.0
    return {"b": b, "c": c, "mcnemar_p": p, "diff_pts": diff, "se_pts": se}


def tost(diff, se, delta_pts, z=Z_90):
    """Two-one-sided-tests equivalence for the paired difference.

    Equivalence at one-sided alpha iff the (1-2alpha)=90% CI lies inside
    [-delta, +delta]. Returns (pass, ci_low, ci_high).
    """
    lo = diff - z * se
    hi = diff + z * se
    return (lo > -delta_pts and hi < delta_pts), lo, hi


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values. pvals: list[(label, p)] -> dict label->q."""
    m = len(pvals)
    order = sorted(pvals, key=lambda x: x[1])
    q = {}
    running = 1.0
    for rank_from_top, (label, p) in enumerate(reversed(order)):
        j = m - rank_from_top                      # ascending rank 1..m
        val = min(running, p * m / j)
        running = val
        q[label] = min(1.0, val)
    return q


def bonferroni(p, m):
    return min(1.0, p * m)


# ---------------------------------------------------------------------------
# Committed b/c cross-check (from RESULTS*.json)
# ---------------------------------------------------------------------------
def committed_bc():
    """Pull the discordant counts already committed, for verification."""
    got = {}
    d = json.load(open(os.path.join(DRIVING_DIR, "RESULTS.json")))["runs"]
    got[("driving", "int8_vs_fp16")] = tuple(d["int8"]["discordant_b_c"])
    got[("driving", "int4_vs_fp16")] = tuple(d["int4"]["discordant_b_c"])
    p8 = json.load(open(os.path.join(PARETO_DIR, "RESULTS_pareto_int8.json")))["runs"]
    got[("pareto", "int8_t720_vs_t448")] = tuple(p8["t720"]["discordant_b_c"])
    got[("pareto", "int8_t1080_vs_t448")] = tuple(p8["t1080"]["discordant_b_c"])
    p4 = json.load(open(os.path.join(PARETO_DIR, "RESULTS_pareto_int4.json")))["runs"]
    got[("pareto", "int4_t720_vs_t448")] = tuple(p4["t720"]["discordant_b_c"])
    got[("pareto", "int4_t1080_vs_t448")] = tuple(p4["t1080"]["discordant_b_c"])
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    drv, n_drv = driving_correctness()
    rob, n_rob = robot_correctness()
    committed = committed_bc()
    report = {"n_driving": n_drv, "n_robot": n_rob,
              "alpha": ALPHA, "delta_preset_pts": DELTA_PRESET_PTS,
              "families": {}, "verification": []}

    def cmp(domain_runs, n, test, ref):
        return paired_stats(domain_runs[ref], domain_runs[test], n)

    # --- consistency check: recomputed b/c vs committed RESULTS*.json ---
    checks = [
        (("driving", "int8_vs_fp16"), cmp(drv, n_drv, "int8", "fp16")),
        (("driving", "int4_vs_fp16"), cmp(drv, n_drv, "int4", "fp16")),
        (("pareto", "int8_t720_vs_t448"), cmp(drv, n_drv, "int8_t720", "int8_t448")),
        (("pareto", "int8_t1080_vs_t448"), cmp(drv, n_drv, "int8_t1080", "int8_t448")),
        (("pareto", "int4_t720_vs_t448"), cmp(drv, n_drv, "int4_t720", "int4_t448")),
        (("pareto", "int4_t1080_vs_t448"), cmp(drv, n_drv, "int4_t1080", "int4_t448")),
    ]
    print("=== Step 0: recomputed discordant b/c vs committed RESULTS*.json ===")
    all_match = True
    for key, st in checks:
        rec = (st["b"], st["c"])
        com = committed.get(key)
        match = com == rec
        all_match = all_match and match
        report["verification"].append({"comparison": f"{key[0]}:{key[1]}",
                                       "recomputed_b_c": list(rec),
                                       "committed_b_c": list(com) if com else None,
                                       "match": match})
        print(f"  {key[0]:7} {key[1]:24} recomputed b/c={rec}  "
              f"committed={com}  {'MATCH' if match else 'MISMATCH'}")
    print(f"  -> {'ALL MATCH' if all_match else 'SOME MISMATCH'}\n")

    # --- Family 1: Pareto resolution sweep (pre-registered 4 vs-448 tests) ---
    pareto_tests = {
        "int8_t720_vs_t448": cmp(drv, n_drv, "int8_t720", "int8_t448"),
        "int8_t1080_vs_t448": cmp(drv, n_drv, "int8_t1080", "int8_t448"),
        "int4_t720_vs_t448": cmp(drv, n_drv, "int4_t720", "int4_t448"),
        "int4_t1080_vs_t448": cmp(drv, n_drv, "int4_t1080", "int4_t448"),
    }
    m = len(pareto_tests)
    qvals = bh_fdr([(k, v["mcnemar_p"]) for k, v in pareto_tests.items()])
    print(f"=== Family 1: Pareto resolution sweep (pre-registered {m} vs-448 tests) ===")
    print(f"  Bonferroni alpha = {ALPHA}/{m} = {ALPHA/m:.4f};  BH-FDR q-values\n")
    fam1 = {}
    for k, st in sorted(pareto_tests.items(), key=lambda x: x[1]["mcnemar_p"]):
        pb = bonferroni(st["mcnemar_p"], m)
        sig_b = pb < ALPHA
        sig_fdr = qvals[k] < ALPHA
        fam1[k] = {**st, "p_bonferroni": pb, "q_bh": qvals[k],
                   "sig_bonferroni": sig_b, "sig_bh": sig_fdr}
        print(f"  {k:24} diff={st['diff_pts']:+.1f}pts  p={st['mcnemar_p']:.4f}  "
              f"p_bonf={pb:.4f} ({'SURVIVES' if sig_b else 'not'})  "
              f"q_bh={qvals[k]:.4f} ({'SURVIVES' if sig_fdr else 'not'})")
    posthoc = cmp(drv, n_drv, "int8_t1080", "int8_t720")
    print(f"\n  POST-HOC (excluded from family, not corrected): "
          f"int8_t1080_vs_t720 diff={posthoc['diff_pts']:+.1f}pts "
          f"p={posthoc['mcnemar_p']:.4f}")
    report["families"]["pareto_resolution"] = {
        "pre_registered_m": m, "tests": fam1,
        "post_hoc_int8_1080_vs_720": posthoc}

    # --- Family 2+3: quantization no-loss (driving + robot) + TOST ---
    quant = {
        "driving:int8_vs_fp16": (cmp(drv, n_drv, "int8", "fp16"), n_drv),
        "driving:int4_vs_fp16": (cmp(drv, n_drv, "int4", "fp16"), n_drv),
        "robot:int8_vs_fp16": (cmp(rob, n_rob, "int8", "fp16"), n_rob),
        "robot:int4_vs_fp16": (cmp(rob, n_rob, "int4", "fp16"), n_rob),
    }
    print(f"\n=== Families 2-3: quantization no-loss (McNemar + TOST equivalence) ===")
    print(f"  TOST margin delta preset = {DELTA_PRESET_PTS} pts (= study MDE); "
          f"sweep {DELTA_SWEEP_PTS}\n")
    fam23 = {}
    for k, (st, n) in quant.items():
        sweep = {}
        for d in DELTA_SWEEP_PTS:
            ok, lo, hi = tost(st["diff_pts"], st["se_pts"], d)
            sweep[str(d)] = {"equivalent": ok, "ci90": [lo, hi]}
        ok_preset, lo, hi = tost(st["diff_pts"], st["se_pts"], DELTA_PRESET_PTS)
        fam23[k] = {**st, "tost_delta_sweep": sweep,
                    "tost_preset_equivalent": ok_preset,
                    "ci90_pts": [lo, hi]}
        print(f"  {k:24} diff={st['diff_pts']:+.1f}pts  p={st['mcnemar_p']:.3f} "
              f"(ns)  90%CI [{lo:+.1f},{hi:+.1f}]  "
              f"TOST(d={DELTA_PRESET_PTS:.0f})={'EQUIVALENT' if ok_preset else 'NOT established'}")
        print(f"  {'':24} delta-sweep: " + "  ".join(
            f"{d:.0f}pts={'PASS' if sweep[str(d)]['equivalent'] else 'FAIL'}"
            for d in DELTA_SWEEP_PTS))
    report["families"]["quantization_no_loss"] = fam23

    # --- Verdict ---
    print("\n=== Corrected verdict ===")
    p720 = fam1["int8_t720_vs_t448"]
    p1080 = fam1["int8_t1080_vs_t448"]
    print(f"  * Pareto: only INT8 1080p-vs-448 survives family-wise correction "
          f"(p_bonf={p1080['p_bonferroni']:.4f}). The 720p headline "
          f"(p={p720['mcnemar_p']:.4f}) does NOT survive Bonferroni "
          f"(alpha={ALPHA/m:.4f}); it is 'suggestive / borderline', not "
          f"significant. 'Knee at 720p' must be framed as a cost/benefit "
          f"argument, not a significance result.")
    n_eq = sum(1 for v in fam23.values() if v["tost_preset_equivalent"])
    print(f"  * Quantization no-loss: {n_eq}/4 comparisons establish equivalence "
          f"within delta={DELTA_PRESET_PTS:.0f} pts (TOST). Where TOST fails, the "
          f"honest claim is 'no loss detected; data compatible with a true loss "
          f"up to the 90% CI bound', not 'no loss'.")

    if a.out:
        json.dump(report, open(os.path.join(HERE, a.out), "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
