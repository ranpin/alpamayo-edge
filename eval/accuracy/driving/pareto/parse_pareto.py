#!/usr/bin/env python3
"""Parse resolution-Pareto run artifacts pulled from orin-dog.

Inputs (fetched into this directory, names preserved):
    run_pareto_<tag>.log    llm_inference log; one "Processing vision inputs"
                            line per request with ms timestamps (structure
                            verified on the 2026-10-09 n=184 logs), plus the
                            final "Processing complete" line.
    pw_pareto_<tag>.log     tegrastats --interval 250, started by run_pareto.sh
                            only AFTER the first request began processing and
                            stopped right after llm_inference exited, so every
                            sample lies inside the processing window.

Per-request E2E = delta between consecutive request-start timestamps; the last
request ends at "Processing complete". E2E ~= TTFT + 1-2 decode tokens (the MC
answer is a single letter), which is the deployment-relevant number anyway.

Power fields are the same three rails measure_power.sh uses:
VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 (mW).

Usage:
    .venv/bin/python eval/accuracy/driving/pareto/parse_pareto.py \
        --tags int4_448 int4_720 int4_1080 [--out pareto_timing_power.json]
"""
import argparse
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

TS = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]")
RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")


def log_seconds(line):
    m = TS.match(line)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def parse_run(tag):
    path = os.path.join(HERE, f"run_pareto_{tag}.log")
    starts, complete = [], None
    for line in open(path):
        if "Processing vision inputs" in line:
            starts.append(log_seconds(line))
        elif "Processing complete" in line:
            complete = log_seconds(line)
    if any(t is None for t in starts) or complete is None:
        raise SystemExit(f"{tag}: unparsable timestamps "
                         f"(starts={len(starts)}, complete={complete})")
    if starts[0] > starts[-1]:  # crossed midnight
        raise SystemExit(f"{tag}: midnight rollover not handled")
    e2e = [b - a for a, b in zip(starts, starts[1:])]
    e2e.append(complete - starts[-1])
    wall = complete - starts[0]
    return {"n_requests": len(starts), "e2e_s": e2e, "wall_s": wall}


def parse_power(tag):
    path = os.path.join(HERE, f"pw_pareto_{tag}.log")
    totals, gpus = [], []
    for line in open(path):
        mw = {}
        for rail in RAILS:
            m = re.search(rail + r"\s+(\d+)mW", line)
            if m:
                mw[rail] = int(m.group(1))
        if len(mw) == len(RAILS):
            totals.append(sum(mw.values()) / 1000)   # W
            gpus.append(mw["VDD_GPU_SOC"] / 1000)
        elif "GR3D_FREQ" in line:
            # some JetPack builds name rails differently; keep count for honesty
            totals.append(None)
    good = [t for t in totals if t is not None]
    if not good:
        raise SystemExit(f"{tag}: no parsable power samples in {path}")
    return {"n_samples": len(good), "n_unparsed": len(totals) - len(good),
            "avg_total_W": statistics.mean(good),
            "avg_gpu_soc_W": statistics.mean(gpus),
            "peak_total_W": max(good)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = {}
    for tag in a.tags:
        run = parse_run(tag)
        pw = parse_power(tag)
        n = run["n_requests"]
        e2e = run["e2e_s"]
        j_req = pw["avg_total_W"] * run["wall_s"] / n
        results[tag] = {
            "n_requests": n,
            "e2e_mean_s": statistics.mean(e2e),
            "e2e_median_s": statistics.median(e2e),
            "e2e_std_s": statistics.stdev(e2e) if n > 1 else 0.0,
            "e2e_min_s": min(e2e), "e2e_max_s": max(e2e),
            "wall_s": run["wall_s"],
            **pw,
            "j_per_request": j_req,
        }
        r = results[tag]
        print(f"{tag:14} n={n}  E2E mean {r['e2e_mean_s']:.3f}s "
              f"(median {r['e2e_median_s']:.3f}, std {r['e2e_std_s']:.3f}, "
              f"min {r['e2e_min_s']:.3f}, max {r['e2e_max_s']:.3f})  "
              f"P avg {r['avg_total_W']:.1f}W (peak {r['peak_total_W']:.1f}W, "
              f"{r['n_samples']} samples)  J/req {j_req:.1f}")

    if a.out:
        json.dump(results, open(os.path.join(HERE, a.out), "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
