"""Score Alpamayo2-Edge predictions against chunk3116 GT (same protocol as R1 arm).

minADE6@6.4s = min over 6 candidates of mean 2D (xy) distance over 64 waypoints;
also reports @5s (first 50 waypoints) to relate to the official 0.52m figure
(different clip set, so not same-protocol).
"""
import json, csv
import numpy as np
from safetensors.numpy import load_file

W = "/home/nvidia/chenrunbin/a2e"
E = "/home/nvidia/chenrunbin/alpamayo-edge/eval_data"
T0 = 5_100_000

gt = json.load(open(f"{E}/gt_chunk3116.json"))
print("GT type:", type(gt).__name__, "n:", len(gt))
k0 = next(iter(gt))
print("GT sample keys:", list(gt[k0].keys()) if isinstance(gt[k0], dict) else np.array(gt[k0]).shape)

clips = [l.strip() for l in open(f"{E}/selected_clips.txt") if l.strip()]
rows = []
for clip in clips:
    f = f"{W}/eval_a2e/out/{clip}_{T0}/output.safetensors"
    try:
        t = load_file(f)
    except (OSError, FileNotFoundError):
        print(f"MISS {clip[:8]}"); continue
    pred = t["pred_xyz"]  # [1, sets, samples, 64, 3]
    pred = pred.reshape(-1, 64, 3)[:, :, :2]  # [6,64,2] xy
    g = gt[clip]
    gxy = np.array(g["gt_xy"] if isinstance(g, dict) and "gt_xy" in g else g, dtype=np.float64)[:, :2]
    assert gxy.shape == (64, 2), f"GT shape {gxy.shape}"
    d = np.linalg.norm(pred - gxy[None], axis=2)  # [6,64]
    ade64 = d.mean(axis=1)
    ade50 = d[:, :50].mean(axis=1)
    rows.append({"clip_id": clip,
                 "min_ade_5s": round(float(ade50.min()), 5),
                 "min_ade_6.4s": round(float(ade64.min()), 5),
                 "mean_ade_6.4s": round(float(ade64.mean()), 5)})
    print(f"{clip[:8]} minADE6@6.4s={ade64.min():.4f} @5s={ade50.min():.4f}", flush=True)

out = f"{E}/minade_chunk3116_a2e.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

a64 = np.array([r["min_ade_6.4s"] for r in rows])
a5 = np.array([r["min_ade_5s"] for r in rows])
n = len(a64)
sem = a64.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
print(f"\n=== Alpamayo2-Edge n={n} ===")
print(f"minADE6@6.4s mean={a64.mean():.5f} median={np.median(a64):.5f} sd={a64.std(ddof=1):.4f} 95%CI=[{a64.mean()-1.96*sem:.4f},{a64.mean()+1.96*sem:.4f}]")
print(f"minADE6@5s   mean={a5.mean():.5f}")
print(f"CSV: {out}")
