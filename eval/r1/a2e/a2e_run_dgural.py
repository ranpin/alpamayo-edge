"""Alpamayo2-Edge inference for the dgural demo anchors (6 static + 8 rolling t0>=2s).

Builds inputs in memory (frames + ego parquet + real nav), runs the official
run_data with 6 candidates, selects best-of-6 against GT (minADE selection,
same convention as the reported metric), and writes prediction JSONs in the
same schema as r1_multi_pred.json / roll_pred.json for the render scripts.
"""
import json, time
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path
from safetensors.numpy import load_file

W = "/home/nvidia/chenrunbin/a2e"
DG = f"{W}/dgural"
E = "/home/nvidia/chenrunbin/alpamayo-edge/eval_data"
OUT = Path(f"{W}/eval_a2e/dgural_out")

CAMS = [(0, "camera_cross_left_120fov"), (1, "camera_front_wide_120fov"),
        (2, "camera_cross_right_120fov"), (3, "camera_rear_left_70fov"),
        (5, "camera_rear_right_70fov"), (6, "camera_front_tele_30fov")]
PIXEL_BOUNDS = {c: ((163840, 196608) if c in (1, 6) else (81920, 98304)) for c, _ in CAMS}

manifest = json.load(open(f"{DG}/manifest_a2e_dgural.json"))
nav = json.load(open(f"{DG}/nav_dgural.json"))
mcd = json.load(open(f"{E}/dgural_multi/multi_clip_data.json"))      # static GT
rcd = json.load(open(f"{E}/roll/roll_clip_data.json"))               # rolling GT

def nav_at(cid8, t0_us):
    rows = nav[cid8]["rows"]
    le = [r for r in rows if r[0] <= t0_us]
    return (le[-1] if le else rows[0])[1]

def ego_history(cid, t0_us):
    df = pd.read_parquet(f"{DG}/ego/{cid}.egomotion.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values.astype(np.float64)
    trel = (ts - ts[0]) / 1e6
    P = df[["x", "y", "z"]].values.astype(np.float64)
    Rall = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values)
    q = (t0_us + np.arange(-1_900_000, 1, 100_000, dtype=np.int64)) / 1e6
    assert q[0] >= trel[0] and q[-1] <= trel[-1], f"ego range {trel[0]:.2f}..{trel[-1]:.2f} vs t0 {t0_us/1e6}"
    xyz = np.stack([np.interp(q, trel, P[:, k]) for k in range(3)], axis=1)
    rot = Slerp(trel, Rall)(q)
    r0_inv = rot[-1].inv()
    v0 = float(np.linalg.norm((xyz[-1] - xyz[-2]) / 0.1))
    return (r0_inv.apply(xyz - xyz[-1])).astype(np.float32), (r0_inv * rot).as_matrix().astype(np.float32), v0

def build_data(tag):
    m = manifest[tag]
    imgs, cam_ids, ts_list = [], [], []
    for cam_id, cam in CAMS:
        ent = m["cams"][cam]
        for j, f in enumerate(ent["files"]):
            a = np.array(Image.open(f"{DG}/frames/{tag}/{f}").convert("RGB"))
            imgs.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0))
            cam_ids.append(cam_id)
            ts_list.append(ent["actual_ts"][j])
    exyz, erot, v0 = ego_history(m["cid"], m["t0"])
    data = {
        "image_frames": torch.stack(imgs),
        "camera_indices": torch.tensor(cam_ids, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(ts_list, dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(exyz).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(erot).unsqueeze(0),
        "nav_text": [nav_at(m["cid8"], m["t0"])],
        "camera_pixel_bounds": dict(PIXEL_BOUNDS),
        "min_pixels": 163840, "max_pixels": 196608,
        "input_profile": "camera_subsampled", "history_num_steps": 20,
        "clip_id": m["cid"], "t0_us": m["t0"],
        "provenance": {"dataset": "nvidia/PhysicalAI-Autonomous-Vehicles", "clip_id": m["cid"], "t0_us": m["t0"]},
    }
    return data, v0

def select_best(tag, t0_us, gt_xyz, v0):
    d = OUT / f"{tag}_{t0_us}"
    pred = load_file(str(d / "output.safetensors"))["pred_xyz"].reshape(-1, 64, 3)[:, :, :2]
    rec = json.load(open(d / "output.json"))
    cots = rec.get("text", {}).get("cot") or []
    gt = np.array(gt_xyz)[:, :2]
    ades = np.linalg.norm(pred - gt[None], axis=2).mean(axis=1)
    b = int(np.argmin(ades))
    xy = pred[b]
    return {
        "pred_xy": xy.tolist(),
        "ade3": round(float(np.linalg.norm(xy[:30] - gt[:30], axis=1).mean()), 5),
        "ade6": round(float(ades[b]), 5),
        "fde": round(float(np.linalg.norm(xy[-1] - gt[-1])), 5),
        "cot": (cots[b] if isinstance(cots, list) and len(cots) > b else (cots if isinstance(cots, str) else "")),
        "v0": round(v0, 3),
        "all6_ade6": [round(float(a), 5) for a in ades],
    }

from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.inference import run_data
from alpamayo2_edge.sample import validate_sample

settings = dict(seed=43, top_p=0.98, top_k=None, temperature=0.6,
                num_traj_sets=1, num_traj_samples=6,
                max_generation_length=256, diffusion_kwargs={"inference_step": 10})

t_start = time.time()
model, info = load_model(f"{W}/Alpamayo2-Edge", device="cuda", dtype="bfloat16",
                         attn_implementation="sdpa", local_files_only=True)
print(f"MODEL_LOADED {time.time()-t_start:.1f}s", flush=True)

static_out, roll_out = {}, {}
tags = list(manifest.keys())
for i, tag in enumerate(tags):
    m = manifest[tag]
    t = time.time()
    try:
        data, v0 = build_data(tag)
        validate_sample(data)
        run_data(model, data, f"{tag}_{m['t0']}", OUT / f"{tag}_{m['t0']}", settings, "cuda")
        gt = (mcd[m["cid8"]] if tag in mcd else rcd[tag])["gt_xyz"]
        sel = select_best(tag, m["t0"], gt, v0)
        if tag in mcd:
            static_out[tag] = sel
        else:
            roll_out[tag] = sel
        print(f"[{i+1}/{len(tags)}] {tag} OK ade6={sel['ade6']:.3f} ({time.time()-t:.1f}s)", flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[{i+1}/{len(tags)}] {tag} FAIL {type(ex).__name__}: {ex}", flush=True)

json.dump(static_out, open(f"{DG}/r1_multi_pred_a2e.json", "w"))
json.dump(roll_out, open(f"{DG}/roll_pred_a2e.json", "w"))
print(f"DGURAL_DONE static={len(static_out)}/6 roll={len(roll_out)}/8 total={time.time()-t_start:.1f}s")
