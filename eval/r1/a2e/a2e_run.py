"""Batch Alpamayo2-Edge inference for the 18 chunk3116 clips (minADE6 arm).

Inputs are built in memory per clip (frames PNG + egomotion parquet + real nav),
mirroring load_physical_aiavdataset.py; nothing intermediate is written to disk.
Settings = official CLI defaults with num_traj_samples=6 (six independent
trajectory candidates, each with its own sampled reasoning = R1 6-seed analogue).
"""
import json, time
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path

W = "/home/nvidia/chenrunbin/a2e"
E = "/home/nvidia/chenrunbin/alpamayo-edge/eval_data"
FR = f"{W}/frames_a2e"
OUT = Path(f"{W}/eval_a2e/out")
T0 = 5_100_000

CAMS = [(0, "camera_cross_left_120fov"), (1, "camera_front_wide_120fov"),
        (2, "camera_cross_right_120fov"), (3, "camera_rear_left_70fov"),
        (5, "camera_rear_right_70fov"), (6, "camera_front_tele_30fov")]
PIXEL_BOUNDS = {c: ((163840, 196608) if c in (1, 6) else (81920, 98304)) for c, _ in CAMS}

manifest = json.load(open(f"{W}/frames_manifest.json"))
nav = json.load(open(f"{W}/nav_18clips.json"))
clips = [l.strip() for l in open(f"{E}/selected_clips.txt") if l.strip()]

def ego_history(clip):
    df = pd.read_parquet(f"{E}/chunk3116/{clip}.egomotion.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values.astype(np.float64)
    trel = (ts - ts[0]) / 1e6
    P = df[["x", "y", "z"]].values.astype(np.float64)
    Rall = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values)
    q = (T0 + np.arange(-1_900_000, 1, 100_000, dtype=np.int64)) / 1e6
    assert q[0] >= trel[0] and q[-1] <= trel[-1], f"ego range {trel[0]:.2f}..{trel[-1]:.2f}"
    xyz = np.stack([np.interp(q, trel, P[:, k]) for k in range(3)], axis=1)
    rot = Slerp(trel, Rall)(q)
    r0_inv = rot[-1].inv()
    local_xyz = r0_inv.apply(xyz - xyz[-1])
    local_rot = (r0_inv * rot).as_matrix()
    return local_xyz.astype(np.float32), local_rot.astype(np.float32)

def build_data(clip):
    m = manifest[clip]
    imgs, cam_ids, ts_list = [], [], []
    for cam_id, cam in CAMS:
        ent = m[cam]
        for j, f in enumerate(ent["files"]):
            a = np.array(Image.open(f"{FR}/{clip}/{f}").convert("RGB"))
            imgs.append(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0))
            cam_ids.append(cam_id)
            ts_list.append(ent["actual_ts"][j])
    image_frames = torch.stack(imgs)
    exyz, erot = ego_history(clip)
    return {
        "image_frames": image_frames,
        "camera_indices": torch.tensor(cam_ids, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(ts_list, dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(exyz).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(erot).unsqueeze(0),
        "nav_text": [nav[clip]["nav_text"]],
        "camera_pixel_bounds": dict(PIXEL_BOUNDS),
        "min_pixels": 163840, "max_pixels": 196608,
        "input_profile": "camera_subsampled", "history_num_steps": 20,
        "clip_id": clip, "t0_us": T0,
        "provenance": {"dataset": "nvidia/PhysicalAI-Autonomous-Vehicles", "clip_id": clip, "t0_us": T0},
    }

from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.inference import run_data
from alpamayo2_edge.sample import validate_sample

settings = dict(seed=43, top_p=0.98, top_k=None, temperature=0.6,
                num_traj_sets=1, num_traj_samples=6,
                max_generation_length=256, diffusion_kwargs={"inference_step": 10})

t0 = time.time()
model, info = load_model(f"{W}/Alpamayo2-Edge", device="cuda", dtype="bfloat16",
                         attn_implementation="sdpa", local_files_only=True)
print(f"MODEL_LOADED {time.time()-t0:.1f}s gpu={torch.cuda.get_device_name('cuda')}", flush=True)

n_ok = 0
for i, clip in enumerate(clips):
    t = time.time()
    try:
        data = build_data(clip)
        validate_sample(data)
        run_data(model, data, f"{clip}_{T0}", OUT / f"{clip}_{T0}", settings, "cuda")
        n_ok += 1
        print(f"[{i+1}/18] {clip[:8]} OK {time.time()-t:.1f}s", flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[{i+1}/18] {clip[:8]} FAIL {type(ex).__name__}: {ex}", flush=True)
print(f"RUN_DONE {n_ok}/{len(clips)} total={time.time()-t0:.1f}s")
