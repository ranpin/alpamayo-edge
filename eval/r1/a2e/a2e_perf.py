"""Alpamayo2-Edge Orin 性能测量（表 3 2-Edge 列补齐）：
三段拆解（prefill / CoT decode / diffusion expert）+ 内存 + 候选边际成本。
方法对齐 R1 perf campaign：warmup 后计时、torch.cuda.synchronize 包裹、
系统 available RAM delta 记峰值（Orin 统一内存）、功耗由 tegrastats 外部采样。
"""
import json, time, threading, subprocess
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

W = "/home/nvidia/chenrunbin/a2e"
DG = f"{W}/dgural"
CLIP = "ecafce84"  # 静态 demo clip（t0=9.0s），与 demo 页一致
N_TIMED = 3

manifest = json.load(open(f"{DG}/manifest_a2e_dgural.json"))
nav = json.load(open(f"{DG}/nav_dgural.json"))

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
    xyz = np.stack([np.interp(q, trel, P[:, k]) for k in range(3)], axis=1)
    rot = Slerp(trel, Rall)(q)
    r0_inv = rot[-1].inv()
    return (r0_inv.apply(xyz - xyz[-1])).astype(np.float32), (r0_inv * rot).as_matrix().astype(np.float32)

CAMS = [(0, "camera_cross_left_120fov"), (1, "camera_front_wide_120fov"),
        (2, "camera_cross_right_120fov"), (3, "camera_rear_left_70fov"),
        (5, "camera_rear_right_70fov"), (6, "camera_front_tele_30fov")]

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
    exyz, erot = ego_history(m["cid"], m["t0"])
    return {
        "image_frames": torch.stack(imgs),
        "camera_indices": torch.tensor(cam_ids, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(ts_list, dtype=torch.int64),
        "ego_history_xyz": torch.from_numpy(exyz).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(erot).unsqueeze(0),
        "nav_text": [nav_at(m["cid8"], m["t0"])],
        "camera_pixel_bounds": {c: ((163840, 196608) if c in (1, 6) else (81920, 98304)) for c, _ in CAMS},
        "min_pixels": 163840, "max_pixels": 196608,
        "input_profile": "camera_subsampled", "history_num_steps": 20,
        "clip_id": m["cid"], "t0_us": m["t0"],
        "provenance": {"dataset": "nvidia/PhysicalAI-Autonomous-Vehicles", "clip_id": m["cid"], "t0_us": m["t0"]},
    }

def avail_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024

def sync():
    torch.cuda.synchronize()

results = {"clip": CLIP, "avail_base_mb": avail_mb()}
print(f"avail_base={results['avail_base_mb']}MB", flush=True)

from alpamayo2_edge.checkpoint import load_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device
from alpamayo2_edge.inference import predict

t0 = time.perf_counter()
model, info = load_model(f"{W}/Alpamayo2-Edge", device="cuda", dtype="bfloat16",
                         attn_implementation="sdpa", local_files_only=True)
sync()
results["load_s"] = round(time.perf_counter() - t0, 2)
results["avail_loaded_mb"] = avail_mb()
results["model_footprint_mb"] = results["avail_base_mb"] - results["avail_loaded_mb"]
print(f"load={results['load_s']}s footprint={results['model_footprint_mb']}MB", flush=True)

# ---- 埋点 ----
class Timer:
    def __init__(self):
        self.t = 0.0
        self.n = 0
    def reset(self):
        self.t = 0.0
        self.n = 0

t_prefill, t_decode, t_vlm, t_expert = Timer(), Timer(), Timer(), Timer()
decode_steps = {"n": 0}
first_fwd = {"done": False}
orig_model_fwd = model.vlm.model.forward

def timed_model_fwd(*a, **k):
    sync(); s = time.perf_counter()
    out = orig_model_fwd(*a, **k)
    sync(); dt = time.perf_counter() - s
    if not first_fwd["done"]:
        t_prefill.t += dt; first_fwd["done"] = True
    else:
        t_decode.t += dt; decode_steps["n"] += 1
    return out

model.vlm.model.forward = timed_model_fwd

def wrap(timer, fn):
    def inner(*a, **k):
        sync(); s = time.perf_counter()
        out = fn(*a, **k)
        sync(); timer.t += time.perf_counter() - s; timer.n += 1
        return out
    return inner

model._generate_with_shared_prefill = wrap(t_vlm, model._generate_with_shared_prefill)
model._sample_actions = wrap(t_expert, model._sample_actions)

data = build_data(CLIP)
settings = dict(seed=43, top_p=0.98, top_k=None, temperature=0.6,
                num_traj_sets=1, num_traj_samples=6,
                max_generation_length=256, diffusion_kwargs={"inference_step": 10})
prepared = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device="cuda")

# 内存轮询
peak = {"min_avail": results["avail_loaded_mb"], "stop": False}
def poll():
    while not peak["stop"]:
        peak["min_avail"] = min(peak["min_avail"], avail_mb())
        time.sleep(0.3)
th = threading.Thread(target=poll, daemon=True)
th.start()

# warmup（编译/图缓存预热，不计）
first_fwd["done"] = False
predict(model, prepared, settings, "cuda")
print(f"warmup done, decode_steps={decode_steps['n']}", flush=True)

runs = []
for i in range(N_TIMED):
    for t in (t_prefill, t_decode, t_vlm, t_expert):
        t.reset()
    decode_steps["n"] = 0
    first_fwd["done"] = False
    s = time.perf_counter()
    predict(model, prepared, settings, "cuda")
    sync()
    e2e = time.perf_counter() - s
    runs.append(dict(e2e=round(e2e, 2), prefill=round(t_prefill.t, 2),
                     cot_decode=round(t_decode.t, 2), decode_steps=decode_steps["n"],
                     expert=round(t_expert.t, 2), vlm_total=round(t_vlm.t, 2)))
    print(f"run{i+1}: {runs[-1]}", flush=True)

# 单候选边际成本
settings1 = dict(settings, num_traj_samples=1)
for t in (t_prefill, t_decode, t_vlm, t_expert):
    t.reset()
decode_steps["n"] = 0
first_fwd["done"] = False
s = time.perf_counter()
predict(model, prepared, settings1, "cuda")
sync()
results["single_candidate"] = dict(e2e=round(time.perf_counter() - s, 2),
                                   prefill=round(t_prefill.t, 2),
                                   cot_decode=round(t_decode.t, 2),
                                   decode_steps=decode_steps["n"],
                                   expert=round(t_expert.t, 2))
print(f"single: {results['single_candidate']}", flush=True)

peak["stop"] = True
results["peak_runtime_delta_mb"] = results["avail_loaded_mb"] - peak["min_avail"]
results["runs"] = runs
results["torch_max_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
print("RESULTS_JSON_BEGIN")
print(json.dumps(results, indent=1))
print("RESULTS_JSON_END")
