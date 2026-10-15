#!/usr/bin/env python3
"""闸门 4：Alpamayo-1.5-10B 在 orin-goat 上加载 + 单次前向（CPU bf16 SDPA）。

对官方 src/alpamayo1_5/test_inference.py 的适配（每处都是设备条件所迫，逐条记录）：
  1. 权重/processor 走本地 staging：goat 不可达 huggingface.co（白名单出网）。
  2. device=cpu：CUDA torch 无安装源（pypi.jetson.org 与 developer.download.nvidia.com
     均不可达）⇒ CPU 前向。闸门判据是"加载 + 单次前向"（内存可行性），不是速度。
  3. attn_implementation=sdpa：flash-attn 需 nvcc 编 CUDA 版；README 明示 SDPA fallback，
     模型类 _supports_sdpa=True。
  4. 输入 = 公开 sample clip 01d3588e（4 相机 × 4 帧 @10Hz，t0=5.1s，与官方口径一致）；
     官方测试 clip 030c760c 在 gated 全集（goat 不可达、会话无 token）⇒ 不用。
     ego history 为合成（静止于原点：xyz 全零 + 单位旋转）⇒ 无 GT future、不报 minADE。
  5. 不用 helper.get_processor（其 BASE_PROCESSOR_NAME 指向 HF）：等价地用本地 processor
     目录构造，再把 model.tokenizer（已加 traj/special tokens）换入——与官方实现同构。
"""
import json
import os
import threading
import time

import numpy as np
import torch
from PIL import Image

STAGE = "/home/nvidia/chenrunbin/alpamayo-edge/alpamayo_gate4"
WEIGHTS = os.path.join(STAGE, "Alpamayo-1.5-10B")
PROC = os.path.join(STAGE, "processor")

from transformers import AutoProcessor  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

# ---- RAM 监控（/proc/meminfo，5s 采样）----
peak = {"used_max_g": 0.0}
stop = threading.Event()


def monitor():
    while not stop.is_set():
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            d[k] = int(v.strip().split()[0]) / 1048576  # GB
        peak["used_max_g"] = max(peak["used_max_g"], d["MemTotal"] - d["MemAvailable"])
        time.sleep(5)


th = threading.Thread(target=monitor, daemon=True)
th.start()

# ---- 输入构造（与官方 load_physical_aiavdataset 输出同构）----
CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
        "camera_cross_right_120fov", "camera_front_tele_30fov"]  # 按 index [0,1,2,6] 排序
frames = []
for cam in CAMS:
    for j in range(1, 5):  # t0-0.3, t0-0.2, t0-0.1, t0
        im = np.array(Image.open(f"{STAGE}/frames/{cam}_{j}.png").convert("RGB"))
        frames.append(torch.from_numpy(im).permute(2, 0, 1))  # (3,H,W) uint8
image_frames = torch.stack(frames).view(4, 4, 3, *frames[0].shape[-2:])
meta = np.load(f"{STAGE}/meta.npz", allow_pickle=True)
camera_indices = torch.from_numpy(meta["camera_indices"])
ego_xyz = torch.from_numpy(meta["ego_history_xyz"])
ego_rot = torch.from_numpy(meta["ego_history_rot"])
print(f"INPUT_OK frames={tuple(image_frames.shape)} dtype={image_frames.dtype} "
      f"cam_idx={camera_indices.tolist()} ego={tuple(ego_xyz.shape)}/{tuple(ego_rot.shape)}",
      flush=True)

# ---- G4c: 加载 ----
t0 = time.time()
model = Alpamayo1_5.from_pretrained(
    WEIGHTS, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()
load_s = time.time() - t0
print(f"G4c_LOAD_OK {load_s:.1f}s  RAM_used_peak={peak['used_max_g']:.1f}G", flush=True)

processor = AutoProcessor.from_pretrained(
    PROC, min_pixels=helper.MIN_PIXELS, max_pixels=helper.MAX_PIXELS)
processor.tokenizer = model.tokenizer

messages = helper.create_message(
    frames=image_frames.flatten(0, 1), camera_indices=camera_indices)
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt")
model_inputs = {"tokenized_data": inputs, "ego_history_xyz": ego_xyz,
                "ego_history_rot": ego_rot}
ids = inputs.get("input_ids")
print(f"TOKENIZED_OK input_ids={None if ids is None else tuple(ids.shape)} "
      f"keys={sorted(inputs.keys())}", flush=True)

# ---- G4d: 单次前向 ----
torch.manual_seed(42)
t1 = time.time()
with torch.autocast("cpu", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True)
fwd_s = time.time() - t1
stop.set()
print(f"G4d_FORWARD_OK {fwd_s:.1f}s  RAM_used_peak={peak['used_max_g']:.1f}G", flush=True)
print(f"TRAJ pred_xyz={tuple(pred_xyz.shape)} pred_rot={tuple(pred_rot.shape)}", flush=True)
cot = str(extra["cot"][0])
print(f"COT ({len(cot)} chars): {cot[:800]}", flush=True)

np.savez(f"{STAGE}/gate4_result.npz",
         pred_xyz=pred_xyz.float().numpy(), pred_rot=pred_rot.float().numpy())
json.dump({"load_s": load_s, "forward_s": fwd_s,
           "ram_used_peak_g": peak["used_max_g"],
           "traj_shape": list(pred_xyz.shape),
           "input_ids_shape": None if ids is None else list(ids.shape),
           "cot_head": cot[:2000]},
          open(f"{STAGE}/gate4_result.json", "w"), indent=1, ensure_ascii=False)
print("GATE4_ALL_DONE", flush=True)
