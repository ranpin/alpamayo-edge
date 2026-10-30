#!/usr/bin/env python3
"""把 Alpamayo-1.5-10B checkpoint 重打包为 alpamayo_r1 格式（纯 config.json 改写，不动权重）。

依据（2026-10-30 静态侦察）：
- 1.5 与 R1 同架构：1159 张量同名同形、分片落位全同；R1 仅多 7 个派生 buffer
  （3×freqs=logspace(0,log10(100),10) 由 modeling init 合成并 register_buffer、
  4×action_space.*=fp32(config 常数)，两代 config 的该组常数逐位相同）。
- TRTEdge 0.10.1 加载器非 strict（缺失键 debug-skip），故 v1 不注入 buffer；
  若 export 报缺再升级 v2 注入（值直接取 R1 shard5，已证两代同值）。
- model_type 门只在 export.py 两个 frozenset + chat_template/checkpoint_utils；
  改成 alpamayo_r1 后 chat template/prompt 协议自动=R1 的——受控对比恰需要
  （唯一变量=权重/RL 后训练）。
- vlm_name_or_path=nvidia/Cosmos-Reason2-8B 保留（1.5 真实 backbone，加载器
  凭它取 VLM text config，设备侧需 HF_ENDPOINT=hf-mirror）；include_camera_ids
  等 3 个 1.5-only 键保留（TRTEdge 不消费，最小 diff 原则）。
- 🔴 tokenizer 陷阱（2026-10-30 实测）：若 vlm_name_or_path 指向本地 processor 目录
  （如 gate4 的 alpamayo_gate4/processor），导出进引擎目录的 chat template 不认识
  {"type":"image"} 内容（tokenizer_config 无 image_pad/vision_start/vision_end），
  推理时全部请求报 EDGELLM_BAD_MEDIA_COUNT（"pad count is smaller than this
  request's media count"）。修复 = 引擎建成后把 R1 引擎 llm/ 的 tokenizer.json /
  tokenizer_config.json / processed_chat_template.json 三件拷入 1.5 引擎 llm/ 覆盖
  （两代同词表 151936、同架构，R1 的即 Qwen3-VL-8B-Instruct tokenizer）；这也让
  跨代对比更受控——两代用完全相同的 tokenizer/chat template，唯一变量=权重。
  或：导出前把 vlm_name_or_path 改为 Qwen/Qwen3-VL-8B-Instruct 再导出（走 HF 抓取）。

用法（在有权重的设备上）：python3 repack_alpamayo15_to_r1.py <1.5_ckpt_dir>
产出 config.json.orig 备份；幂等（已是 r1 格式则跳过）。
"""
import json
import os
import sys

d = sys.argv[1]
p = os.path.join(d, "config.json")
c = json.load(open(p))
mt = c.get("model_type")
if mt == "alpamayo_r1":
    print("already r1 format, skip")
    sys.exit(0)
assert mt == "alpamayo1_5", f"model_type={mt!r} 既非 alpamayo1_5 也非 alpamayo_r1，拒改"
if not os.path.exists(p + ".orig"):
    os.rename(p, p + ".orig")

c["model_type"] = "alpamayo_r1"
c["architectures"] = ["AlpamayoR1"]


def fix_targets(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "_target_" and isinstance(v, str):
                o[k] = v.replace("alpamayo1_5.", "alpamayo_r1.")
            else:
                fix_targets(v)
    elif isinstance(o, list):
        for x in o:
            fix_targets(x)


fix_targets(c)
c.setdefault("expert_cfg", {})["dtype"] = "bfloat16"  # 对齐 R1 schema
if "diffusion_cfg" in c:
    c["diffusion_cfg"]["x_dims"] = "???"  # 对齐 R1（hydra MISSING 占位，TRTEdge 不消费）
json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
print(f"repack done: {p}")
print(f"  model_type={c['model_type']} architectures={c['architectures']}")
print(f"  vlm_name_or_path={c.get('vlm_name_or_path')} (保留)")
