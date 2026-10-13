# 闸门 4：Alpamayo-1.5-10B 边缘可行性（2026-10-12，✅ 通过判据 / ❌ 20GB 预算）

**判据（next_plan.md 预注册）**："Alpamayo 权重能否在 20 GB 内加载并跑通一次前向；
在 goat（64 GB）上加载成功 + 单次前向。不过则 VLA 轨道判死，写成结论而非 future work。"

## 判定

| 问题 | 答案 | 依据 |
|---|---|---|
| goat（64GB）加载 + 单次前向 | ✅ **通过** | 加载 5.7s（mmap），前向 3505s（CPU），轨迹 (1,1,1,64,3) + CoT 输出 |
| "20 GB 内"加载 | ❌ **不可能** | FP16-only 权重 **22.16 GB**（5 分片字节校验）；官方单样本推理 **~24 GB VRAM**（README 硬件表，H100 实测）；无量化路径（TRTEdge 不支持 Alpamayo 架构，FINDINGS 已定案） |
| VLA 轨道（ADE/FDE 评测） | **受阻但非判死** | 三个具体受阻点见下 |

**三个受阻点（写成结论）**：
1. **20GB 边缘预算装不下**：FP16-only 22.16G 权重 > 20G；32GB 级 Orin（dog）连权重加载
   都超限（对照：FP16 CR2 引擎 15G 在 dog 上 deserialize 即 OOM 的先例）。
2. **CUDA PyTorch 无安装源**：goat 可达 pypi.org（CPU wheel）但 pypi.jetson.org 与
   developer.download.nvidia.com 均不可达（白名单出网）⇒ Orin GPU 用不上。
3. **CPU 前向 58.4 分钟/次**：单 clip 全管线（VLM rollout ≤256 token + diffusion expert）。
   100 clips 的 ADE/FDE 评测 ≈ 4 天纯 CPU ⇒ 大规模评测不可行；小样本（~10 clips，一夜）
   技术上可做，但受 4 制约（真实 ego-history 需 gated 数据集，goat 不可达 HF）。

## 实测记录（gate4_result.json / .npz，本目录）

- 加载：22.16 GB bf16，5.7s（safetensors mmap），RAM 峰值 42.5G（加载阶段）
- 前向：3505.4s，RAM 峰值 **46.0G / 61G**（其他用户同时占 ~33G——mmap 页可回收才挤得下）
- 输入 token：3,086（4 相机 × 4 帧 × 180 token/帧 @320×576 + 文本/特殊 token）
- 输出：pred_xyz (1,1,1,64,3) = **64 路点 @10Hz = 6.4s 视界**（与 PhysicalAI-AV/Alpamayo
  规格一致）；CoT = "Keep lane since the lane is clear ahead"
- **物理合理性**：合成 ego 历史 = 静止于原点 ⇒ 模型输出平缓起步（0.08→1.5 m/s 递增）、
  直线（横向偏移 ~1mm）、CoT 与轨迹一致——不是乱数，是功能性正确的前向。

## 输入与官方口径的差异（诚实边界）

| 项 | 官方 test_inference.py | 本运行 | 原因 |
|---|---|---|---|
| clip | 030c760c（gated 全集） | 01d3588e（公开 sample） | goat 不可达 HF、会话无 token |
| ego history | 数据集真值 | 合成（静止：xyz=0 + 单位旋转） | 同上 |
| GT future / minADE | 有 | **无**（不报 minADE） | 同上 |
| device | cuda（H100/3090…） | **cpu**（bf16 + SDPA） | CUDA torch 无安装源 |
| attn | flash_attention_2 | sdpa | flash-attn 需 nvcc 编 CUDA 版；README 明示 SDPA fallback |
| 相机 | 4 路 × 4 帧 @10Hz，t0=5.1s | **同**（帧从公开 mp4 按 n=144/147/150/153 精确抽取） | — |

⇒ 本运行证明的是**可加载性与前向可运行性**（闸门判据），不是精度；minADE 需真值，
留给有 token/网络的环境。

## 离线适配（4 处，全部记录在 gate4_driver.sh 头注释）

1. `config.json` 的 `vlm_name_or_path`：`nvidia/Cosmos-Reason2-8B` → 本地 processor 目录
   （CR2 tokenizer.json 词表已验证与 Qwen3-VL **完全一致**；preprocessor_config.json 与
   官方 helper 用的 Qwen3-VL-2B-Instruct **逐字节相同**）。
2. processor 目录补 `Qwen3-VL-8B-Instruct` 的 config.json/generation_config.json
   （`_initialize_qwenvl3_vlm` 要从该路径读 Qwen3VLConfig；架构与 CR2 实测一致：
   36 层 / hidden 4096 / vocab 151936）。
3. **上游设计地雷（PROBLEMS J3）**：`base_model.py:211` 默认 `vlm_name_or_path="Qwen/Qwen3-VL-8B-Instruct"`
   sed 为本地路径——transformers 的 `to_diff_dict` 会用**无参默认值**重构 config，而
   `ReasoningVLAConfig.__init__` 带网络副作用（`_build_processor`）⇒ 离线环境必须让默认值可解析。
4. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`：漏网远程请求快速失败。

另：staged CR2 tokenizer_config.json（TRTEdge 导出版）的 `extra_special_tokens` 是 list，
transformers 4.57 要求 dict ⇒ 已移除该字段（token 本就在 tokenizer.json 的 added_tokens 里）。

## 复现

```bash
# Mac：权重（公开，无 gate；Xet 协议会 401，必须禁用走传统 HTTP）
HF_HUB_DISABLE_XET=1 hf download nvidia/Alpamayo-1.5-10B --local-dir ~/Downloads/Alpamayo-1.5-10B
# 5 分片字节数须与 HF API tree 完全一致（4928204944/4915963032/4983071160/4980341192/2349614196）

# goat：环境（uv + py3.12 + torch 2.8.0 CPU；pypi.org/astral.sh/github 可达）
bash gate4_goat_env.sh          # 含 tarballs/ 40G 冗余清理（device_layout.md 预授权）
# staging：4 相机帧（公开 sample clip 01d3588e，ffmpeg n=144/147/150/153）+ processor 目录
#   + meta.npz（合成 ego）→ alpamayo_gate4/；权重 scp 后逐分片字节校验
bash gate4_driver.sh            # 离线适配 + gate4_forward.py（RAM 监控线程内置）
```

## 产物与设备布局

- 本目录：`gate4_result.json`（计时/RAM/形状/CoT）、`gate4_result.npz`（pred_xyz/pred_rot）
- goat：`alpamayo_gate4/`（权重 22G + staging + 结果）、`alpamayo1.5/`（clone + a1_5_venv）
- 脚本（git）：`scripts/orin/gate4_goat_env.sh`、`gate4_driver.sh`、`gate4_forward.py`
