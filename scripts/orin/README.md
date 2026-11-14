# 设备端脚本（Orin 上实际跑的那几个）

从两台 Orin 归档到仓库的原始脚本。设备上的副本在各自的
`chenrunbin/alpamayo-edge/scripts/`（设备目录名是历史遗留，布局见
[docs/device_layout.md](../../docs/device_layout.md)）。

本目录按阶段分两节：**Phase R（Alpamayo-R1，含 1.5 跨代）** → **其余（gate4 / TRTEdge 构建）**。
CR2-8B 量化部署与 Phase W（Cosmos3-Edge）的设备脚本已迁至
[cosmos-edge 仓 scripts/orin/](https://github.com/ranpin/cosmos-edge/tree/main/scripts/orin)。

## Phase R：Alpamayo-R1 AV 轨迹部署（2026-10-26~）

TRTEdge 0.10.1 **原生支持 `alpamayo_r1`**（推翻闸门 4 基于 0.9.0 的「不支持」旧判），
官方指南 `docs/source/user_guide/examples/vla/alpamayo.md`，action 输出 = `output_trajectory`
的 `(accel, kappa)` 序列，官方 minADE 基线 0.82113 m / 644 clips。action 专家撞 **K6 同源墙**
（`trt::attention_onnx/rope_onnx/kv_cache_update_onnx` → ONNX 标准 Attention/RotaryEmbedding/
TensorScatter，需 TRT≥11）。最小验证 = **只导 + 只建 action 单引擎**（~2.3B FP16），不先碰 10B llm build。

| 脚本 | 设备 | 作用 | 产出/结论 |
|---|---|---|---|
| `apply_alpamayo_action_decomposed_patch.py` | — | **R1 墙解法**：action 专家加三个分解实现，`USE_DECOMPOSED_ALPAMAYO=1` 门控（独立于 cosmos3 的 `USE_DECOMPOSED_POLICY`） | 绕开三个 TRT-11-only 算子 |
| `test_alpamayo_action_decomposed.py` | CPU | 导出前数值对拍，**16 项全过** | kv_cache err=0 / attention err~1e-7 / rope err=0 |
| `r1_action_minverify.sh` | goat | **最小验证全链**：下载→补丁→对拍→只导 action（`--skip-llm --skip-visual`）→算子普查→只建 action 引擎 | **✅ 已跑通**：export 峰值 10.5G、build 89.9s 峰值 CPU 12.1G、`action.engine` 4.25G、墙算子全 0 |
| `r1_full_chain_infer.sh` | goat | **全链推理 + seed 对照**：llm/visual 导出与构建（含磁盘/内存账与 `HF_ENDPOINT` 坑）→ 输入构造（16 帧 camera-major + 官方指南真实 ego 历史）→ 七配置对照（确定性底噪 / seed 轴 / ego 历史轴，`temperature=0`） | **✅ 已跑通**：三引擎建成（llm 14.15G / visual 1.08G / action 4.25G）、推理 ~6s、输出文本 + 64 waypoint 轨迹；SEED/EGO-HIST=3.84×（证据 `eval/r1/`） |

与 cosmos3 policy 补丁（cosmos-edge `scripts/orin/apply_policy_decomposed_patch.py`）的**三点不同**，
均从源码实测钉死、且都只在 **B>1** 才暴露（导出 dummy 恰好 `B=1`，直接复用会在建 `maxBatchSize=6` 引擎时才炸）：

1. **RoPE 的 cos/sin 只有 S 行、跨 batch 共享**（forward 里 `rope_cos[0:1].reshape(-1,half)`）
   ⇒ 不能复用 cosmos3 的 `cos.reshape(b,s,half)`（B>1 元素不足），改为按 `position_ids` 行索引后广播。
2. **attn_mask 是 4D `[B,1,S,Sk]`，而 GQA 折叠后 scores 是 5D `[B,Hkv,rep,S,Sk]`**
   ⇒ 不升维直接相加只在 B==1 碰巧广播正确；对拍里的护栏实证 **B=2 时 naive 写法直接抛 RuntimeError**。
3. **`kv_cache_update` 的 `cache_indices` 是运行期张量输入**（导出 dummy=3027、dim0 动态）
   ⇒ trace 期值未知，`torch.cat` 静态切片不可用，改用 `scatter` 动态写入（索引仅 `[B,H,S,D]`，
   比 gather+mask 的 `[B,H,C,D]` 省 ~C/S 倍）。

### 跨代：Alpamayo-1.5 repack + 导航引导检验（2026-10-30）

1.5 经 repack 跑上 Orin、与 R1 同口径 minADE 对比、导航引导假设否定，结论见
[`../../eval/r1/crossgen_a15.md`](../../eval/r1/crossgen_a15.md)。

| 脚本 | 设备 | 作用 | 产出/结论 |
|---|---|---|---|
| `repack_alpamayo15_to_r1.py` | goat | 把 1.5 checkpoint 重打包为 `alpamayo_r1` 格式（纯 config 改写、不动权重、幂等、留 .orig） | 1.5 走 R1 全链；docstring 含 🔴 tokenizer 陷阱（须拷 R1 引擎 tokenizer 三件套，否则 EDGELLM_BAD_MEDIA_COUNT） |
| `derive_a15_oracle_nav.py` | goat | 从 GT 未来轨迹反推 oracle nav（按航向变化分 RIGHT/LEFT/STRAIGHT），建 `io/r1_eval_in18_nav.json` | 本批 6 RIGHT + 12 STRAIGHT + 0 LEFT |
| `a15_nav_eval.sh` | goat | 1.5 6-seed 推理 + merge + compute_minade（nav/无nav 共用，入参区分） | oracle nav 仅 −1% within-noise ⇒ 导航假设否定 |

## 其余脚本（按用途索引）

| 用途 | 脚本 |
|---|---|
| 闸门 4（VLA 可行性） | `gate4_goat_env.sh` / `gate4_driver.sh` / `gate4_forward.py`（Alpamayo-1.5-10B 加载 + CPU bf16 单次前向） |
| TRTEdge 0.10.1 自建 | `w0b_build_trtedge_0101.sh`（与 cosmos-edge 仓共享脚本，两仓并存） |
