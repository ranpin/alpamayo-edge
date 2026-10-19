# 设备端脚本（Orin 上实际跑的那几个）

从两台 Orin 归档到仓库的原始脚本。设备上的副本在各自的
`chenrunbin/alpamayo-edge/scripts/`（见 [`../../docs/device_layout.md`](../../docs/device_layout.md)）。

本目录共 30 个脚本，按阶段分三节：**第一阶段（CR2-8B 量化部署）** → **Phase W（Cosmos3-Edge）**
→ **其余（按用途索引）**。

## 第一阶段：CR2-8B 量化部署

| 脚本 | 设备 | 作用 | 产出 |
|---|---|---|---|
| `measure_power.sh` | orin-dog (32GB) | bench 期间并行采 `tegrastats`，算**三轨之和**（VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0）的 GPU-active 均值与能效 | `logs/pw_*.log` → §1.4 功耗/能效 |
| `dog_fp16_validate.sh` | orin-dog | 把 goat 建好的 FP16 引擎（16.4 GB）拉回 32GB 机，试"**build 不下、但拿现成引擎能不能跑**" | `dog_fp16_validate.out`：传输完整、引擎 15.15 GB，但**推理未跑起来**（见下） |
| `build_bench.sh` | orin-goat (64GB) | 解包 ONNX → 建 INT4/INT8 引擎（与 FP16 同参数）→ 三精度 prefill/decode bench | `logs/build_int{4,8}.log` → §1.1 加速比 |
| `q_power.sh` | orin-goat | INT4/INT8 功耗（同 tegrastats 口径） | `logs/q_power.out` |
| `fp16_power.sh` | orin-goat | FP16 功耗，量化能效比的分母 | — |

## 注意
- **`dog_fp16_validate.out` 里 `FP16_PREFILL:` / `FP16_DECODE:` 都是空的，`PEAK_INFER_MEM_MB=1468`**
  ——1468 MB 只是这台机的空载内存，说明 `llm_bench` **根本没跑到分配 15 GB 那一步就退了**：
  FP16 引擎在 32GB 的 dog 上**反序列化即 OOM**（要 15 GB，而 16.4 GB 的 tar 文件自身页缓存已把 30 GB 吃满，
  无 sudo 密码无法 drop cache）。**这是一次失败的验证，不是成功的**。
  结论因此更硬：FP16 在 32GB 机上 **build（峰值 54.8 GB）与 load（15 GB + 页缓存 ≈ 30 GB）双双不可行**
  ⇒ 量化不只是提速，是**能否部署**的前提。详见 [`../../docs/orin_goat_migration_plan.md`](../../docs/orin_goat_migration_plan.md) Phase 4.5。
- 已发布的 FP16 基线（TTFT 298 ms / TPOT 89.3 ms）全部来自 **orin-goat（64GB）**，不是这台机。
- 脚本里的绝对路径已随 2026-07-27 的设备整理更新（原先写的是 `/home/vision/...` 顶层与
  `/home/nvidia/chenrunbin.crb/...`）；脚本头有注释标明改动。
- `measure_power.sh` 原先把 `pw_*.log` 写到**共享账号的家目录顶层**，这也是设备上文件散落的原因之一，
  现已改为写进项目自己的 `logs/`。

## Phase W：Cosmos3-Edge 部署链（2026-10-15~19）

六道 JetPack 6.2 兼容墙与打通见 [`../../docs/PROBLEMS.md`](../../docs/PROBLEMS.md) §K/§L。

| 脚本 | 设备 | 作用 | 产出/结论 |
|---|---|---|---|
| `w0b_build_trtedge_0101.sh` | dog | 自建 TRTEdge v0.10.1（共享 0.9.0 安装不动），`-DBUILD_EXPERIMENTAL_MODELS=ON` | K1（CuTe DSL 缺 sm_87+cuda12）暴露处 |
| `w0c_goat_export_env.sh` | goat | python 导出环境（torch 2.13.0 / transformers 5.14.1 / onnx 1.19.0 硬 pin），与 gate4 venv 隔离 | export 是 CPU 任务 |
| `w1_goat_engine_build.sh` | goat | TRTEdge 0.10.1 C++ 构建 + reasoning 引擎构建（dog GPU 被 system_agent 占 26.5G ⇒ 改 goat）；需先打 vector_types.h 补丁（K3） | reasoning 引擎（分解注意力版） |
| `w1_cosmos3_engine_build.sh` | dog | reasoning 引擎构建（dog 版，与 goat 同源，maxInputLen 2048 / KV 4096） | 插件路径须指 0.10.1 自建 build |
| `apply_decomposed_attn_patch.py` | — | **K4 解法**：visual 导出加分解注意力路径（基础 ONNX 算子块对角 ragged self-attn） | 绕开 ViTAttentionPlugin 无 legacy 后端 + TRT_Attention 需 TRT≥11 |
| `apply_policy_decomposed_patch.py` | — | **K6 解法**：policy 的 und_prefill/gen 加 `decomposed_attention` + `decomposed_rope`，`USE_DECOMPOSED_POLICY=1` 门控 | 绕开 ONNX 标准 Attention/RotaryEmbedding 需 TRT≥11 |
| `test_policy_decomposed.py` | CPU | 导出前数值对拍（attention vs SDPA、rope vs LLaMA 全维公式），6 项全过 | err~1e-7 / err=0，避免 ~1h 导出后才发现语义 bug |
| `w2_policy_build.sh` | goat | policy 四组件分组件构建（und_prefill → gen → 重导出补 vae → vae） | vae_encoder 首次被杀（exit=120，内存压力） |
| `w2_policy_rebuild_decomposed.sh` | goat | 分解版重导 und_prefill+gen 再建引擎，分段执行保磁盘余量（起始 free 6.3G） | 四组件引擎建成 |
| `w2_policy_smoke.sh` | goat | `cosmos3_policy_inference` 冒烟（官方 UMI 首帧 + "pick up the object"，4 步扩散） | exit=0、action [1,16,10] 全 finite、5154ms/round；**只验管线不验语义** |

> W3 帧无关复测脚本在 [`../../eval/cosmos3/w3_frame_sensitivity.py`](../../eval/cosmos3/w3_frame_sensitivity.py)
> （归评测资产，不是设备构建脚本）。

## 其余脚本（按用途索引）

| 用途 | 脚本 |
|---|---|
| len4096/16384 引擎构建 | `build_len4096.sh` / `build_len16384.sh`（Pareto 用，老引擎不动保可比性） |
| 15% 归因对照（单变量） | `build_controls.sh` / `sweep_len4096.sh` / `rerun_untruncated.sh` |
| 分辨率 Pareto | `run_pareto.sh` / `pareto_int4_driver.sh` / `pareto_int8_driver.sh` / `pareto_int8_1080_retry.sh`（KV 预留 32768→16448 省 2.41G） |
| 可靠性（31min 零降频） | `sustained_load.sh`（30min 持续 decode + tegrastats 采温频功耗） |
| 闸门 4（VLA） | `gate4_goat_env.sh` / `gate4_driver.sh` / `gate4_forward.py`（Alpamayo-1.5-10B 加载 + CPU bf16 单次前向） |
