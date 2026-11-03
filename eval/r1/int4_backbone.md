# R1 backbone INT4 AWQ 量化：干净实验（精度近乎免费 + decode 2.52×）

> **状态（2026-11-03）**：把 **R1 自身 AV 训练 backbone**（不是拿 CR2 量化引擎顶替）走
> INT4 AWQ 量化→Orin 重建→全链评测，**三闸门全过 + AV 域校准对照臂**：
> 1. **G1** Kaggle 免费云量化（5 版内核鏖战驯服 py3.13 环境）→ INT4 ONNX（5.8GB tgz）；
> 2. **G2** Orin 构建+smoke（四连环修复）→ **CoT 文本与 FP16 逐字相同**；
> 3. **G3** 同口径 minADE6：**INT4 0.800 vs FP16 0.804 m（Δ=-0.004m，判据"近乎免费"）**；
> 4. **AV 域校准对照臂**（预注册后续变量，§五）：换 dgural 公开 AV 帧重量化，**minADE6 0.825 vs
>    FP16 0.804 m（Δ=+0.021m，仍"近乎免费"）**——校准域不改变结论（差异全在噪声内）。
>
> 速度：**decode 2.52×**（39.8 vs 100.3 ms）、prefill 0.96×、引擎 **3.11× 小**（4.88 vs 15.19 GB）。
> **含义**：CR2 推理轨道的「量化几乎免费」结论延伸到 **VLA 轨迹轨道**——「激进量化+保留大模型」
> 部署策略对 Alpamayo-R1 同样成立。基线与口径见 [`README.md`](README.md) §五；
> 跨代对比见 [`crossgen_a15.md`](crossgen_a15.md)。
> ⚠️ 帧与 GT 同 README §五：gated + license:other 数据集，只在机上跑，逐 clip 原始数据不入库。

## 一、动机与实验设计

**为什么必须量化 R1 自身 backbone**（2026-11-02 侦察定案）：

- 直接拿现有 CR2-8B 量化引擎换入 R1 管线**不可比**——R1 backbone 是 AV 训练版（权重数值不同），
  换入 = action 专家面对没见过的 KV 分布，测出来是「换模型崩塌」而非「量化损失」；
- FP4/NVFP4 被 **Orin sm_87 硬件墙**排除（NVFP4 kernel 目标 SM100+/Blackwell；fp8 需 Hopper+）
  ⇒ Orin 的 4-bit 唯一路 = **INT4 AWQ**；
- R1 backbone 与 CR2-8B 同族（Qwen3-VL-8B 系）⇒ 可 repack 成 qwen3_vl 格式走已验证的量化管线。

**预注册判据**（开工前定死）：ΔminADE6 < 0.1m 近乎免费 / 0.1–0.5m 量化代价 / > 0.5m 不可用；
INT8 预注册为精度损失大时的后续分支（**未触发**）。

**三个待实证点，全部得到回答**：

| 待实证点 | 答案 |
|---|---|
| (i) 量化工具认不认 R1 repack | ✅ qwen3_vl 族路径全通（AWQ 多模态校准 + ONNX 导出） |
| (ii) action_inference 能否加载 INT4 llm 引擎 | ✅ KV 接口兼容（kv_cache_dtype 仍 fp16，action 专家正常消费量化 backbone 的 KV） |
| (iii) action 专家对量化 KV 的容忍度 | ✅ 高容忍：轨迹扰动 0.048（归一化单位）< ego-history 轴 0.071 << seed 轴 0.271 |

## 二、G1：Kaggle 云量化（5 版内核鏖战）

**repack 配方**（全程免 HF token，R1 gated=False）：R1 的 750 个 `vlm.*` 张量剥前缀
（名字集合==CR2-8B，形状除 vocab 155697 vs 151936 外全同）+ CR2 config 模板只改 vocab_size
+ Qwen3-VL-8B-Instruct 公开 tokenizer ⇒ qwen3_vl 格式 checkpoint（17.6GB）。
TRTEdge 钉 **v0.10.1**（==goat 运行时）；export 带 `--max-kv-cache-capacity 4096`（与部署引擎一致）。

**失败链与修复**（每版都死得更深，全部环境层）：

| 版本 | 死于 | 根因 | 修复 |
|---|---|---|---|
| v1 | 16s | Kaggle 镜像升 py3.13，scikit-build-core metadata 崩；`pip -q` 吞报错 | 不装包本体：直装 tools-extra 依赖 pin + PYTHONPATH 源码树 + entry point 经 `python -c` |
| v2 | 6.6min | hub 1.x `hf download --include "A" "B"` 把第二个 pattern 误解析成文件名 | 两处下载改 `snapshot_download` Python API |
| v3 | 10.9min | 预装 torchaudio（cu128）与装的 torch 2.13.0+cu130 冲突，transformers audio_utils 模块级硬导入 | `pip uninstall -y torchaudio` + ENV_OK 闸门导入双 entry 链 |
| v4 | 12.8min | **AWQ 多模态校准 OOM**：torch 2.13 SDPA 在 T4（sm_75）只剩 math backend，全分辨率 MMMU 图像展开数千 visual token → 物化 S² 注意力 3.46GiB | 补丁 D：校准图像封顶 ~40 万像素（≈392 visual token），配方不变 |
| v5 | **61min 跑通** | — | QUANTIZE_OK（AWQ 校准 ~31min）→ EXPORT_OK → tgz 5.8GB |

**关键机制**（v4 诊断出土）：TRTEdge 对 VLM+awq_lite **必走多模态校准**
（`_is_image_blind_calibration`：text-only 校准会「把 image-token ranges 优化掉、模型在图像输入上退化」）
——所以 OOM 不能靠改文本校准绕过，只能给图像分辨率封顶。

## 三、G2：Orin 构建与四连环修复

`llm_build` 一次成功（引擎 4.88GB = FP16 的 1/3.11，config 架构全一致），**但运行期首次 prefill 即
pluginV3Runner 断言**。逐层剥出四连环：

1. **CuTe DSL artifact 缺 int4 组**：0.10.1 的 `Int4GroupwiseGemmPluginV2` 全走 CuTe DSL kernel，
   而 goat 十月构建只生成了 `fmha` 组 ⇒ 宏 `CUTE_DSL_INT4_FP16_GEMM_ENABLED` 未定义 ⇒
   插件 `enqueue()` 编译进去的是 `#else return -1;` **死桩**。
   修复：`cutedsl_venv`（cutlass-dsl 4.7.0 + cupy，tuna 镜像）跑
   `build_cutedsl.py --kernels int4_fp16_gemm --gpu_arch sm_87`（24 变体，~2min，合并进现有 bundle）
   + 重配置 `ENABLE_CUTE_DSL="fmha;int4_fp16_gemm"` + 重编插件（先备份 .so）。
   官方基准表本就有 Orin 64GB INT4 AWQ 行（含 Qwen3-VL-8B-Instruct）——此路官方跑通过。
2. **旧引擎插件配置陈旧**：重编后断言移到 `onShapeChange`——旧引擎由死桩插件序列化
   （`mMaxLockWorkspaceBytes=0`）。**教训：引擎的插件配置是构建期序列化产物，换插件库必须重建引擎。**
3. 🔴 **tokenizer 陷阱（与 [`crossgen_a15.md`](crossgen_a15.md) §二同款坑再次出现）**：
   Kaggle repack 带进引擎目录的是 Qwen3-VL 公开 tokenizer，**缺 R1 的 4 个 `traj_history` 特殊 token**
   ⇒ 轨迹历史 tokenize 被跳过 ⇒ CoT 输出退化成 `'.'`、轨迹 diff 0.226（≈seed 轴=随机）。
   修复 = 从 FP16 引擎目录拷 `tokenizer.json` / `tokenizer_config.json` /
   `processed_chat_template.json` 三件覆盖（tokenizer 与量化无关）。
4. **FP16 回归验证**：重编插件后 FP16 探针对拍十月 det_a——文本逐字相同、轨迹 ulp 级（0.0005）
   ⇒ fmha 路径无损，FP16 资产安全。

**G2 smoke 终值**（同输入/同 seed 5/temp 0，对照同日 FP16 探针）：
**CoT 文本逐字相同**；轨迹 mean L2 diff **0.0477** / max 0.148
（= mean|traj| 0.5527 的 **8.6%**，低于 ego-history 轴 12.8%、远低于 seed 轴 49%）。

## 四、G3：minADE6 同口径对比（主结果）

**口径**：chunk 3116 分层 18 clips × 6 seeds（108/108 轨迹完整），官方 `compute_minade.py
--num_traj_samples 6`，与 FP16 基线**同一份请求 JSON / GT / tokenizer / chat template / 积分口径**；
唯一变量 = llm 引擎权重精度（INT4 vs FP16；visual/action 两边同为 FP16 副本）。
脚本：[`scripts/orin/int4_g3_eval.sh`](../../scripts/orin/int4_g3_eval.sh)。

| 指标 | FP16 | INT4 AWQ | Δ（INT4−FP16） |
|---|---|---|---|
| **minADE6@6.4s** | **0.80401 m** | **0.80046 m** | **−0.00356 m（−0.4%）** |
| minADE6@3s | 0.17460 m | 0.17590 m | +0.00130 m |
| 逐 clip @6.4s | — | 10 优 / 8 劣 | max ±0.22 m，mean\|d\|=0.084 |
| 逐 clip @3s | — | 6 优 / 6 劣 / 6 平 | max +0.054/−0.038，mean\|d\|=0.020 |

**判定：Δ=0.004m << 预注册阈值 0.1m ⇒ 「近乎免费」**。逐 clip 差异对称分布（无系统性退化），
量级在 6-seed min 的固有方差内；作为参照，FP16 自身 95% CI 半宽就有 0.30m。

**速度**（llm_bench，同引擎参数 maxInputLen 3424 / maxKV 4096 / batch 1）：

| | FP16 | INT4 AWQ | 比值 |
|---|---|---|---|
| prefill@512 | 277.9 ± 0.5 ms | 290.3 ± 0.25 ms | 0.96×（dequant 开销，compute-bound 段小亏） |
| decode@pastKV512 | 100.3 ± 5.1 ms（10.0 tok/s） | **39.8 ± 9.7 ms（25.1 tok/s）** | **2.52×**（权重读取主导段大赚） |
| 引擎体积 | 15.19 GB | 4.88 GB | **3.11×** |

与七月 CR2 结论同构（decode INT4 快、prefill 略亏），且 decode 加速比更大（2.52× vs CR2 的 1.61×）。

## 五、AV 域校准对照臂（预注册后续变量）

**预注册问题**：§四的 AWQ 校准用 MMMU 通用多模态图像——**换成 AV 域图像会不会改变「近乎免费」结论**？

**设计**：与 MMMU 臂**逐字孪生**，唯一实验变量 = 校准图像域。校准集 = `dgural/PhysicalAI-Autonomous-Vehicles-Sample`
（PhysicalAI-AV 的公开 100-clip 子集，gated=false，Kaggle 直下零 gated 经手）字典序前 8 个 clip ×
R1 的 4 路部署相机（cross_left/right_120fov + front_tele_30fov + front_wide_120fov）× 4 帧
= **128 帧 == MMMU 臂样本数**；经 TRTEdge 自家注册表注入 `av_frames` 数据集（driver 日志验证
`Image calibration dataset: av_frames`），补丁 D 图像封顶同样生效（1920×1080→40 万像素）。
内核 `mamihlapinatapai/r1-vlm-int4-av`（[`cloud/av/kaggle_r1_vlm_int4_av.py`](../../cloud/av/kaggle_r1_vlm_int4_av.py)），
评测 [`scripts/orin/int4av_g3_eval.sh`](../../scripts/orin/int4av_g3_eval.sh)。

**G1'/G2'**：Kaggle 量化 v1-v3 鏖战 cudnn 子库混装（镜像预装 nvidia-cudnn-cu12 与 torch cu130 的
nvidia-cudnn-cu13 装进同一 lib 目录文件级混装，首个 conv3d 即 SUBLIBRARY_VERSION_MISMATCH；三层防御
= 强制重装 torch 声明的精确 cudnn + LD_LIBRARY_PATH 置顶、非致命自适应 conv3d 探针坏则关 cudnn 走原生
conv、atexit 兜底清 working）→ COMPLETE；Orin `llm_build` 一次成功（CuTe DSL int4 组 MMMU 臂已生成，
**插件无需重做**）→ 引擎 4.88GB；tokenizer 三件从 FP16 引擎目录覆盖（G2 教训）→ smoke CoT 与 MMMU 臂
15/18 逐字相同、18/18 轨迹完整。

**G3' 三臂对照**（同口径 chunk 3116，18 clips × 6 seeds = 108/108）：

| 指标 | FP16 | INT4-MMMU | INT4-AV |
|---|---|---|---|
| **minADE6@6.4s** | **0.80401 m** | 0.80046 m | **0.82495 m** |
| minADE6@3s | 0.17460 m | 0.17590 m | 0.17523 m |
| Δ@6.4s vs FP16 | — | −0.0036 m | **+0.0209 m** |

**判定：AV 臂 Δ@6.4s=+0.021m << 预注册阈值 0.1m ⇒ 仍「近乎免费」。**

AV vs MMMU 的 +0.0245m 聚合差经逐 clip 解构 = **噪声，非系统性退化**：

1. **单个 outlier 驱动**——`bdf1fffb`（三臂均 >1.2m 的难 clip，6-seed min 方差天然大）AV-MMMU=+0.383，
   独占聚合差的 +0.0213/0.0245；**剔除后只剩 +0.0034m（≈0）**；
2. **符号检验 8 优 / 10 劣**——对称无方向性；
3. **聚合差 +0.0245m < 自身配对标准误 0.0268m**——统计上与 0 不可区分；
4. **@3s 与 FP16 几乎逐字一致**（+0.0006m）。

**结论：校准数据域（MMMU 通用多模态 vs dgural 公开 AV 帧）不改变「量化近乎免费」结论**——AV 域校准
既没更好也没更差，差异全在 6-seed min 估计器的固有方差内；这反证 §四 MMMU 臂的 Δ≈0 不是「通用校准
恰好适配 AV 任务」的偶然。

## 六、诚实边界

1. **n=18 小样本**：只能说「未检出可测退化」（Δ 比判据阈值小 25 倍、比 FP16 CI 半宽小 75 倍），
   不能声称大规模统计等价。
2. **AWQ 校准域已对照验证**（§五）：MMMU 通用多模态与 dgural 公开 AV 帧两臂 minADE6 差 +0.024m，
   逐 clip 解构全在噪声内（单个 outlier 驱动、< 配对标准误）⇒「校准域不敏感」已由对照臂坐实，
   不再是旁证。两臂均 128 样本、图像封顶 40 万像素、cnn_dailymail 未用于本路径。
3. **校准 tokenizer 是 Qwen3-VL 公开版**（repack 所用）：base BPE 与 R1 完全一致
   （R1 只追加特殊 token，校准文本不含它们），激活统计实际等价；运行期 tokenizer 已修正为 R1 原版。
4. **只量化了 backbone**：visual/action 引擎保持 FP16（实验设计如此——隔离「backbone 量化」单变量；
   action 专家消费量化 KV 的容忍度即由此测得）。
5. 静态单帧评测口径（与 FP16 基线一致）；滚动/闭环场景未测。
6. 逐 clip 数据来自 gated 数据集，不入库；本文聚合数字即全部对外呈现。

## 七、文件与复现

| 文件 | 说明 |
|---|---|
| [`cloud/kaggle_r1_vlm_int4.py`](../../cloud/kaggle_r1_vlm_int4.py) | G1 Kaggle 内核 v5（含 4 个 TRTEdge 补丁 + 5 版失败史注释） |
| [`cloud/kernel-metadata.json`](../../cloud/kernel-metadata.json) | `kaggle kernels push -p cloud/` 即用 |
| [`scripts/orin/int4_g3_eval.sh`](../../scripts/orin/int4_g3_eval.sh) | G3 评测（6 seeds × 18 clips → merge → compute_minade） |
| `outputs/r1_vlm_int4_onnx.tgz.sha256` | 产物指纹归档（tgz 本体不留 Mac；goat 有解包 ONNX + 引擎） |
| goat `engines/alpamayo_r1_int4/` | INT4 llm + FP16 visual/action 三件套（12G） |
| goat `eval_data/minade_chunk3116_int4.csv` | 逐 clip 结果（gated 数据，只在机上） |
| goat `logs/int4_*.log` | 生成/重编/重测/G3 全日志 |
| [`cloud/av/kaggle_r1_vlm_int4_av.py`](../../cloud/av/kaggle_r1_vlm_int4_av.py) | AV 域校准臂 Kaggle 内核 v3（cudnn 混装三层防御 + atexit 兜底清理） |
| [`cloud/av/kernel-metadata.json`](../../cloud/av/kernel-metadata.json) | AV 臂内核元数据（`mamihlapinatapai/r1-vlm-int4-av`） |
| [`scripts/orin/int4av_g3_eval.sh`](../../scripts/orin/int4av_g3_eval.sh) | G3' AV 臂评测（与 int4_g3_eval.sh 逐字同构，仅换引擎/输出名） |
| `outputs/r1_vlm_int4_av_onnx.tgz.sha256` | AV 臂产物指纹归档 |
| goat `engines/alpamayo_r1_int4_av/` | AV 域校准 INT4 llm + FP16 visual/action 三件套（12G） |
| goat `eval_data/minade_chunk3116_int4_av.csv` | AV 臂逐 clip 结果（gated 数据，只在机上） |

CuTe DSL INT4 artifact 复现：`cutedsl_venv/bin/python kernelSrcs/build_cutedsl.py
--kernels int4_fp16_gemm --gpu_arch sm_87`（合并模式，勿带 `--clean`）→
`cmake -B build_orin -DENABLE_CUTE_DSL="fmha;int4_fp16_gemm"` → 重编 → **重建引擎**。
