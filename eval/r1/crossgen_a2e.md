# 跨代对比臂 3：Alpamayo2-Edge（2.65B）vs R1（10B）——轨迹域「缩小免费」

> **状态（2026-11-15）**：`nvidia/Alpamayo2-Edge` 于 **2026-11-10 正式发布**（非 gated，权重
> OpenMDW-1.1 / 代码 Apache-2.0）。经 **PyTorch BF16 直跑上 Orin**（官方 edge 部署路为 Thor 专属），
> 在 chunk 3116 与 R1 完成**同口径 minADE 对比**。四条结论：
> 1. **2.65B 点估计优于 10B R1**（minADE6@6.4s **0.734 vs 0.804 m**，−8.7%），配对检验
>    **p=0.61 ⇒ 统计平手**——「缩小免费」在 VLA 轨迹域成立（n=18 分辨率内）；
> 2. 与 **VQA 推理域对照**（Cosmos3-Edge 2B reasoner 显著输 CR2-8B 量化版，McNemar p=0.0115，
>    见 [cosmos-edge 仓](https://github.com/ranpin/cosmos-edge)）⇒ **缩小代价是域依赖的**；
> 3. 逐 clip 形态：**2-Edge 压缩尾部**（高速大位移 clip 赢 0.5–1.5 m）、容易 clip 略抬底；
> 4. 官方 edge 部署 recipe = **Thor 专属**（FP8/sm_110/TRT 11.3）；Orin 唯一可行路 = PyTorch BF16
>    直跑，需三处本地补丁（全部披露于 [`a2e/official_pkg_orin.patch`](a2e/official_pkg_orin.patch)）。
>
> 口径同 [`README.md`](README.md) §五：**同 18 clips、同 t0=5.1s、6 候选取 min、同 GT、同 2D xy 评分**。
> ⚠️ 帧与 GT 来自 gated + license:other 数据集，只在机上跑，逐 clip 原始数据不入库。

## 一、动机与模型

**谱系修正**：2026-10-15/10-29 两次调研结论「Alpamayo 家族最小 backbone 是 10B」在当时正确
（HF repo 2026-08-12 建立但未公开），**2026-11-10 正式发布后作废**——Alpamayo2-Edge 是
2B 级边缘 AV VLA。要回答的问题：**专用小模型（2.65B）在同一轨迹基准上能否打平/打赢 10B R1？**
这直接检验本项目「缩小不免费」结论（源自 VQA 域）在轨迹域是否成立。

**模型**：总 2.65B BF16（权重 ~5.3G）= **Cosmos3-Edge VLM backbone 2.45B**（LM 1.96B +
SigLIP2 0.41B + projector 0.08B）+ **0.2B flow-matching 扩散轨迹头**。输入 = 6 相机 × 22 帧 +
20 点 ego 历史 + **非空导航文本**；输出 = Chain-of-Causation CoT + 64 waypoints（0.1–6.4s，
accel+κ unicycle 积分）——**与 R1 同动作空间谱系**（`action_space` 常数与 1.5 逐位一致：
accel_mean 0.029027 / std 0.681043）。官方数字：minADE6@**5s** = 0.52 m（1392 PhysicalAI-AV
样本）、AlpaSim 闭环 15.15 km/incident、LingoQA 75.2；峰值显存 12.06 GiB（BF16、6 候选）。

## 二、怎么跑上 Orin 的（PyTorch BF16 + 三处补丁）

**部署路径侦察**：官方 edge recipe（`NVlabs/alpamayo-recipes/alpamayo2edge_edgellm`）实测环境 =
DRIVE AGX **Thor**（DRIVE OS 7.2.7.0 / CUDA 13.4 / TRT 11.3.1.9 / `--target-sm 110`，FP8 权重+激活+KV，
~200 ms/请求），且 stock TRTEdge action exporter **只支持 R1**；TRTEdge main（含 v0.12.0）无
`alpamayo2` 支持；Orin sm_87 无 FP8/NVFP4（硬件代差，同 FP4 旧结论）。⇒ **Orin 唯一可行路 =
官方 PyTorch 推理包 BF16 直跑**（12.06 GiB 峰值，goat 64G 宽裕）。

**环境（全部披露的偏差）**：py3.10 venv（官方 pin `>=3.12,<3.13`，但 torch 2.8.0 cu126 aarch64
wheel 只有 cp310）+ torch 2.8.0 / torchvision 0.23.0（jetson-ai-lab cu126 wheel，与官方 pin 精确一致）
+ transformers 5.17.0（首个支持 `cosmos3_edge` 的版本）+ scipy 1.15.3（py3.10 上限，官方要 ≥1.16）
+ numpy 1.26.4（jetson torch wheel 按 numpy 1.x 编译，numpy 2 会炸 `from_numpy`）。
torch wheel 经 **Kaggle kernel 中转**下载（jetson-ai-lab 国内单连接 ~26KB/s 且并行 Range 被服务端
截断；sha256 双端校验）。

**官方包三处补丁**（[`a2e/official_pkg_orin.patch`](a2e/official_pkg_orin.patch)，基于官方 commit
`02f55a7`，语义等价、均单元验证）：
1. `pyproject.toml`：`requires-python` 放宽到 `>=3.10`；
2. `chat_template.py`：f-string `z` 格式标志（py3.12+ 特性，负零归一）→ py3.10 等价函数
   （`-0.0`→`0.0`，用官方注释所述的相机微秒偏差场景验证）；
3. `action_space/utils.py` 4 处 `cholesky/cholesky_solve` + 1 处 `vector_norm` → CPU 执行
   （**JetPack 6.2 cusolver 11.6.4 缺 `cusolverDnXsyevBatched_bufferSize`**，torch 2.8 cu126 的
   `libtorch_cuda_linalg.so` dlopen 失败；Tikhonov 小矩阵 CPU 求解数值同解、开销可忽略；
   `vector_norm`→`norm` 为防御性等价替换）。

**输入构造镜像官方 loader**（`load_physical_aiavdataset.py` 契约）：6 相机索引 [0,1,2,3,5,6]
（cross_left/front_wide/cross_right/rear_left_70fov/rear_right_70fov/front_tele），前 2 路取
t0−2/−1/−0.5/−0.1/0s 五帧、其余四路取 −1/−0.1/0s 三帧，**帧选「≤请求时刻的最近帧」并保留真实
时间戳**（时间戳是模型输入，进 prompt 的相对时间标签）；ego 历史 20 点（−1.9..0s @0.1s，
位置线性插值 + 旋转 slerp）变换到 t0 ego 系（`R(t0)⁻¹`，与官方公式逐项一致）；**nav 用数据集自带
真实标注**（`navigation/navigation.parquet`，16.3M 行，18/18 clips 命中，取 ≤t0 最近一条，
如 "Continue straight" / "Continue in 93m"）——非 oracle 反推。

**取数**：6 路相机 chunk_3116 zip（各 1.27–2.12 GB，全量 ~10.6 GB）用 **HTTP Range 部分下载**
只取 18 clips 的 mp4 + timestamps parquet（实际下载 ~2.7 GB），Mac 端 ffmpeg 抽帧后 scp 到 goat
（647 MB PNG，逐 clip 22 帧全齐）。

## 三、结果：统计平手、点估计小模型略优

同口径：chunk 3116 分层 18 clips、t0=5.1s、6 候选取 min、同 GT（`gt_chunk3116.json`）、
ADE = 64 点 2D xy 欧氏距离均值（@5s = 前 50 点）。2-Edge 设置 = 官方 CLI 默认
（seed 43、temperature 0.6、top_p 0.98、diffusion 10 步）+ `num_traj_samples=6`
（6 条独立候选、各带独立采样的 reasoning，对应 R1 臂的 6 seeds）。

| 指标 | R1 FP16 (10B) | R1 INT4 (10B) | 1.5 FP16 (10B) | **2-Edge (2.65B)** |
|---|---|---|---|---|
| mean minADE6@6.4s | 0.80401 m | 0.80046 m | 1.8153 m | **0.73405 m** |
| 95% CI | [0.506, 1.102] | — | — | [0.493, 0.976] |
| median @6.4s | 0.602 m | — | 1.320 m | 0.597 m |
| mean minADE6@5s | — | — | — | 0.488 m |

**配对统计（2-Edge vs R1 FP16，n=18）**：逐 clip **10 优 / 8 劣**；paired t **p=0.61**；
Wilcoxon p=0.83；差值（R1−2-Edge）mean +0.070 m，95% CI **[−0.196, +0.336]** ⇒
**未检出显著差异，点估计偏向 2-Edge**。

**逐 clip 形态 = 尾部压缩**：2-Edge 的最大增益全在 R1 的高速/大位移困难 clip——
`d1a20a67`（v0=26.6 m/s）**1.01 vs 2.50**、`bdf1fffb` **0.43 vs 1.42**、`6b7467e8`（大转角）
**1.30 vs 1.81**、`53a9f0d2` **0.28 vs 0.80**；代价是容易 clip 略抬底（`932ad789` 2.03 vs 1.17、
`9e163ccb` 1.04 vs 0.35、`7451ffbe` 0.84 vs 0.28）。均值改善由尾部驱动。

**速度**：18 clips 全链总 **211 s（~11 s/clip，含 6 候选 + CoT 生成）**，PyTorch BF16 零优化
（SDPA、无 compile/CUDA graph）——对照 R1 TRTEdge FP16 单候选 ~5.7 s。2-Edge 免量化直跑
（权重 5.3 G），无「量化必需」问题。

**官方数字旁证**：我们 @5s = 0.488 m vs 官方 0.52 m（1392 样本）——clip 集不同（我们是 18 条
困难机动 clip）不可直接比，量级一致。

## 四、结论：缩小代价的域依赖

1. **轨迹域缩小免费**：2.65B 专用模型 ≈ 10B R1（统计平手、点估计略优、速度更快、免量化）。
2. **推理域缩小不免费**：Cosmos3-Edge 2B reasoner 在 n=184 驾驶 VQA 显著输 CR2-8B 量化版
   （42.4% vs 51.6%，McNemar p=0.0115）。
3. 合并表述：**「缩小不免费」精确化为「推理域缩小不免费、轨迹域缩小免费」**——轨迹任务输出
   低维（64×2）、有强域内训练与 nav/ego 条件化，不需要大 backbone 的通用推理容量；VQA 需要。
4. 与量化结论的关系：R1 INT4（0.800）≈ R1 FP16（0.804）≈ 2-Edge（0.734）——轨迹任务上
   「10B+量化」与「2.65B 原生」是同级候选，选型轴转到速度/内存/部署形态（2-Edge 免量化、
   PyTorch 即跑；R1 需 TRTEdge 全链）。

## 五、诚实边界

1. **n=18 太小**：p=0.61 只能说「未检出显著差异」，对 0.07 m 量级差异检验功效不足；
   「点估计略优」**不构成优越性声称**。
2. **输入协议不同**：2-Edge 带真实 nav + 6 相机 22 帧；R1 臂无 nav + 4 相机 4 帧——模型能力差
   与输入信息差**混杂**。（R1 跨代研究测得 nav 对 R1/1.5 仅 −1% within-noise，但 2-Edge 是
   nav-conditioned 训练，nav 贡献无法从本数据剥离。）
3. **单次运行**（seed 43、temperature 0.6 采样）；6 候选来自同一 prefill 的独立 reasoning 采样，
   与 R1 臂 6 个独立 seed 全链运行不完全同构。
4. **PyTorch 路径非官方部署形态**（官方 = Thor FP8 TRT ~200 ms）；三处补丁语义等价但未经官方确认。
5. clip 集与 R1 臂相同（分层困难 clip），绝对值与官方 1392 样本不可比。
6. 帧/GT/nav 来自 gated 数据集：只在机上跑，逐 clip 原始数据不入库（本文 clip id 仅作定位）。

## 六、文件与复现

| 文件 | 跑在哪 | 说明 |
|---|---|---|
| [`a2e/fetch_a2e.py`](a2e/fetch_a2e.py) | Mac | Range 部分下载 6 相机 zip 取 18 clips mp4+timestamps → ffmpeg 抽 22 帧/clip（需 `~/.hf_token`） |
| [`a2e/a2e_build_env2.sh`](a2e/a2e_build_env2.sh) | goat | venv 构建：本地 wheel + 清华镜像 + clone 官方包 + 补丁前置 |
| [`a2e/official_pkg_orin.patch`](a2e/official_pkg_orin.patch) | goat | 官方包三处补丁（基于 commit `02f55a7`，`git apply` 即用） |
| [`a2e/a2e_run.py`](a2e/a2e_run.py) | goat | 内存构造输入（镜像官方 loader 契约）+ 批量推理（官方 `run_data`，6 候选） |
| [`a2e/a2e_score.py`](a2e/a2e_score.py) | goat | 评分：minADE6@6.4s/@5s 对照 `gt_chunk3116.json` → CSV + 聚合 |

机上产物（goat，**不入库**）：`~/chenrunbin/a2e/`（权重 5.0G、venv、`frames_a2e/` 647M、
`eval_a2e/out/` 18 clips 原始输出）、`eval_data/minade_chunk3116_a2e.csv`。
本文 §三 的聚合数字即其全部对外呈现。脚本内联路径为本次会话工作目录（`/tmp/a2e`、
`/home/nvidia/chenrunbin/a2e`），复现时改成你的位置。
