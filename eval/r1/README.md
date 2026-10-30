# Phase R 评测：Alpamayo-R1 AV 轨迹（全链跑通 + minADE 与官方基线一致）

> **状态（2026-10-27）**：全链三引擎在 Orin sm_87 / JetPack 6.2 / TRT 10.7 上**建成并推理成功**，
> 输出 `output_text`（VLM 思维链）+ `output_trajectory` `[64,2]` = `(accel, kappa)`。
> **minADE 定量评测已做**：chunk 3116 分层抽 **18 clips × 6 seeds**，
> **minADE6@6.4s = 0.804 m，与官方基线 0.821 m 一致（−2.1%，落在我们 95% CI 内）**。
> **跨代对比（2026-10-30）**：1.5 经 repack 管线（纯 config 改写、不动权重）跑上 Orin，同口径
> **R1 胜 1.5（0.804 vs 1.815 m，+126%）**，且「1.5 需要导航引导」假设被语义+量化两个独立测量否定——
> 详见 [`crossgen_a15.md`](crossgen_a15.md)。
> 墙的打通过程见 [`../../docs/PROBLEMS.md`](../../docs/PROBLEMS.md) **§M**。
> ⚠️ 输入帧与 GT 来自 **gated + license:other** 数据集，**只在机上跑，帧与逐 clip 原始数据不入库**。

## 一、这是什么

Alpamayo-R1-10B = **VLM 主干（`qwenvl3`，~8.8B）+ 扩散 action 专家（~2.3B，36 层）**，
单次 `action_inference` 跑完两阶段：VLM 读 4 相机 × 4 时刻图像 + 16 点 ego 历史 →
生成思维链文本，action 专家消费 VLM 的 KV cache → 出 **6.4 秒 / 64 个 waypoint（10Hz）** 的
`(加速度, 曲率)` 序列，按 **unicycle 模型**在 BEV 空间积分即得轨迹。

TRTEdge 0.10.1 **原生支持** `alpamayo_r1`（`export.py:162 _ACTION_MODEL_TYPES`），
**推翻了闸门 4 基于 0.9.0 的「TRTEdge 不支持 Alpamayo」旧判**。官方指南
`docs/source/user_guide/examples/vla/alpamayo.md`，官方 minADE 基线 **0.82113 m / 644 clips**。

## 二、三引擎实测（goat，FP16-only）

| 引擎 | 大小 | 构建耗时 | 峰值内存 | 注意力实现 |
|---|---|---|---|---|
| `llm.engine` | 15,193,670,188 B (14.15 GiB) | 172.3 s | **CPU 31.8 G** / GPU 14.5 G | `AttentionPlugin` ×36（TRTEdge 插件） |
| `visual.engine` | 1,165,202,124 B (1.08 GiB) | 112.2 s | CPU 5.4 G / GPU 1.4 G | `ViTAttentionPlugin` ×27（TRTEdge 插件） |
| `action.engine` | 4,565,059,244 B (4.25 GiB) | 89.9 s | CPU 12.1 G / GPU 4.3 G | **分解补丁**（见下） |

**只有 action 专家撞墙**：它用 `trt::attention_onnx` / `rope_onnx` / `kv_cache_update_onnx`，
lower 成 ONNX 标准 `Attention` / `RotaryEmbedding` / `TensorScatter`，**需 TRT≥11**。
llm 与 visual 走 TRTEdge 自己的插件（`trt_edgellm` 域），**算子普查实测墙算子全为 0**，无需新补丁
——goat 的 trtedge-0.10.1 带自生成的 CuTe DSL sm_87+cuda12 预编译件，与 cosmos3「插件版 visual 亦已建成」互证。

action 侧解法 = `scripts/orin/apply_alpamayo_action_decomposed_patch.py`（门控 `USE_DECOMPOSED_ALPAMAYO=1`），
导出前对拍 **16/16 全过**（`test_alpamayo_action_decomposed.py`）。导出图实测
`Attention=0 RotaryEmbedding=0 TensorScatter=0`，代之以 `ScatterElements=72`、`Softmax=36`
（= `num_hidden_layers` 36，每层 k/v 各一次 scatter，**数字自洽**）。

推理期：单次 ~6 s（引擎加载另需 ~25 s），GPU 统一内存 **19919 MiB**，
shared execution context 3740115456 B（base 406719488 / vision 3740115456 / action 158074880）。

## 三、seed 敏感性对照（`r1_seed_sensitivity.py`，证据 `seed_sensitivity/`）

W4 的教训直接适用：**测 conditioning 必须带一个非 conditioning 轴作对照**，否则
「conditioning 弱」与「输出被噪声主导」不可区分。R1 的对照轴是 `--noiseSeed`。

七次推理（`temperature=0`，16 张 gate4 多相机帧）：

| 轴 | mean\|diff\| | 占 mean\|traj\|=0.5527 |
|---|---|---|
| 确定性底（同参两次，**逐位相同**） | 0.00000000 | 0% |
| **SEED**（0/1/2/3 vs 参考 seed 5，4 对） | **0.2711** | **49.05%** |
| SEED（全部 10 对互比均值） | 0.1703 | 30.82% |
| **EGO 历史**（真实 ~9 m/s vs 全零，seed 固定） | 0.0706 | **12.78%** |
| *参照：轨迹内相邻两步* | *0.0269* | *4.86%* |

**SEED / EGO-HIST = 3.84×**，对比 **W4 Cosmos3 policy 的 39.3×** ⇒ R1 的条件耦合比 Cosmos3 policy
**强约一个数量级**，但 seed 仍主导轨迹头。

**文本头则是干净确定的**：`temperature=0` 下 6 次跑（det_a/det_b/s0–s3）输出**完全同一句**
`'Keep lane to continue driving since no critical agent needs attention.'`；
换 ego 历史则整句翻转为 `'Nudge to the left to increase clearance from the parked vehicle on the right.'`
⇒ **VLM 文本头强条件驱动且可复现；扩散轨迹头 seed 主导。**

### ⚠️ 解读口径（别误判成缺陷）

`compute_minade.py` 是 `resps[:N]` 采 N 条再 `ade.min()`——**minADE 的 "min" 本就跨 seed 取最优**。
所以 seed 间差异大是**扩散式多模态轨迹预测器的设计属性**，官方指标已把它吸收进去，不等于缺陷。
但**做 demo 必须固定 `--noiseSeed`** 才可复现。

## 四、轨迹积分（官方口径复算）

按 `compute_minade.py` 的 `estimate_v0`（三阶 Tikhonov λ=1e-6 + ridge 1e-4）+ unicycle 积分，
v0 = **8.641 m/s（31.1 km/h）**：

| | 真实 ego 历史 | 全零 ego 历史 |
|---|---|---|
| accel（反归一化） | [−1.515, −0.203] mean **−0.712** m/s² | [−1.503, −0.059] mean −0.618 m/s² |
| κ（反归一化） | [−6.25e−4, +5.57e−4] /m | [−6.5e−4, +5.0e−4] /m |
| 速度 6.4 s 末 | 8.64 → **4.08** m/s | 8.64 → 4.69 m/s |
| 纵向位移 | **45.1 m** | 48.3 m |
| 横向 y（+ = 左） | **+0.12 m ⇒ LEFT** | +0.11 m ⇒ LEFT |
| 航向变化 | 峰值 +0.28°，末 −0.07° | 峰值 +0.28°，末 −0.12° |

全部数值在 `action_space_cfg` 界内（accel ±9.8、curvature ±0.33）、全 finite、无发散。
**v1 的横向符号（向左）与其自身文本「Nudge to the left」一致**——语言头与动作头方向自洽。

## 五、minADE 定量评测（chunk 3116，18 clips × 6 seeds）

### 数据与口径

**来源**：`nvidia/PhysicalAI-Autonomous-Vehicles`（`gated: auto` + `license: other`）。
🔴 **官方那 644 clips 拿不到**——`compute_minade.py:19` 写的是 *"Reference: **chefu** PyTorch
baseline minADE = 0.82113 m on 644 clips"*，`chefu` 是 NVIDIA 内部人名，**这 644 条是内部评测集**；
两个公开 split（`alpamayo_aligned` = train 276,299 / test 28,317 / alpasim 1,536；
`default` = 153,625 / 90,928 / 61,599）里**都没有 644 这个规模**。
⇒ 改用 `alpamayo_aligned` 的 **test** split（数据集 README 确认它 *"follows the train/test
partition of NVIDIA's Alpamayo models"*）——**分布对，但 clip 集不同，不可声称同口径复现**。

**选片**：chunk 3116 是纯 test chunk（100 clips 全 test+valid）。按轨迹形态**分层抽 18 条**
（每 (形态×速度) 格取 |横向| 最大的 3 条）：straight 9 / turn 6 / gentle 3；fast 9 / mid 5 / slow 4。
v0 跨 **0.0–26.6 m/s**、maxlat 跨 **0.00–49.10 m**、fwd 跨 0.0–167.2 m，含静止（`0338f722`）
与起步加速（`82cc0eec` v0=0→fwd 24.6m）两个边界样本。

**GT 构造**：从 `labels/egomotion/` 的 per-clip parquet（`timestamp` µs + 四元数 + xyz，~100 Hz）
按 **t0=5.1 s**（官方口径，见 `gate4_forward.py`）取 16 点历史 + 64 点未来（均 10 Hz），
变换到 t0 的 ego 系 `p_ego = R0ᵀ(p−p0)`。**关键修正**：`p0` 必须在 t0 **精确插值**而非取最近采样点
——否则快车（18 m/s）会因 ~3 ms 采样偏移产生 5.7 cm 残差；修正后历史末点残差 **0.00000 m**。

**帧抽取**：相机 mp4 是 1920×1080 / 30 fps / hevc / 20.13 s。用随附的
`<clip>.<cam>.timestamps.parquet`（604 帧的权威时间戳）定位 t0−0.3/−0.2/−0.1/0 的最近帧
= **索引 144/147/150/153**（相机时间戳与 egomotion 共享 epoch，恒定偏移 −10.85 ms）。
相机顺序按 `gate4_forward.py` 的 `camera_indices [0,1,2,6]`：
cross_left → front_wide → cross_right → front_tele（camera-major，每相机 4 帧）。

### 结果

官方 `compute_minade.py --num_traj_samples 6`（每 clip 6 个 `--noiseSeed`，取 min）：

| 指标 | 我们（n=18） | 官方参考 | 差异 |
|---|---|---|---|
| **minADE6@6.4s** | **0.80401 m** | **0.82113 m**（chefu PyTorch, 644 clips） | **−2.1%** |
| minADE6@3s | 0.17460 m | 0.254 m（paper Table 6, AR1 **0.5B**+Route） | −31%（**不同模型**，仅参考） |

| 统计量 | @3s | @6.4s |
|---|---|---|
| mean | 0.17460 | **0.80401** |
| median | 0.18458 | 0.60216 |
| sd | 0.1075 | 0.6458 |
| **95% CI** | [0.1249, 0.2243] | **[0.5057, 1.1024]** |
| min / max | 0.0309 / 0.3371 | 0.0949 / 2.5043 |

**官方基线 0.82113 落在我们 95% CI [0.506, 1.102] 内** ⇒ 统计上一致。

### 分层拆解

| 轨迹形态 | n | @3s | @6.4s |
|---|---|---|---|
| gentle（maxlat 1–5 m） | 3 | 0.1797 | **0.3617** |
| straight（maxlat <1 m） | 9 | 0.1319 | 0.7316 |
| turn（maxlat >5 m） | 6 | 0.2361 | **1.1338** |

| 速度档 | n | v0 均值 | @3s | @6.4s |
|---|---|---|---|---|
| fast（>8 m/s） | 9 | 15.5 | 0.2132 | 0.8976 |
| mid（3–8） | 5 | 6.1 | 0.1518 | 0.8495 |
| slow（<3） | 4 | 1.0 | 0.1163 | 0.5366 |

**物理合理性交叉验证**（这是「数字可信」的关键旁证）：
- ADE 与速度/距离**正相关**：corr(v0, @6.4s)=**+0.43**、corr(fwd, @6.4s)=**+0.61**、
  corr(maxlat, @6.4s)=+0.27 ⇒ 越快越远绝对误差越大，符合运动学预期，不是随机数
- **相对误差中位数仅 1.91%**（ADE@6.4s ÷ 纵向位移）
- 最差 3 条全是高速/大位移：`d1a20a67`（v0=26.6, fwd=167.2m）2.504 m、
  `6b7467e8`（turn, maxlat=49.1m）1.811 m、`bdf1fffb`（turn）1.418 m
- 最好 3 条全是近静止：`1d782ead` 0.095 m、`0338f722` 0.095 m、`1f2e312c` 0.253 m

### ⚠️ 诚实边界

1. **n=18 太小**，95% CI 宽达 [0.51, 1.10]（半宽 0.30 m）⇒ 只能说**「与基线一致」**，
   **不能说「复现了 0.82113」更不能说「优于基线」**。−2.1% 的点估计差异远小于 CI 半宽，无统计意义。
2. **clip 集与官方 644 不同** ⇒ 严格来说**不可同口径对比**，只能说「在 Alpamayo 官方 test 划分的
   一个分层子集上，量级与官方基线一致」。
3. @3s 对比的 paper 数字是 **AR1 0.5B + Route**（不同模型、不同规模），**不是同模型对照**，仅供参考。
4. **分层格子样本极少**（gentle 只有 3 条、slow 只有 4 条），分层数字**只能看趋势不能当结论**。
5. 帧与 GT 来自 gated 数据集 ⇒ **帧、GT、逐 clip 原始数据一律不入库**，只留聚合数字与方法。

## 六、两个必须知道的输入坑

1. 🔴 **`alpamayo_gate4/meta.npz` 的 ego 历史是退化占位数据**——实测 `ego_history_xyz` **全零**、
   `ego_history_rot` **全单位阵**（闸门 4 当年只验「模型能否加载 + 前向跑通」，没管数据真实性）。
   直接拿来喂会得到 `v0=0`，积分出「静止车刹车到倒车 7 米」的物理荒谬轨迹。
   **改用官方指南 `alpamayo.md` 的真实轨迹**（16 点，隐含 ~9.0 m/s）。
2. 🔴 **导出必须设 `HF_ENDPOINT=https://hf-mirror.com`**——export 会去取
   `Qwen/Qwen3-VL-8B-Instruct/config.json`，goat 直连 huggingface.co 不通；
   漏设会卡在网络重试且 **RSS 停在 767 MB 不动**（看着像「在加载」，其实一步没走）。

另：`hf` CLI **不在系统 PATH**，只在 `trtedge-0.10.1/export_venv/bin/hf`
（`which hf` 返回空；且 `ls export_venv/bin | grep hf` 会误命中 `torchfrtrace`）。

## 七、未做（诚实边界）

- ⬜ **官方 644-clip 同口径评测做不到**：那 644 条是 NVIDIA 内部集（`compute_minade.py:19` 注释
  *"chefu PyTorch baseline"*），公开数据集无此清单。已在 `alpamayo_aligned` test split 分层抽
  **18 clips** 做了量级一致的评测（§五：0.804 m vs 基线 0.821 m），但**扩到更大样本、或同口径
  复现 0.82113，仍受限于数据可得性**。
- ⚠️ **mRoPE 黄旗未独立验证**：`action/config.json` 写 `rope_type: mrope`、`mrope_section: [24,20,20]`、
  `mrope_interleaved: true`，而分解补丁实现的是 **rotate_half** 约定。判断大概率无碍的依据：
  ① `rope_onnx` 接收的是**已算好的 cos/sin**，mRoPE 分段逻辑在上游；② 导出 dummy 的
  `attention_pos_id = arange(n_diff)` 是 **1D** 位置，而 mRoPE 需 (t,h,w) 三维 ⇒ 疑为从 VLM 继承的残留字段。
  **间接证据（§五 大幅加强）**：18 个不同场景的轨迹 minADE 与官方基线一致、ADE 与速度/距离正相关、
  相对误差中位数仅 1.91%——**若 RoPE 约定错，输出会是垃圾而不可能对上 GT**。但仍**没有直接对照**。
- ⬜ **图像轴的 conditioning 敏感性未单独定量测**：§三 的 seed 对照只用了 gate4 单一场景。
  不过 §五 的 minADE 跨 **18 个不同场景**且与基线一致，已**间接证明图像 conditioning 有效**
  （否则不同场景会给出同样的错误轨迹、ADE 会爆）。「固定 seed、只换图像」的定量敏感性对照仍未做。
- ⚠️ 输入图像与 GT 来自 **gated + license:other** 数据集，**只在机上跑，帧/GT/逐 clip 原始数据
  一律不入库、不上站**（见 [`../../docs/PROBLEMS.md`](../../docs/PROBLEMS.md) 与仓库 `.gitignore`）。

## 八、文件

| 文件 | 说明 |
|---|---|
| `r1_seed_sensitivity.py` | 一键重算 §三 全部数字（确定性底噪 / seed 轴 / ego 历史轴 / 文本头 / 物理单位反归一化） |
| `seed_sensitivity/r1_seed_{det_a,det_b,s0,s1,s2,s3,h0}.json` | 7 次推理原始输出（`output_text` + `output_trajectory [64,2]`） |
| `extract_frames.py` | §五 帧抽取脚本（用 `<clip>.<cam>.timestamps.parquet` 定位 t0−0.3/−0.2/−0.1/0 → ffmpeg 抽 4 帧/相机） |

复现全链：`scripts/orin/r1_action_minverify.sh`（下载→补丁→对拍→只导 action→算子普查→只建 action 引擎）
+ `scripts/orin/r1_full_chain_infer.sh`（llm/visual 导出与构建 → 输入构造 → 七配置对照）。

⚠️ **§五 的逐 clip 数据不入库**：`minade_chunk3116.csv`、`selected_clips.csv`、
`chunk3116_clips.csv`、帧与 GT 均来自 gated + license:other 数据集，**只在 goat 机上**
（`/home/nvidia/chenrunbin/alpamayo-edge/eval_data/`）。本 README §五 的聚合数字
（mean / median / 95% CI / 分层拆解）即其全部对外呈现。
