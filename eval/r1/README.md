# Phase R 评测：Alpamayo-R1 AV 轨迹（全链已跑通）

> **状态（2026-10-27）**：全链三引擎在 Orin sm_87 / JetPack 6.2 / TRT 10.7 上**建成并推理成功**，
> 输出 `output_text`（VLM 思维链）+ `output_trajectory` `[64,2]` = `(accel, kappa)`。
> **minADE 定量评测尚未做**（需官方 644 clips + GT，见下「未做」）。
> 墙的打通过程见 [`../../docs/PROBLEMS.md`](../../docs/PROBLEMS.md) **§M**。

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

## 五、两个必须知道的输入坑

1. 🔴 **`alpamayo_gate4/meta.npz` 的 ego 历史是退化占位数据**——实测 `ego_history_xyz` **全零**、
   `ego_history_rot` **全单位阵**（闸门 4 当年只验「模型能否加载 + 前向跑通」，没管数据真实性）。
   直接拿来喂会得到 `v0=0`，积分出「静止车刹车到倒车 7 米」的物理荒谬轨迹。
   **改用官方指南 `alpamayo.md` 的真实轨迹**（16 点，隐含 ~9.0 m/s）。
2. 🔴 **导出必须设 `HF_ENDPOINT=https://hf-mirror.com`**——export 会去取
   `Qwen/Qwen3-VL-8B-Instruct/config.json`，goat 直连 huggingface.co 不通；
   漏设会卡在网络重试且 **RSS 停在 767 MB 不动**（看着像「在加载」，其实一步没走）。

另：`hf` CLI **不在系统 PATH**，只在 `trtedge-0.10.1/export_venv/bin/hf`
（`which hf` 返回空；且 `ls export_venv/bin | grep hf` 会误命中 `torchfrtrace`）。

## 六、未做（诚实边界）

- ❌ **minADE 定量评测未做**：需官方 644 clips 的多相机帧 + GT（`clip_id -> gt_xy/ego_history_xyz/ego_history_rot`）。
  当前只跑了 **1 个 clip**（gate4 留下的公开样例 `01d3588e-…`），**没有 GT 就无法算 ADE**，
  故只能声称「轨迹结构合法、物理合理、与文本方向自洽」，**不能声称精度**。
- ⚠️ **mRoPE 黄旗未独立验证**：`action/config.json` 写 `rope_type: mrope`、`mrope_section: [24,20,20]`、
  `mrope_interleaved: true`，而分解补丁实现的是 **rotate_half** 约定。判断大概率无碍的依据：
  ① `rope_onnx` 接收的是**已算好的 cos/sin**，mRoPE 分段逻辑在上游；② 导出 dummy 的
  `attention_pos_id = arange(n_diff)` 是 **1D** 位置，而 mRoPE 需 (t,h,w) 三维 ⇒ 疑为从 VLM 继承的残留字段。
  **间接证据**：轨迹连贯、物理合理、无发散（若 RoPE 约定错，输出大概率是垃圾）。但**没有直接对照**。
- ❌ **图像轴未测**：只测了 ego 历史轴与 seed 轴，**没测「换不同驾驶场景图像」**对轨迹的影响
  （手上只有 gate4 那 16 帧单一场景）。这是 conditioning 强度结论的一个真实缺口。
- ⚠️ 输入图像来自 **gated + license:other** 数据集，**只在机上跑，不入库、不上站**
  （见 [`../../docs/PROBLEMS.md`](../../docs/PROBLEMS.md) 与仓库 `.gitignore`）。

## 七、文件

| 文件 | 说明 |
|---|---|
| `r1_seed_sensitivity.py` | 一键重算上表全部数字（确定性底噪 / seed 轴 / ego 历史轴 / 文本头 / 物理单位反归一化） |
| `seed_sensitivity/r1_seed_{det_a,det_b,s0,s1,s2,s3,h0}.json` | 7 次推理原始输出（`output_text` + `output_trajectory [64,2]`） |

复现全链：`scripts/orin/r1_action_minverify.sh`（下载→补丁→对拍→只导 action→算子普查→只建 action 引擎）
+ `scripts/orin/r1_full_chain_infer.sh`（llm/visual 导出与构建 → 输入构造 → 七配置对照）。
