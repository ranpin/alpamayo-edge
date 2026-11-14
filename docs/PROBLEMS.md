# 问题记录 —— Alpamayo 家族篇（§M）

> 2026-11-15 仓库按 NVIDIA 模型家族重组后，本文件只保留 **§M（Alpamayo-R1 action 专家）**；
> §A–J（CR2-8B/2B 量化部署）与 §K/L（Cosmos3-Edge 部署与 policy 动作链）已随 CR2/Cosmos3 资产
> 迁入 [cosmos-edge 仓的 PROBLEMS.md](https://github.com/ranpin/cosmos-edge/blob/main/docs/PROBLEMS.md)
> （编号保持不变，历史互引不失效）。
> 每条：**现象 → 根因 → 解决 → 教训**。均为真实发生、可复现。

---

## M. Alpamayo-R1 action 专家：K6 同源墙，分解补丁第三次打通（2026-10-26，Phase R 最小验证 ✅）

**背景**：闸门 4 曾基于 TRTEdge **0.9.0** 判定「TRTEdge 不支持 Alpamayo」而关闭 AV 轨迹路线。
2026-10-19 静态侦察**推翻该旧判**——`export.py:162 _ACTION_MODEL_TYPES = frozenset(["alpamayo_r1"])`，
0.10.1 **原生支持** `alpamayo_r1`，且官方指南 `docs/source/user_guide/examples/vla/alpamayo.md` 齐备、
`action_build`/`action_inference` 二进制已编好、`compute_minade.py` 带公开基线 **0.82113 m / 644 clips**。
⇒ 路线重开，用户拍板「按最小验证走」：**只导 + 只建 action 单引擎**，不先碰 10B llm build。

### M1. 墙：action 专家撞 K6 同源墙

`models/alpamayo/modeling_alpamayo_action.py` 的 `ActionAttention` 用三个 `trt::` custom op，
lower 成 **ONNX 标准算子**，只有 TRT≥11 parser 认识（Orin JetPack 6.2 = TRT 10.7）：

| custom op | lower 成 | 与既有墙的关系 |
|---|---|---|
| `trt::attention_onnx` | `Attention` | **= K6 同一个** |
| `trt::rope_onnx` | `RotaryEmbedding` | **= K6 同一个** |
| `trt::kv_cache_update_onnx` | `TensorScatter` | **新**（cosmos3 policy 没有这个） |

三个 op 在 `models/ops.py` 里都是 **stub（`return x.clone()`）**——真实计算由 TRT 侧承担，
所以 PyTorch 端数值对拍无法覆盖它们，必须自己写分解实现。

### M2. 解法：分解补丁（`scripts/orin/apply_alpamayo_action_decomposed_patch.py`）

同 K4/K6 思路，用基础 ONNX 算子实现，任何 TRT 版本可解析。门控 **`USE_DECOMPOSED_ALPAMAYO=1`**
（独立于 cosmos3 的 `USE_DECOMPOSED_POLICY`，两套补丁互不干扰）。改两处：`models/ops.py` 追加三个
分解实现，`modeling_alpamayo_action.py:187` 的局部 import 换成走包装器（别名回原名，调用点不动）。

**🔴 不能直接复用 cosmos3 policy 补丁——三个陷阱都只在 B>1 暴露，而导出 dummy 恰好是 `B=1`**
（`get_onnx_export_args` 里 `B = 1`），照抄会在建 `--maxBatchSize 6` 引擎时才炸：

| # | 陷阱（源码实测） | 处理 |
|---|---|---|
| 1 | RoPE 的 cos/sin **只有 S 行、跨 batch 共享**（forward 里 `rope_cos = rope_rotary_cos_sin[0:1].reshape(-1,half)`） | cosmos3 版 `cos.reshape(b,s,half)` 在 B>1 **元素不足**；改为按 `position_ids` 行索引后广播。position_ids 须保留作图输入防 DCE |
| 2 | `attn_mask` 是 **4D `[B,1,S,Sk]`**，而 GQA 折叠后 scores 是 **5D `[B,Hkv,rep,S,Sk]`** | 不升维直接相加只在 B==1 碰巧广播正确 ⇒ `unsqueeze(2)` 升到 `[B,1,1,S,Sk]` |
| 3 | `kv_cache_update` 的 `cache_indices` 是**运行期张量输入**（dummy=`torch.full((B,),3027)`、dynamic_shapes 标 `{0:batch}`） | trace 期值未知 ⇒ `torch.cat` 静态切片不可用；改 `scatter`（索引仅 `[B,H,S,D]`，比 gather+mask 的 `[B,H,C,D]` 省 ~C/S 倍） |

**导出前数值对拍**（`test_alpamayo_action_decomposed.py`，**16/16 全过**）：`kv_cache_update` vs 直接切片赋值
**err=0.000e+00**（逐位精确，含「写入区外逐位不变」护栏，覆盖 start=3027/0/C−S × B=1/2）；
`attention` vs `F.scaled_dot_product_attention` **err~1e-7**；`rope` vs LLaMA 全维公式 **err=0**；dispatch 3 项。
其中陷阱 2 的护栏**实证了陷阱为真**：B=2 时 naive 写法直接抛 `RuntimeError`（不是算错，是根本广播不了）。

### M3. 实测结果：通了

| 阶段 | 实测 |
|---|---|
| 权重下载 | 21 G / 5 shard，hf-mirror + `HF_HUB_DISABLE_XET=1`，~50 MB/s 约 7 分钟 |
| **export**（`--skip-llm --skip-visual`） | 峰值 RSS **10.5 G**、最低 MemAvailable **37 G**、约 5 分钟；`model.onnx` 6,314,629 B（图）+ `model.onnx.data` **4,561,240,064 B**（外置权重） |
| **算子普查** | `Attention=0 RotaryEmbedding=0 TensorScatter=0 KVCacheUpdate=0`；`ScatterElements=72`、`Softmax=36`、`MatMul=325`（共 3218 节点） |
| **action_build**（`--maxBatchSize 6`） | Engine generation **89.9 s**、峰值 **CPU 12088 MiB / GPU 4349 MiB**、**0 error** ⇒ `action.engine` **4,565,059,244 B** |

**自洽性交叉验证**：`action/config.json` 给出 `num_hidden_layers: 36` ⇒ `Softmax=36`（每层 1 个注意力）、
`ScatterElements=72 = 36×2`（每层 k/v 各一次 scatter）。**数字精确对上**，证明分解逐层落进了图里，
不是「碰巧没有墙算子」。

⇒ **结论：Orin sm_87 / JetPack 6.2 / TRT 10.7 上，Alpamayo-R1 的 turnkey action 路径跑得起来。**
资源开销远低于预期（对比闸门 4 当年「8B 级加载峰值 54.8 G」，action 专家构建峰值仅 12.1 G）。
**分解注意力补丁的普适性第三次得到验证**（K4 cosmos3 visual → K6 cosmos3 policy → M2 alpamayo action）。

### M4. 全链已通（2026-10-27）：三引擎建成 + 推理成功 + 轨迹积分

llm / visual 走 TRTEdge 自己的插件（算子普查实测墙算子全 0，无需新补丁）；只有 action 专家需分解补丁。
三引擎在 goat 建成（FP16-only）：

| 引擎 | 大小 | 构建耗时 | 峰值内存 | 注意力实现 |
|---|---|---|---|---|
| `llm.engine` | 14.15 GiB | 172.3 s | **CPU 31.8 G** / GPU 14.5 G | `AttentionPlugin` ×36 |
| `visual.engine` | 1.08 GiB | 112.2 s | CPU 5.4 G / GPU 1.4 G | `ViTAttentionPlugin` ×27 |
| `action.engine` | 4.25 GiB | 89.9 s | CPU 12.1 G / GPU 4.3 G | **分解补丁**（§M2） |

我此前外推「llm build 峰值 ~43 G、可能 OOM」**没有发生**——实测 31.8 G；但 MemAvailable 峰值时确实掉到
**8.5 G**，离危险不远，`--maxBatchSize 1` 的降压决定是对的。推理期 GPU 统一内存 **19919 MiB**、单次 ~6 s。

**推理成功且输出物理合理**（16 张 gate4 多相机帧 + 官方指南真实 ego 历史 ~9 m/s，`estimate_v0`=8.641 m/s）：
`output_text` = *"Keep lane to continue driving since no critical agent needs attention."*；
`output_trajectory [64,2]` 反归一化后 accel mean **−0.712 m/s²**、κ mean −5.7e−5 /m，
积分得 6.4 s 走 **45.1 m**、速度 8.64→4.08 m/s、横向 **+0.12 m（向左）**、航向变化 <0.3°，
全部在 action_space 界内、全 finite、无发散。**横向符号与文本「nudge left」自洽**。

### M5. seed 敏感性对照（W4 教训的 R1 版，`eval/r1/r1_seed_sensitivity.py`）

W4 的教训直接适用：**测 conditioning 必须带一个非 conditioning 轴作对照**。R1 的对照轴是 `--noiseSeed`。
**关键设计：必须 `temperature=0`**——VLM 文本默认 temp=1.0 采样，生成的 CoT token 会进 action 专家
conditioning，不停掉文本采样就量不出干净的 seed 轴。七次推理实测：

| 轴 | mean\|diff\| | 占 mean\|traj\|=0.5527 |
|---|---|---|
| 确定性底（同参两次，**逐位相同**） | 0.00000000 | 0% |
| **SEED**（0/1/2/3 vs 参考 seed5，4 对） | **0.2711** | **49.05%** |
| **EGO 历史**（真实 ~9 m/s vs 全零，seed 固定） | 0.0706 | **12.78%** |
| *参照：轨迹内相邻两步* | *0.0269* | *4.86%* |

**SEED / EGO-HIST = 3.84×**，对比 **W4 Cosmos3 policy 的 39.3×** ⇒ R1 条件耦合比 Cosmos3 policy
**强约一个数量级**，但 seed 仍主导轨迹头。**文本头则干净确定**：temp=0 下 6 次跑输出完全同一句，
换 ego 历史整句翻转（"Keep lane…" ↔ "Nudge to the left…"）⇒ **VLM 文本头强条件驱动且可复现；
扩散轨迹头 seed 主导**。

⚠️ **解读口径（别误判成缺陷）**：`compute_minade.py` 是 `resps[:N]` 采 N 条再 `ade.min()`——
**minADE 的 "min" 本就跨 seed 取最优**，故 seed 间差异大是扩散式多模态轨迹预测器的设计属性，
官方指标已吸收，不等于缺陷。但**做 demo 必须固定 `--noiseSeed`** 才可复现。

### M6. minADE 定量评测（2026-10-27，chunk 3116，18 clips × 6 seeds）

**数据**：`nvidia/PhysicalAI-Autonomous-Vehicles`（`gated: auto` + `license: other`）。🔴 **官方那 644 clips
拿不到**——`compute_minade.py:19` 写 *"Reference: **chefu** PyTorch baseline minADE = 0.82113 m on 644 clips"*，
`chefu` 是 NVIDIA 内部人名、644 条是内部评测集；两个公开 split（`alpamayo_aligned` = train 276,299 /
test 28,317 / alpasim 1,536；`default` = 153,625 / 90,928 / 61,599）里**都没有 644 这个规模**。
⇒ 改用 `alpamayo_aligned` 的 **test** split（数据集 README 确认它 *"follows the train/test partition of
NVIDIA's Alpamayo models"*）——**分布对，但 clip 集不同，不可声称同口径复现**。

**选片**：chunk 3116 是纯 test chunk（100 clips 全 test+valid）。按轨迹形态**分层抽 18 条**
（每 (形态×速度) 格取 |横向| 最大的 3 条）：straight 9 / turn 6 / gentle 3；fast 9 / mid 5 / slow 4；
v0 跨 0.0–26.6 m/s、maxlat 跨 0.00–49.10 m，含静止与起步加速两个边界样本。

**GT 构造**：从 `labels/egomotion/` per-clip parquet（`timestamp` µs + 四元数 + xyz，~100 Hz）按
**t0=5.1 s**（官方口径，见 `gate4_forward.py`）取 16 点历史 + 64 点未来（均 10 Hz），变换到 t0 的 ego 系
`p_ego = R0ᵀ(p−p0)`。🔴 **关键修正**：`p0` 必须在 t0 **精确插值**而非取最近采样点——否则快车（18 m/s）
会因 ~3 ms 采样偏移产生 **5.7 cm 残差**（断言 `|he[-1]|<0.05` 当场抓到）；修正后残差 **0.00000 m**。

**帧抽取**：mp4 是 1920×1080 / 30 fps / hevc / 20.13 s。用随附 `<clip>.<cam>.timestamps.parquet`
（604 帧权威时间戳）定位 t0−0.3/−0.2/−0.1/0 的最近帧 = **索引 144/147/150/153**（相机与 egomotion
共享 epoch，恒定偏移 −10.85 ms）。相机顺序按 `camera_indices [0,1,2,6]`：cross_left → front_wide →
cross_right → front_tele（camera-major）。goat 无 ffmpeg ⇒ 另建 `frames_venv` 装 `imageio-ffmpeg`
（自带 aarch64 静态 ffmpeg v7.0.2），**不碰 export_venv**（它连 pip 都没有）。

**结果**（官方 `compute_minade.py --num_traj_samples 6`）：

| 指标 | 我们（n=18） | 官方参考 | 差异 |
|---|---|---|---|
| **minADE6@6.4s** | **0.80401 m** | **0.82113 m**（chefu PyTorch, 644 clips） | **−2.1%** |
| minADE6@3s | 0.17460 m | 0.254 m（paper Table 6, AR1 **0.5B**+Route） | −31%（**不同模型**，仅参考） |

@6.4s：mean 0.80401 / median 0.60216 / sd 0.6458 / **95% CI [0.5057, 1.1024]** / min 0.0949 / max 2.5043。
**官方基线 0.82113 落在我们 95% CI 内** ⇒ 统计上一致。分层（@6.4s）：gentle(n=3) 0.3617 /
straight(n=9) 0.7316 / **turn(n=6) 1.1338**；fast(n=9,v0≈15.5) 0.8976 / mid(n=5) 0.8495 / slow(n=4) 0.5366。

**物理合理性交叉验证**（「数字可信」的关键旁证）：ADE 与速度/距离**正相关**（corr(v0,@6.4s)=**+0.43**、
corr(fwd,@6.4s)=**+0.61**、corr(maxlat,@6.4s)=+0.27）⇒ 越快越远绝对误差越大，符合运动学预期、不是随机数；
**相对误差中位数仅 1.91%**（ADE@6.4s ÷ 纵向位移）；最差 3 条全是高速/大位移（`d1a20a67` v0=26.6 /
fwd=167.2 m → 2.504 m），最好 3 条全是近静止（→ 0.095 m）。

### M7. 尚未验证 / 诚实边界

- ⚠️ **mRoPE 黄旗（未解决，但间接证据已大幅加强）**：`action/config.json` 写 `rope_type: mrope`、
  `mrope_section: [24,20,20]`、`mrope_interleaved: true`，而分解补丁实现的是 **rotate_half** 约定。
  判断大概率无碍的依据：① `rope_onnx` 接收的是**已算好的 cos/sin**，mRoPE 分段逻辑在上游；
  ② 导出 dummy 的 `attention_pos_id = arange(n_diff)` 是 **1D** 位置，而 mRoPE 需 (t,h,w) 三维
  ⇒ 疑为从 VLM 继承的残留字段。**间接证据（M6 大幅加强）**：18 个不同场景的轨迹 minADE 与官方基线
  一致、ADE 与速度/距离正相关、相对误差中位数仅 1.91%——**若 RoPE 约定错，输出会是垃圾而不可能对上 GT**。
  但**仍无直接对照**（对拍只是「我的分解 vs rotate_half 公式」的自洽检查，不是「rotate_half vs TRT
  `RotaryEmbedding` 真实语义」的检查）。
- ⬜ **官方 644-clip 同口径评测做不到**：那 644 条是 NVIDIA 内部集，公开数据集无此清单。已在
  `alpamayo_aligned` test split 分层抽 18 clips 做了量级一致的评测（M6），但 **n=18 太小**
  （95% CI 半宽 0.30 m）⇒ 只能说「与基线一致」，**不能说「复现了 0.82113」更不能说「优于基线」**；
  −2.1% 的点估计差异远小于 CI 半宽，无统计意义。扩到更大样本仍受数据可得性限制。
- ⬜ **图像轴的 conditioning 敏感性未单独定量测**：M5 的 seed 对照只用了 gate4 单一场景。不过 M6 的
  minADE 跨 **18 个不同场景**且与基线一致，已**间接证明图像 conditioning 有效**（否则不同场景会给出
  同样的错误轨迹、ADE 会爆）。「固定 seed、只换图像」的定量敏感性对照仍未做。
- ⚠️ **两个输入坑**（已入 `eval/r1/README.md` §五）：① `alpamayo_gate4/meta.npz` 的 ego 历史是
  **退化占位数据**（`ego_history_xyz` 全零、`ego_history_rot` 全单位阵，闸门 4 只验加载不验数据），
  直接喂会得 v0=0、积分出「静止车刹车到倒车」的荒谬轨迹，须改用官方指南真实轨迹；
  ② 导出必须设 `HF_ENDPOINT=https://hf-mirror.com`（export 要取 Qwen3-VL config，goat 直连 HF 不通，
  漏设会卡网络重试且 RSS 停在 767 MB 不动）。另 `hf` CLI 只在 `export_venv/bin/`，不在系统 PATH。
- ⚠️ 输入图像来自 **gated + license:other** 数据集，**只在机上跑，不入库、不上站**。

**教训**：① **旧闸门结论要随工具版本复核**——闸门 4 的「不支持」是 0.9.0 时代的事实，0.10.1 已原生支持；
把版本绑定的否定结论当成永久结论，会白白关掉一条主线（本次差点因此错过 AV 轨迹路线）。
② **复用补丁前先核对调用形态，别只看算子名相同**——三个陷阱全是「同名算子、不同张量约定」，
且都被 `B=1` 的导出 dummy 掩盖。③ **对拍要带「反例护栏」**：不只验「我的实现 == 参考」，
还要验「错误实现 != 参考」，否则测不出自己究竟覆盖了没有。④ 长构建/导出**先挂内存哨兵再发射**
（只观察不自动 kill，见 [[observe-before-stopping-jobs]]）：本次探针顺带把导出跑完了，
峰值 10.5 G 远低于预估，避免了在共享机上盲目发射的风险。

复现：`scripts/orin/r1_action_minverify.sh`（下载→补丁→对拍→只导 action→算子普查→只建 action 引擎，
各步判据即上表实测数字）。
