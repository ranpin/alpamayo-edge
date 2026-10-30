# 跨代对比：Alpamayo-R1 vs Alpamayo-1.5（同口径 minADE）+ 导航引导检验

> **状态（2026-10-30）**：1.5 经 **repack 管线**（纯 config 改写、不动权重）**首次跑上 Orin**，
> 并与 R1 完成**同口径 minADE 对比**。四条结论：
> 1. **repack 管线跨代可用**——1.5 无需任何权重改动即可走 R1 的 export/build/推理全链；
> 2. 同口径下 **R1 胜 1.5**（minADE6@6.4s **0.804 vs 1.815 m**，1.5 差 +126%）——反直觉（新一代更差）；
> 3. **「1.5 需要导航引导」假设被两个独立测量否定**（oracle nav 仅 −1%、within-noise）；
> 4. 真正病灶是 **1.5 高方差、个别 clip 崩坏**，R1 全程稳定。
>
> 基准与 R1 基线见 [`README.md`](README.md) §五：**同 18 clips × 6 seeds、同 tokenizer、同 chat template、
> 同积分口径，唯一变量 = 权重**。⚠️ 帧与 GT 同 §五：gated + license:other 数据集，只在机上跑，
> 逐 clip 原始数据不入库（本文 clip id 仅作定位）。

## 一、动机

R1（= 「Alpamayo 1」，CES 2026 改名）与 1.5 同为 10B、**同架构**（1159 张量同名同形同分片），
1.5 是 R1 的 RL 后训练版，model card 声称新增**导航引导 / 可变相机数 / 用户问答**。
要回答的问题：**在同一 AV 轨迹基准上新一代 1.5 是否更强？若跑输，原因是什么？**

## 二、1.5 怎么跑起来的（repack 管线）

TRTEdge 0.10.1 的 export 只认 `model_type=alpamayo_r1`（`export.py` 的 frozenset 门），
1.5 的 `model_type=alpamayo1_5` 会被拒。**静态侦察**证明两代可共用同一套 R1 管线：

- 1159 张量同名、同形、分片落位全同；R1 仅多 7 个派生 buffer（3×`freqs` 由 modeling init 用
  `logspace(0,log10(100),10)` 合成并 `register_buffer`、4×`action_space.*`=fp32 config 常数），
  且这组常数两代 config 逐位相同。
- TRTEdge 0.10.1 加载器**非 strict**（缺失键 debug-skip），故无需注入这 7 个 buffer。

⇒ **repack = 纯 config.json 改写**（`scripts/orin/repack_alpamayo15_to_r1.py`，幂等、留 `.orig` 备份）：
`model_type→alpamayo_r1`、`architectures→[AlpamayoR1]`、`_target_` 前缀 `alpamayo1_5.→alpamayo_r1.`、
`expert_cfg.dtype→bfloat16` 对齐 schema。**不动一个权重字节**。这让跨代对比高度受控：
同 tokenizer、同 chat template、同 export/build 路径，**唯一变量 = 权重（RL 后训练）**。

### 🔴 tokenizer 陷阱（实测，必须处理）

若 config 的 `vlm_name_or_path` 指向本地 processor 目录（gate4 的 `alpamayo_gate4/processor`），
导出进引擎的 chat template 不认识 `{"type":"image"}` 内容——该 processor 的 tokenizer
（len 仅 151669）**既无 `image_pad/vision_start/vision_end`，也无任何 Alpamayo special token**，
推理时全部请求报 `EDGELLM_BAD_MEDIA_COUNT`（"pad count is smaller than this request's media count"）。

**修复** = 引擎建成后把 **R1 引擎 `llm/` 的 `tokenizer.json` / `tokenizer_config.json` /
`processed_chat_template.json` 三件拷入 1.5 引擎 `llm/` 覆盖**（两代同词表 151936、同架构，
R1 的即 Qwen3-VL-8B-Instruct + Alpamayo added_tokens，md5 验证逐文件一致）。
这一覆盖同时让跨代对比更受控——两代用**完全相同**的 tokenizer/chat template。
（替代法：导出前把 `vlm_name_or_path` 改成 `Qwen/Qwen3-VL-8B-Instruct` 走 HF 抓取。）

## 三、跨代 minADE 对比（R1 胜）

同 §五 口径：chunk 3116 分层 18 clips × 6 seeds，`compute_minade.py --num_traj_samples 6`，
两代用**同一份请求 JSON / tokenizer / chat template**，唯一变量 = 权重。

| 指标 | R1 | 1.5 | 差（1.5−R1） |
|---|---|---|---|
| mean minADE6@6.4s | **0.8040 m** | **1.8153 m** | **+1.0112 m（+125.8%）** |
| mean minADE6@3s | 0.1746 m | 0.3540 m | +0.1794 m |
| median @6.4s | 0.6022 m | 1.3203 m | +0.7181 m |
| 逐 clip R1 更优 | **13/18** | 5/18 | — |

**反直觉**：新一代（1.5，RL 后训练）在同一轨迹基准上显著输给上一代 R1。下面检验最自然的解释
（「1.5 需要导航引导」）是否成立。

## 四、导航引导假设的检验（否定）

**假设**：1.5 跑输是因为它**需要导航引导输入**（model card 新增能力）而本基准没给；喂导航应能拉回。

### 4.1 技术可行性——全打通

- 导航是**文本**。官方 `helper.create_message` 把它包成 `<|route_start|>{nav_text}<|route_end|>`，
  插在 traj_history 占位与 prompt 之间：`user_text = {hist_traj_placeholder}{route_section}{prompt_text}`。
  （`use_nav_prompt` 是死参数，函数体不消费。）
- TRTEdge 请求里 trajectory 是独立 content 项（runtime 自转 48 个 `<|traj_history|>` token），
  故**把 text 项前缀加上 route 段即与官方逐位一致**——用 response 的 `formatted_complete_request`
  实证摆放正确：`…<|traj_history_end|><|route_start|>Turn right in 15m<|route_end|>output the chain…`。
- R1 tokenizer 的 added_tokens 里**就有** `<|route_start|>`(155690)/`<|route_pad|>`(155691)/
  `<|route_end|>`(155692)，切成**单个** special token（非乱码）；同块的 cot/traj token 能产出连贯
  CoT + 有效轨迹 ⇒ 这些 id 对 1.5 正确，route 同块故也正确（排除 id 错位）。

### 4.2 语义判据（单 clip，负面）

右转 clip `6b7467e8`（实测总转角 −65°、15 m 处起弯）上，同 seed 三变体：
无 nav / oracle「Turn right in 15m」/ counterfactual「Turn left in 15m」——
**三条 CoT 逐字相同**（"Adjust speed for the right curve ahead"），反向导航都没让模型改口，
轨迹仅微扰、无方向性改变 ⇒ 模型跟着视觉上下文走，**不语义响应导航**。

### 4.3 量化复测（18 clips × 6 seeds，oracle nav，仍否定）

我的 18 clip **无导航标注**（gt 只有 `gt_xy/ego_history_xyz/ego_history_rot`），官方 nav_demo 是另一批 clip。
⇒ 从 GT 未来轨迹**反推 oracle nav**（上界测试）：按航向变化量分 **6 右转**（"Turn right in 5/15/40m"）
+ **12 直行**（"Continue straight"），脚本 `scripts/orin/derive_a15_oracle_nav.py`。

| 口径 | 1.5+oracle nav | 1.5 无nav | R1 | nav 增量 |
|---|---|---|---|---|
| mean @6.4s | 1.7969 | 1.8153 | 0.8040 | **−0.018（−1.0%）** |
| mean @3s | 0.3535 | 0.3540 | 0.1746 | −0.0005 |
| 右转(6) @6.4s | 1.5792 | 1.6366 | 1.1338 | −0.057（−3.5%） |
| 直行(12) @6.4s | 1.9057 | 1.9046 | 0.6391 | +0.001 |

逐 clip nav 更优 **9/18**（掷硬币）。**oracle nav（上界）只带来 −1% 的 within-noise 改善，
1.5+nav 仍是 R1 的 2.2×** ⇒ 导航引导**不是** 1.5 跑输的原因。
4.2（语义）与 4.3（量化）两个独立测量一致，按复测纪律定论。

## 五、病灶：1.5 高方差崩坏

逐 clip 揭示 1.5 的均值被几个灾难性 clip 拉爆，nav 对它们几乎无效：

| clip | R1 | 1.5无nav | 1.5+nav |
|---|---|---|---|
| 174407b0 | 0.51 | **5.78** | 5.72 |
| 82cc0eec | 0.54 | **4.91** | 4.93 |
| 8496830f | 0.43 | 2.02 | 2.06 |

即 **1.5 是高方差、个别 clip 崩，R1 全程稳定（0.10–2.50 m）**——不是系统性缺导航。

## 六、诚实边界

1. **n=18 太小**（同 §五），95% CI 宽；跨代差 +1.01 m 远超 CI 半宽，**方向可信、量级不精**。
2. **oracle nav 是上界**（GT 反推、泄漏转向意图）——连上界都只 −1%，真实导航更不会翻盘。
3. **经 TRTEdge repackaged 路径**：nav 格式 / token id 已逐位验证正确，官方 PyTorch 路径翻盘可能性低，
   但**未直接对照**。
4. **1.5 崩坏 clip 的机理未深挖**：是 RL 后训练的真弱点还是 repack 副作用，未逐例归因。
5. 帧/GT 同 §五：gated 数据集，只在机上，逐 clip 原始数据不入库。

## 七、文件与复现

| 文件 | 说明 |
|---|---|
| `scripts/orin/repack_alpamayo15_to_r1.py` | repack（纯 config 改写，幂等，含 tokenizer 陷阱说明） |
| `scripts/orin/derive_a15_oracle_nav.py` | 从 GT 反推 oracle nav + 建 nav 请求 `r1_eval_in18_nav.json` |
| `scripts/orin/a15_nav_eval.sh` | 1.5（含 nav 变体）6-seed 推理 + merge + compute_minade |

机上产物（goat `eval_data/` 与 `io/`，**不入库**）：`minade_chunk3116_a15.csv`、
`minade_chunk3116_a15_nav.csv`、`a15_eval_s{0..5}.json`、`a15_nav_s{0..5}.json`、
`a15_nav_output.json`、`r1_eval_in18_nav.json`。本文 §三/§四/§五 的聚合数字即其全部对外呈现。
