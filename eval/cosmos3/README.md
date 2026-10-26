# Cosmos3-Edge 评测（Phase W · W2）

第二部署目标：`nvidia/Cosmos3-Edge`（4B 全模态世界模型 = 2B VL reasoner + rectified-flow
扩散专家，MoT 单塔）。部署链路与六道 JetPack 6.2 兼容墙见 `docs/PROBLEMS.md` §K，policy
动作链的能力边界见 §L。

🎬 **在线 Demo（reasoning 侧自由场景描述，真实 Orin 推理输出）**：
<https://ranpin.github.io/alpamayo-edge/cosmos3-demo/>（源在 `docs/site/cosmos3-demo/`）

## Reasoner 侧：n=184 驾驶域基准（2026-10-16，W2）

**协议与 §1.11/§1.12 完全一致**（同帧、同题、同标答、贪心、cap 12、6 帧 @448），
唯一变量是模型。输入直接复用 `bench/ood_eval/ood_input_goat.json`（messages 格式模型无关，
runtime 用引擎目录内自己的 chat template）。

| 模型 | 正确率 | Wilson 95%CI | vs CR2-8B FP16 | E2E/请求 |
|---|---|---|---|---|
| CR2-8B FP16（goat） | 51.6% | [44.5,58.7] | — | ~1.2s |
| CR2-8B INT4（dog） | 50.5% | [43.4,57.7] | −1.1（p=0.815 ns） | 0.68s |
| CR2-8B INT8（dog） | 47.8% | [40.7,55.0] | −3.8（p=0.143 ns） | 0.50s |
| **Cosmos3-Edge 2B FP16（goat）** | **42.4%** | [35.5,49.6] | **−9.2 pts（b/c=29/12，p=0.0115 显著）** | **0.91s** |

- 184/184 全 EOS、零解析失败、字母分布均衡（A17 B22 C20 D23 E29 F26 G19 H28，无坍缩）。
- 随机基线 12.5% ⇒ 2B reasoner 达 3.4× chance，有真实场景理解力。
- **结论（跨代验证项目核心发现）**：新一代 2B（Cosmos3，2026-07）在驾驶域显著输给上一代 8B
  的量化版（CR2 INT8/INT4）——"缩小模型不免费、量化几乎免费"在第二个模型家族上成立。
- 引擎：FP16，llm 3.7G + visual 940M（分解注意力版；插件版 visual 亦已建成，性能对比待做）。

## Policy 侧：DROID 动作链端到端打通（2026-10-16 W2 打通 / 2026-10-19 W3 实测边界）

Policy-DROID（`nvidia/Cosmos3-Edge-Policy-DROID`，机器人操作 WAM）四组件引擎全部建成，
`cosmos3_policy_inference` 图+prompt→action.json 冒烟通过（goat）。

| 组件 | 引擎 | 说明 |
|---|---|---|
| und_prefill | 2.7G | 理解 prefill（28 层，分解注意力版） |
| gen | 2.7G | rectified-flow 扩散动作专家（分解注意力版） |
| vae_encoder | 292M | 观测编码 |
| text_tokenizer | 17M | + embed_tokens 513M（staged） |

**第六道墙（K6）与打通**：policy 的 und_prefill/gen 把注意力/RoPE lower 成 **ONNX 标准
`Attention`/`RotaryEmbedding` 算子**，只有 TRT≥11 parser 认识；Orin TRT 10.7 报
"Cannot find plugin: Attention/RotaryEmbedding"。解法同 K4 思路、推广到 RoPE：
`scripts/orin/apply_policy_decomposed_patch.py` 在 `ops.py` 加 `decomposed_attention`
（GQA 折批 + arange 下三角 causal 掩码，覆盖 und 的 is_causal=True 自注意力与 gen 的
is_causal=False 交叉注意力）+ `decomposed_rope`（rotate_half，恒等 gather 保留
position_ids 防 DCE），env `USE_DECOMPOSED_POLICY=1` 门控。分解实现先经
`test_policy_decomposed.py` 数值对拍（attention vs SDPA err~1e-7、rope vs LLaMA 全维
公式 err=0）再导出。详见 PROBLEMS §K6。

**冒烟结果**（`policy_smoke_action.json`，官方 UMI 首帧图 + "pick up the object"，4 步扩散）：
exit=0，action shape **[1,16,10]**，160 值全 finite、范围 [−2.13, 2.77]，逐维分布有结构
（d3≈−1.93±0.15、d5≈+2.15±0.14 紧聚类，非噪声）；**延迟 5154 ms/policy round**（goat）。

> **诚实边界（W3 实测 2026-10-19 / W4 复测修正 2026-10-26）**：W2 冒烟是**部署管线验证**——分解引擎
> 端到端跑通、产出结构合法的动作；**不是动作语义正确性验证**。W3 用真实 DROID 帧实测单帧路径「帧无关」；
> **W4 复测修正**：不止帧——**帧/文本 prompt/CFG 三个 conditioning 轴一致地只有 1–2%，而 diffusion seed
> 主导输出（52%，是 prompt 效应的 ~30×）**，40 步也不收敛出来。即 turnkey 路径输出基本是「seed 去噪出的
> 通用轨迹」，conditioning 弱耦合，**根因未定**（我们的 gen cross-attention 分解补丁接线 vs 模型属性）。
> 且 Edge 输出（10 维 midtrain rot6d + quantile 归一化）与社区 lerobot GT（7 维笛卡尔）表示/归一化不同、
> 无法直接对照。⇒ demo 锚定已验证 works 的 reasoning 侧，policy 动作链定位为「部署打通 + conditioning
> 弱耦合边界（根因未定）」。详见 PROBLEMS §L（含 L4）。

## 文件

| 文件 | 说明 |
|---|---|
| `cosmos3_edge_fp16_outputs_n184.json` | reasoner n=184 原始输出（实测记录） |
| `policy_smoke_action.json` | policy 冒烟动作输出（1×16×10，端到端打通凭证） |
| 打分 | 复用 `eval/accuracy/driving/score_ood.py` 与 `../benchmark/score_mc.py` 的解析/Wilson/McNemar |
