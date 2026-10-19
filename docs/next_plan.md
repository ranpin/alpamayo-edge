# 后续实验规划（2026-07-29 重排）

> 这份规划**推翻了此前"继续往技术深处挖"的排法**。理由写在最后一节——那是本次重排最该记住的部分。

## 0. 现状：性能轴已封闭，短板全在"证据的域相关性"

**已完成且已实测的（16 项，全部上线）**

| 轴 | 结论 |
|---|---|
| 性能 | 三精度 TTFT/TPOT/TPS、功耗/能效、batch 与上下文 sweep；**31 分钟持续负载零降频**（TPOT std 0.0167 ms） |
| | **精度选择随负载形状翻转**：`输出 token < 输入 token / 83` 时 INT8 更快，n=210 实测 INT8 快 **1.43×** |
| | **AV 延迟预算**：真实分辨率下 **0.3 Hz**，vs AV 环路要 10–30 Hz |
| | maxInputLen 4096 容量重建 + **opt 位移代价**三重单变量对照归因（KV 0% / batch 6% / opt 9%） |
| 精度 | n=210 官方基准 + Wilson CI + McNemar + **任务匹配二分类指标**（发现量化把决策阈值推向"否"）+ **MDE 6.7 pts** |
| | 12-prompt perplexity（**两侧预算配平后原结论成立**）；校准集域消融（假设被推翻） |
| 机理 | Roofline、Nsight kernel 剖析、手写 GEMV 微基准（负结果，实测无余量） |
| 结构 | ViT 帧数扫描、2B vs 8B Pareto、**lm_head INT4 实测 +19.6%**（事前预测 +19%，字节预测误差 −0.23%） |
| 基线 | FP16 跨设备（goat）+ 跨设备复现 <3%、sm_87 量化选项空间枚举 |

**量化侧的真实状态**：8B decode 有效带宽 **153.1 GB/s ≈ 实测可达上限 152.3** ⇒ 这条路在
"每 token 字节不变"的前提下已饱和。已证死：W4A8（工具无实现）、手写 kernel（实测 97–100%）。

**最大的单点弱项**：**全部精度证据来自具身机器人操作**（"机器人有没有打开抽屉"），
而项目标题是自动驾驶。这是任何懂行的人的第一问，而目前没有答案。

---

## P0（唯一能改变项目性质的动作）：拿到驾驶域数据

**为什么它排第一**：它同时解锁三件目前全部卡住的事，而且是唯一能修掉上面那个弱项的动作。

- ① 驾驶域精度评测（修掉"标题是 AV、证据是机器人"）
- ② 真实分辨率研究（nuScenes 原生 1600×900 = 1450 token/帧，正好落在 4096 token 装 2 帧）
- ③ Alpamayo VLA 轨道的 ADE/FDE（本项目部署的正是它的 VLM backbone）

**入口已改（2026-07-29 闸门 1 实测结论）：不用 nuScenes，用 NVIDIA 自己的 PhysicalAI-AV。**

用户提醒"Alpamayo 本身好像有数据集"——**对，而且项目里早就识别过、只是从没接上**：
`data/prepare_physicalai_av.py` 是一个 `NotImplementedError` 的空脚手架，注释里写明数据集是
**NVIDIA PhysicalAI-Autonomous-Vehicles**（gated、133 TB，需用官方 `physical_ai_av` devkit 拉子集），
且**轨迹规格与 Alpamayo 对齐：6.4 s 视界、64 路点 @ 10 Hz**。`data/av_subset/` 目前只有
`synthetic_test.jsonl` 合成占位。

### 闸门 1（数据可得性）—— ✅ 通过，实测记录

| 数据集 | gated | 我们的 token | 内容 |
|---|---|---|---|
| `nvidia/PhysicalAI-Autonomous-Vehicles` | **auto**（接受许可即自动放行，非人工审批） | **HTTP 200，已有访问权** | 70,775 文件：51,891 `.zip`（传感器）+ 18,880 `.parquet`（标注/标定，**分 chunk ⇒ 可只拉子集**） |
| `dgural/PhysicalAI-Autonomous-Vehicles-Sample` | **False（公开）** | 200 | 706 文件、**700 个 mp4**，命名含 `camera_front_wide_120fov` / `camera_cross_left_120fov` / `camera_front_tele_30fov` ⇒ **正是我们要的多相机行车视频** |

**为什么它比 nuScenes 更对路**（三条都是实质性的，不只是"顺手"）：
1. **同一厂商**：数据集与 Alpamayo / Cosmos 出自 NVIDIA 同一体系 ⇒ 相机 rig 与坐标系无需猜测适配；
2. **轨迹规格已对齐**：6.4 s / 64 路点 / 10 Hz 正是 Alpamayo 的输出格式 ⇒ ADE/FDE 可直接算，
   不必像 nuScenes 那样先写一层格式转换；
3. **两级入口**：公开 sample 可立刻迭代（免 gate、体积小），gated 全集已可访问且分 chunk ⇒
   ego 轨迹标注按需拉，**不必碰 133 TB**。

**⇒ nuScenes-mini 降为备选**（仅在 PhysicalAI 的 ego 轨迹字段接不通时启用）。

### 闸门 2（能否喂进现有预处理链）—— ✅ **通过（2026-07-29 实测）**

具体动作（**先本地、几十 MB 级**，别一上来拉全集）：
1. 从公开 sample 拉 **1 个 clip 的 `camera_front_wide_120fov.mp4`**，`ffprobe` 读原生分辨率与帧率
   —— 这决定分辨率 Pareto 能扫到哪一档（若原生 ≥1080p，则 §1.10 里 2040 tok/帧 那档终于有真实数据）；
2. 用 §0.0.1 那条链（均匀采 6 帧 → 长边缩放 → 32 对齐）产出 token 数，核对是否符合预期；
3. 送进已建好的 `engines/int4_len4096` 跑一次，确认 4096 容量在真实分辨率下确实够用。

**实测结果（产物 `eval/vision/av_frames/`，含 6 帧 + README）**：

clip `01d3588e-….camera_front_wide_120fov.mp4` = **1920×1080 @ 30 fps, HEVC, 20.17 s / 605 帧, 13 MB**。
**原生就是 1080p** ⇒ 分辨率 Pareto 可以一路扫到原生，不必外推。

| 项 | 实测 | 此前解析预测 |
|---|---|---|
| 原生对齐（32 整数倍） | 1920×**1088** | 1920×1088 ✓ |
| **token/帧** | **2040** | **2040** ✓ |
| 6 帧合计 | 12,240 | 12,240 ✓ |
| 超 `maxInputLen=4096` | **3.0 倍** | "超 3 倍" ✓ |
| `*_len4096` 引擎实装 | **1 帧**原生 1080p | 1 帧 ✓ |
| 缩到长边 448（现口径） | 448×252 → 448×256 → **112 token/帧** | 112 ✓（与 n=210 里 1020 段一致） |

**原生 / 缩放 token 比 = 18.2×** —— "降采样折扣"的确切数字。
⇒ **站点 §1.10 那两行（1 帧原生 1080p、6 相机 1080p 装不下）从解析外推升级为实测支撑。**

**"边下边处理再删"已验证可行**：本地峰值仅 13 MB，处理完删 mp4、只留 6 帧共 1.25 MB；
批量 10 个 clip 也只是 130 MB 峰值 / 12 MB 留存。命令见 `eval/vision/av_frames/README.md`。

**设备侧结论（实测，修正此前设想）**：**scm 不能当下载盘** —— `huggingface.co` TCP 443 不可达
（而 `pypi.org` 通 ⇒ 白名单式受限出网），且**没有 ffmpeg/ffprobe**。但它有 **18 TB 空闲**
⇒ 定位为**归档盘**：本地即下即删产出的帧与标注（体积小得多）往那存长期副本。

### 闸门 3（能否构造带标准答案的评测）—— **真正的分水岭，尚未做**

公开 sample 只有 mp4（700 个），**没有 QA 标注**。所以要么：
- (a) 从 gated 全集的 `.parquet` 里找到可判定的标注（如目标框/红绿灯/可行驶区域），据此**自动生成**
  yes/no 或多选题 + ground truth —— 这是最可信的路，但需先摸清 parquet 的 schema；
- (b) 手工标一小批（n≈100），成本可控但样本量小、且我自己标会引入偏差，须声明；
- (c) 都不行 ⇒ **退化为定性 demo，明确标注"无标注、不作为精度证据"**，并把"驾驶域定量评测缺失"
  继续留在诚实边界里。

### 闸门 4（Alpamayo 权重能否在 20 GB 内加载）—— 尚未做，与闸门 1–3 独立


**半天侦察，按"最可能致命者优先"设闸门 —— 任一条不过就停，把受阻点写成结论而非 future work**

| # | 闸门 | 通过判据 | 不通过的处置 |
|---|---|---|---|
| 1 | 数据可得性 | 能下载、能解出图像 + ego 轨迹标注 | 换 Waymo Open perception mini / 自采公开行车记录 |
| 2 | 喂得进现有预处理链 | 用 §0.0.1 那条链（长边缩放 → 32 对齐 → 6 帧）产出 token 数符合预期 | 若相机内外参耦合太深 ⇒ 只做单相机，不追多相机 |
| 3 | **能否构造带标准答案的评测** | 至少能自建一批**可判定**的问答（如"前方是否有行人"、"能否安全右转"），有 ground truth | **这一条是真正的分水岭**：不过则退化为定性 demo，明确标注"无标注、不作为精度证据" |
| 4 | Alpamayo 权重能否在 20 GB 内加载并跑通一次前向 | 在 goat（64 GB）上加载成功 + 单次前向 | 不过则 VLA 轨道判死，写成"车规 VLA 的输入耦合度是边缘复现的现实障碍（含具体受阻点）" |

> **闸门 4 已执行（2026-10-12）：判据 ✅ 通过 / "20GB 内" ❌ 不可能。**
> goat 上 22.16 GB bf16 权重加载 5.7s（mmap）、CPU 单次前向 3505s 完成，输出 64 路点 @10Hz
> 轨迹 + 合理 CoT（"Keep lane"，静止 ego 历史 ⇒ 平缓起步直线，物理自洽）。但 FP16-only
> 权重 22.16G > 20G、官方单样本推理 ~24GB VRAM、无量化路径 ⇒ **32GB 级边缘部署不可行**。
> VLA 轨道（ADE/FDE）不判死但受阻于三点：20GB 预算、CUDA torch 无安装源（白名单出网）、
> CPU 前向 58 分钟/次（100 clips ≈ 4 天）。全部实测与离线适配记录：`eval/vla_gate4/README.md`、
> PROBLEMS J3。

**闸门 1–3 全过后才做的正式实验**
1. ✅ **已完成（2026-10-09）**：驾驶域精度评测，n=184（闸门 3 评测集），三精度同帧对比——
   **FP16 51.6% / INT8 47.8%（−3.8 pts，p=0.143）/ INT4 50.5%（−1.1 pts，p=0.815），均不显著**
   ⇒ 「量化无显著正确率损失」迁移到驾驶域成立，P0 的核心缺口（"标题是 AV、证据是机器人"）已填。
   产物 `eval/accuracy/driving/`，站点 §1.11（图 20 / 表 11–12）。
2. ✅ **已完成（2026-10-10）**：分辨率 Pareto（448 / 720p / 原生 1080p × 正确率 × E2E × 能耗，
   INT4+INT8，n=184/档）——**分辨率买得到正确率，性价比拐点在 720p**：INT8 47.8→54.9（+7.1）
   →56.0%（+8.2 vs 448）。族内校正后（预注册 4 次 vs-448，Bonferroni α=0.0125 + BH-FDR）只有
   1080p-vs-448 显著（p_bonf=0.043）；720p 裸 p=0.047 校正后不显著 ⇒「拐点」是性价比论点
   （720p 拿到 +8.2 里的 +7.1、成本仅 1/3）；vs 720p 仅 +1.1（事后分析）。INT8 每档同时赢 INT4
   的速度/能耗/正确率。（2026-10-16 Y1 统计校正补 TOST + 族内校正，`eval/accuracy/stats_correction.py`）
   两个运行时边界入 PROBLEMS J1（INT4@1080p 16% 空输出）/J2（INT8@1080p KV32768 OOM，
   KV16448 已解决）。产物 `eval/accuracy/driving/pareto/`，站点 §1.12（图 21 / 表 13）。
   至此「分辨率–精度–延迟」三维决策面补齐，P0 收官。
3. 若闸门 4 也过：Alpamayo ADE/FDE vs 已有 CVM 基线（复用 `eval/action_to_traj.py` + `eval/metrics.py`）

**预期产出**：把"边缘量化部署方法论"从机器人域搬到驾驶域，并给出**分辨率–精度–延迟**三维权衡
——后者配合已有的 §1.9（精度选择交叉点）与 §1.10（延迟预算）能构成完整的部署决策面。

---

## P1（零技术风险，但决定有没有人看得懂）：包装

1. **仓库改名**：~~项目叫 Alpamayo-Edge，仓库叫 `qwen-trajectory-prediction` —— 名不对实是实打实的减分项。~~
   **✅ 已完成（2026-10-09）**：仓库已改名 `ranpin/alpamayo-edge`，站点 URL 随之变为
   `ranpin.github.io/alpamayo-edge/`，edge-ai-docs `docs.json` 与站内链接已同步更新。
   （原"拆成独立仓库"的方案被直接改名取代——整个仓库改名后 `main` 分支的旧轨迹项目仍在同仓库，
   如需彻底分离再拆。）
2. ~~**英文一页版**：英文 README + 摘要 + 四张总览图的英文标注。目标若含 NVIDIA/Qualcomm/外资 AV 团队，
   全中文是硬障碍。~~ **❌ 已取消（2026-10-15 用户决定）**。

---

## Phase V（2026-10-15 新增，当前主线）：VLA 轨迹输出 + demo——从"VLM 部署"到"VLA 全栈"

> **用户决策（2026-10-15）**：训练环节（Phase T 域适配 QLoRA）暂缓，**VLA 优先**，且以
> **演示视频 / 在线 demo 为一等交付物**。动机：QLoRA 微调不构成简历区分度，
> "32GB Orin 跑通 AV VLA 全栈（推理+轨迹）"几乎没有公开先例（唯一参照是 Thor fork，128GB 级机器）。
>
> 同日评测完备性审计：对"完整 VLA 项目"的声称缺**轨迹侧（ADE/FDE/minADE）、CoT 质量、多相机精度**
> 三块——正是 V5 的必做项；运动类任务 / 跨 VLM 对比 / 闭环评测 / J1 机理**明确 scope out**
> （有意边界而非遗漏，审计记录见当日讨论）。

### 设备分工（"为什么主打 32G 而不是 64G"）

闸门 4 判"32GB 不可行"**只针对 FP16-only 的 PyTorch 路线**（22.16G 权重 + 官方 ~24GB VRAM）。
Phase V 走 TRTEdge 量化路线，内存账完全不同：VLM INT4 ~4.6G + 视觉 1.1G + action expert
FP16 4.6G + KV@16k ~4.8G + 词嵌入 1.25G ≈ **16G ⇒ 32G 装得下**。

| 机器 | 角色 | 理由 |
|---|---|---|
| **dog（32G）** | 边缘部署目标机（INT4/INT8 量化 VLA 全栈） | "edge" 是项目立身之本；"32G 跑通"是更强结论；引擎矩阵/TRTEdge 安装/506G 磁盘现成 |
| **goat（64G）** | 参考与构建机：CPU golden reference（D0 demo）、FP16 参考、构建后备 | Alpamayo 权重 22G + alpamayo1.5 repo + venv 已在；引擎与 dog 互通（同 sm_87 + TRT 10.7.0.23） |

### Demo 层级（交付物，与风险解耦）

| 层级 | 依赖 | 内容 |
|---|---|---|
| **D0（保底）** | goat CPU + 真实 ego/GT 数据（HF token） | 过夜跑 10–20 clips，渲染 mp4：前视视频 + CoT 字幕 + 预测轨迹 vs 真值 BEV 叠加 |
| D1 | V1–V2 过 | Orin 实时 CoT 推理演示（轨迹仍离线） |
| **D2（完全体）** | V3–V4 过 | Orin 端实时 VLA（视频进 → CoT+轨迹出）+ **站点静态交互 demo**（预计算结果 + clip 选择器 + BEV canvas，gh-pages 托管） |

D0 把 demo 从最高风险路径（V4 C++ 移植）解耦：即使 Orin 移植失败，仍有 VLA demo + 诚实受阻点分析。

### 闸门（预注册，任一不过即按处置收官）

| # | 闸门 | 通过判据 | 不通过的处置 |
|---|---|---|---|
| V0 | 侦察：fork 许可证可移植？v0.6.0→0.9.0 API 差异面？HF 可达 + ego/GT 数据可得？ | 可行性备忘入库 | 差异过大 ⇒ D0 收官 + 受阻点报告 |
| V1 | Alpamayo VLM（traj token 扩展词表）过 Kaggle ONNX 导出 | 导出成功、词表完整 | 定位 embedding/lm_head resize |
| V2 | INT4/INT8 VLM 引擎建成 + CoT 与 gate4 CPU golden reference 对齐 | 同输入同 CoT | 量化对 traj token 影响分析 |
| V3 | action expert（2.3B）FM one-step ONNX → sm_87 引擎 | 引擎建成（FP16 4.6G） | 回退 FP16-only expert |
| V4 | FM runtime C++ 移植到 TRTEdge 0.9.0 + 单样本轨迹与 CPU 参考一致 | 容差内一致 | **D0/D1 收官**，V4 写成受阻点 |
| V5 | 收官评测 + demo：N clips ADE/FDE/minADE vs CVM + CoT 评审（n=30）+ 多相机精度 + D2 视频 + 站点交互 demo | 指标表 + 可播放 | — |

参照物：`kimsunguk0/tensorrt-edge-llm-for-alpamayo`（TRTEdge v0.6.0 + Alpamayo 1.5 轨迹栈，
C++ FM/action runtime、FM one-step ONNX 导出、live 推理，目标 Jetson Thor）；我们是 v0.9.0（dog commit 1ac0f2b）。
穿插项：帧采样敏感性（原 P2，纯设备时间）在 V2 等待期跑。

---

## Phase W（2026-10-15 新增，**执行顺序在 Phase V 之前**）：Cosmos3-Edge——官方支持的第二部署目标

> **用户决策（2026-10-15）**：项目升级为**双模型单方法论**——Track 1 = Alpamayo 1.5（Phase V，
> AV 专属叙事），Track 2 = Cosmos3-Edge（本节，官方支持叙事）。简历叙事从"部署了一个模型"升级为
> "同一套边缘方法论部署两代 NVIDIA Physical AI 模型（10B AV VLA + 4B 世界动作模型）"。
>
> **事实纠偏（调研结论，勿再混淆）**：Cosmos3-Edge **不是** Alpamayo 2 的 backbone（那是 32B
> Cosmos 3 Super Reasoner，且 2B 轨迹专家与 32B 联合训练、不可换底座拼接）；Cosmos3-Edge 是
> Cosmos 3 家族的**边缘款完整模型**（4B = 2B VL reasoner + rectified-flow 扩散专家，MoT 单塔），
> 与 Alpamayo 2 同代际。Alpamayo 2 Super（34B，gated）本身维持"展望 + 三重估触发条件"不变
> （见 2026-10-15 调研：量化算力墙/TRTEdge 架构支持未知/gated/68G 超 goat）。

**为什么 W 先行**：① turnkey——TRTEdge **v0.10.0 官方支持**（supported-models 在列：
Cosmos3-Edge + Cosmos3-Edge-Policy-DROID），`tensorrt-edgellm-export --task policy`（CPU 导出）→
`cosmos3_policy_build` → `cosmos3_policy_inference`（图+prompt→action.json）全链官方命令；
② demo 天级 vs Phase V 月级且 ~40% 风险；③ dog 需自建 TRTEdge 0.10.x（共享安装不可动），
该 build 同时是 Phase V fork 移植的目标版本；④ 4B 内存宽裕（bf16 ~8G/INT4 ~3G vs 32G dog）。

**关键约束（0.10.0 支持矩阵实测核对）**：Orin JetPack 6.2/CUDA 12.6 = **Compatible 档**
（Official 是 JetPack 7.2）；精度仅 **FP16/INT8/INT4**（NVIDIA 官方 mxfp8/nvfp4 是 Blackwell
专属，sm_87 不可用）；**0.11.0 矩阵已删 JetPack 6.2 行 ⇒ 版本锁 0.10.x，不追新**。
社区先例：A10G int8wo <10GB（linjiw，含质量评测）、TRTEdge INT4-AWQ（ubrobotics）、
TRTEdge webcam 实时 demo <10GB（Derman；注明 NVIDIA 原始 demo 跑在 Jetson Orin）。

**权重**：非 gated，`HF_ENDPOINT=https://hf-mirror.com` 直下（HF 主站自 2026-10-15 晨不可达）；
Cosmos3-Edge 9.18G / Policy-DROID 9.17G（transformer 6.75G + vae 1.41G + vision_encoder 0.98G）。

### 闸门

| # | 闸门 | 通过判据 | 不通过的处置 |
|---|---|---|---|
| W0a | 权重落地 + 架构核实 | ✅ 2026-10-16：hf-mirror 下载 Cosmos3-Edge 8.6G + Policy-DROID 8.6G，字节校验，Mac+goat 双落地 | — |
| W0b | dog 自有目录建 TRTEdge **0.10.x**（JetPack 6.2 Compatible 档）+ `-DBUILD_EXPERIMENTAL_MODELS=ON` | ✅ v0.10.1 在 dog+goat 均建成（六道墙见 PROBLEMS K）；共享安装未动 | — |
| W0c | `tensorrt-edgellm-export --task policy` 走通（CPU 导出，地点 Mac/goat/Kaggle 三选一实测） | ✅ reasoning + policy 导出均成功（goat CPU）；policy 的 und_prefill/gen 走 K6 分解补丁（`USE_DECOMPOSED_POLICY=1`） | — |
| W0d | AV embodiment 路径核实 | ✅ 动作三模式 fd/id/policy；AV 轨迹走 id 模式（PyTorch 路径）；TRTEdge turnkey = reasoning + DROID policy；Guardrail gated 为风险项 | — |
| W1 | 引擎构建：FP16 → INT4/INT8 | 🔄 **FP16 已成（goat）：llm 3.7G + visual 940M（分解注意力版），单样本冒烟通过（驾驶场景描述连贯/EOS）**；INT4/INT8 量化未开始；dog 侧被 system_agent 占 26.5G GPU 内存阻塞（待协调） | 量化走 TRTEdge quantize.py 或 Kaggle 管线 |
| W2 | 评测：reasoner 侧复用 **n=184 驾驶域基准**（2B reasoner vs 8B CR2 跨代对比）；policy 侧 DROID 动作链打通 | ✅ reasoner n=184 = **42.4%**（−9.2 vs CR2 FP16，p=0.0115 显著）；**policy 四组件引擎全建成（K6 分解补丁绕过 TRT10.7），冒烟 exit=0、action[1,16,10] 全 finite、5154ms/round**；动作语义定量评测留 W3 | — |
| W3 | demo：动作推理视频（真实推理输出）+ 站点交互页 | 🔄 **侦察完成（2026-10-19）**：policy 动作链部署打通，但单帧 turnkey 路径**帧无关**（跨帧复测 f20/60/100/140 输出 mean diff≈0.009，vision-only 单帧 conditioning 能力边界、非引擎 bug，见 PROBLEMS §L）⇒ **demo 改锚定 reasoning 侧**（已验证 works：真实驾驶帧→Orin→场景理解描述），policy 动作链定位为「部署打通 + 帧无关边界」 | reasoning demo 待做 |

**与 homebot-wm 的边界**：Policy-DROID 是机器人操作 WAM，与 homebot-wm 主线（压缩+部署视觉
扩散模型到 Orin）同构——**方法共享、叙事分开**：本项目讲"AV/边缘方法论双模型实证"，
homebot-wm 讲"机器人世界模型"，不重复计功。

---

## P2（零成本、加固而非开拓，可穿插）

- **帧采样敏感性**：同 210 题换 3 组不同采样帧重跑，回答"6 帧协议是不是结论的来源"。
  纯设备时间，无新依赖。
- **token 预算统一化消融**：按面积缩放使 tokens/帧恒定（当前是长边固定 ⇒ 112 vs 154 不一致），
  测正确率是否受影响。顺带消除延迟不确定性。

---

## 明确不做（含收回我此前的建议）

| 项 | 处置与理由 |
|---|---|
| **ViT INT8** | **停**（PROBLEMS I1 已结案）。9 次尝试全卡在模型加载，根因定位到"PATCH A 分片代码从未执行"；重启只需一行 `python -c 'import tensorrt_edgellm as m; print(m.__file__)'`。**但即使成功也只给端到端 10–15%，不改变 0.3 Hz 的部署结论。** |
| vocab reduction / 投机解码 | 降到最低。工具虽原生支持，但**性能轴已不是瓶颈**，说服力才是。 |
| 扩样本到 n=510 | **不做**。功效计算：INT4 需 n≈1650、INT8 需 n≈10700 ⇒ 扩到 510 不会翻案。 |
| 拆 lm_head 的 calib-batch confound | 不做。−1.4 pts 远低于 MDE 6.7 pts，拆成两半仍然都测不出来 —— **在测不准的量上做归因是浪费**。 |
| W4A8 / 手写 kernel / KV cache 量化 | 已证死或已证无余量（METHODOLOGY §3.6）。 |
| 高通 8397 | 用户已明确搁置。若重启，价值在"同模型 × 两家 NPU 的跨厂商对照"，先做 SDK 可得性侦察（最大风险）。 |

---

## 本次重排最该记住的一条

前一轮我把优先级排成了"继续往技术深处挖"（ViT → vocab reduction → 投机解码），结果 ViT 一条
花掉 **9 次尝试、十几小时云端时间**，产出是一条**已定位但不改变任何结论**的工具链故障。

错在于我问的是**"技术上还能不能再深一点"**，而不是**"这个项目现在最缺什么"**。

- 性能轴：已到硬件可达上限，且 §1.10 证明再快 15% 也够不上 AV 环路 ⇒ **深挖的边际价值≈0**
- 证据轴：全部来自机器人域，与项目标题不符 ⇒ **这才是唯一的实质缺口**

⇒ **判优先级要看"哪条能改变项目的性质/结论"，不是"哪条技术上更硬"。**
配套的执行纪律：**每条实验预先写下闸门与不通过时的处置**（本文 P0 那张表就是这么写的），
避免再出现"连续 9 次凭猜测修同一个失败"。
