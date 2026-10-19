# Alpamayo-Edge

**NVIDIA Physical AI 模型在 Jetson AGX Orin 上的边缘量化部署与全套实测研究**——一套管线部署
两代模型：**Alpamayo 1.5**（10B AV VLA，其 VLM backbone = Cosmos-Reason2-8B，已量化落地）+
**Cosmos3-Edge**（4B 世界动作模型，进行中）；INT4(AWQ)/INT8(SmoothQuant) 量化，精度、延迟、
能耗、内存、可靠性五轴证据，零预算、零训练、全 Orin 实测。

🎬 **在线 Demo（Cosmos3-Edge 2B 驾驶场景理解 · 真实 Orin 推理输出）**：<https://ranpin.github.io/alpamayo-edge/cosmos3-demo/>

📊 **呈现站点（论文式，图 1–21 / 表 1–13）**：<https://ranpin.github.io/alpamayo-edge/>

## 一分钟版本

| 问题 | 实测答案 |
|---|---|
| 8B 驾驶 VLM 能上 32GB Orin 吗？ | **只有量化能**：FP16 build/加载双 OOM；INT4 4.6G / INT8 8.3G 落地 |
| 量化掉多少精度？ | **基本不掉**：TOST 等价检验（δ=7pts=本研究 MDE）——INT4 两域 + INT8 机器人域成立等价；INT8 驾驶域未检出损失但不排除 ≤7.5pts（90%CI 下界 −7.5） |
| INT4 还是 INT8？ | **看负载**：长生成选 INT4（decode 快 1.61×）；看图答短答案选 INT8（prefill 快 1.43×，且驾驶域各分辨率档全胜） |
| 分辨率值不值？ | **值，性价比拐点 720p**：INT8 448→1080p +8.2pts（族内校正后显著）；720p 已拿到其中 +7.1、成本仅 1/3（720p 单独未过校正 ⇒ 性价比论点，非显著性结论） |
| 能进 AV 感知环路吗？ | **不能，也不该**：真实分辨率 0.1–1 Hz ⇒ 定位事件级场景理解（风险解释/日志挖掘/驾驶员问答） |
| 还有优化空间吗？ | **量化侧已穷尽**：decode 带宽达硬件可达上限（153.1≈152.3 GB/s）、手写 kernel 负结果、31min 零降频 |
| VLA（轨迹输出）呢？ | PyTorch 路线在 32GB 级不可行（FP16-only 22.16G）；**双轨推进中**：Track 1 = Alpamayo 1.5 全栈 TRTEdge 移植（Phase V），Track 2 = Cosmos3-Edge 4B WAM 官方管线（Phase W，先行） |

## 现状（2026-10-15）

**第一阶段收官**：P0 驾驶域数据四道闸门全过、两个正式实验（驾驶域三精度评测、分辨率 Pareto）
完成、VLA 可行性闸门实测关闭（结论见上表）。
**第二阶段双轨进行中**（规划与闸门见 [docs/next_plan.md](docs/next_plan.md)）：
- **Phase W（先行）**：Cosmos3-Edge（4B 世界动作模型，TRTEdge 0.10.x 官方支持）——W0 侦察中
- **Phase V**：Alpamayo 1.5 全栈（VLM + 轨迹专家）TRTEdge 移植——参照 Thor fork，待 W 的
  TRTEdge 0.10.x 构建完成后开工

## 核心结论（全部实测，细节与图表在站点）

1. **量化是边缘落地的必需项，不是优化项**：FP16 引擎在 32GB Orin 上 build（峰值 54.8G）
   与加载（~30G）双双 OOM；INT4（4.6G）/ INT8（8.3G）可落地。
2. **量化基本无精度损失——TOST 等价检验（不止"不显著"）**：
   官方基准 n=210：FP16 76.2% / INT8 75.2% / INT4 73.8%；
   自建驾驶域基准 n=184（PhysicalAI-AV 人工核验标注）：51.6% / 47.8% / 50.5%。
   四组量化对比（两域 × INT4/INT8）McNemar 全不显著；"不显著"≠"无损失"（欠功效 null）⇒
   补 **TOST 等价检验**（δ=7pts=本研究 MDE）：INT4 两域 + INT8 机器人域**成立等价**；
   INT8 驾驶域 90%CI [−7.5,−0.1] 下界略超 −7 ⇒ 诚实口径"未检出损失、但不排除最高 ~7.5pts
   真实损失"。（复现 `eval/accuracy/stats_correction.py`）**缩小模型不免费**（2B 显著掉 11 pts）
   ⇒ 激进量化 + 保留大模型。
3. **选型随负载形状翻转**：decode 主导（长文本生成）选 INT4（快 1.61×、能效高 64%、省显存 42%）；
   prefill 主导（看图答短答案——驾驶场景问答正是此类）选 **INT8**（n=210 实测快 1.43×）。
4. **分辨率买得到正确率，性价比拐点在 720p**：INT8 47.8 → 54.9（+7.1 pts）→ 56.0%
   （+8.2 pts vs 448）。**族内校正后**（预注册 4 次 vs-448，Bonferroni α=0.0125）只有
   1080p-vs-448 存活（p_bonf=0.043 / BH q=0.043）；720p 裸 p=0.047 校正后不显著
   （p_bonf=0.188 / q=0.094）⇒「拐点在 720p」是**性价比论点**（720p 拿到 +8.2 里的 +7.1、
   成本仅 1080p 的 1/3：E2E 0.50→3.45→11.26 s、能耗 27.5→202→650 J/请求），非显著性结论；
   720p→1080p 仅 +1.1（事后分析，不校正）。INT8 在每一档同时赢 INT4 的速度/能耗/正确率。
   （复现 `eval/accuracy/stats_correction.py`）
5. **部署定位 = 事件级场景理解（0.1–1 Hz），不是帧率级感知**：原生 1080p 六相机超上下文
   上限 3 倍装不下；距 10 Hz 差 4.6×，量化只给 1.34×。
6. **性能已到硬件天花板**：8B INT4 decode 有效带宽 153.1 GB/s ≈ 该机实测可达上限 152.3；
   手写 W4A16 GEMV 只达可达带宽 52%（TRT 已达 97%，负结果入库）；31 分钟持续负载零降频。
7. **VLA 轨道（Alpamayo-1.5-10B）在 32GB 级 Orin 不可行**：FP16-only 权重 22.16G、官方单样本
   推理 ~24GB VRAM、无量化路径；64GB 机上加载 + CPU 单次前向已验证可行（58 min/次，
   轨迹输出物理自洽）⇒ 受阻点明确而非判死（[eval/vla_gate4/](eval/vla_gate4/)）。

## 与 Alpamayo 家族的关系

- 本项目部署的 **Cosmos-Reason2-8B 正是 Alpamayo 1.5 的 VLM backbone**（卡面与 config
  `vlm_name_or_path` 考证确认）——即 NVIDIA 现役开源 AV 栈的感知推理层。
- **Alpamayo 2 Super（2026-10 开源，34B = 32B VLM + 2B diffusion expert）**：BF16 权重 ~68G
  超过 64GB Orin 总内存，且模型 gated ⇒ 超出本项目硬件范围。它的出现印证了本项目的定位结论：
  NVIDIA 的 AV 基础模型正在 scale 出 Orin 级边缘预算，边缘侧的价值在**量化后的事件级理解**。
- 第三方已有 TRTEdge + Alpamayo 1.5 轨迹栈的先例（`kimsunguk0/tensorrt-edge-llm-for-alpamayo`，
  目标 Jetson Thor）——VLA 轨道若重启，这是最现实的参考路径（见 next_plan 讨论）。

## 仓库布局

```
├── docs/            结果报告 / 方法论 / 问题记录(A–J) / 构建日志 / 规划
│   ├── figures/       图片唯一真源（21 张）
│   └── site/          呈现站点源码（index.html；部署于 gh-pages 分支，figures 为符号链接）
├── eval/            评测资产：n=210 基准、n=184 驾驶域基准、分辨率 Pareto、VLA 闸门、roofline、kernel
├── cloud/           Kaggle 免费云量化/导出管线
├── scripts/orin/    设备端脚本归档
├── configs/         设备环境配置（orin.env）
├── data/            驾驶域评测集（manifest 入 git；帧为可重建大件，gitignore）
├── outputs/         量化产物归档（tgz 大件 gitignore，SHA256 与日志入 git）
├── .claude/         可复用工作流 skill（edge-quantize-deploy）
└── legacy/          前身项目：基于 Qwen3-4B 的文本轨迹预测实验（2026-06~07），
                     结论为"真实行人数据上 CVM 基线反超纯文本 LLM"，由此转向本项目
                     （部署 AV 栈的真实感知推理层而非用 LLM 回归轨迹）
```

## 文档导航

| 文档 | 内容 |
|---|---|
| **[站点](https://ranpin.github.io/alpamayo-edge/)** | **论文式呈现（首选入口）**：摘要 → 核心结果（§1.1–1.12）→ 方法细节 → 思考讨论 |
| [docs/edge_deploy_status.md](docs/edge_deploy_status.md) | 结果报告：链路、体积、性能/功耗/精度/Pareto/VLA 闸门全套指标 |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | 方法与可复现性：模型/数据/校准集来源、指标定义与测法 |
| [docs/PROBLEMS.md](docs/PROBLEMS.md) | 统一问题记录（A–J）：现象 → 根因 → 解决 → 教训 |
| [docs/orin_build_notes.md](docs/orin_build_notes.md) | 工程日志：Orin 构建配方、FMHA 崩溃根因、复现命令 |
| [docs/next_plan.md](docs/next_plan.md) | 规划与决策记录：P0 四道闸门（全过）、P2 可选加固、明确不做清单 |
| [docs/device_layout.md](docs/device_layout.md) | 1 台 Mac + 2 台 Orin + Kaggle 的文件布局、单点与可重建性 |
| [docs/plan.md](docs/plan.md) | 里程碑历史（M0–M5 + 各批次状态表） |
| [docs/FINDINGS.md](docs/FINDINGS.md) | 选型背景（已定案） |
| [eval/accuracy/benchmark/](eval/accuracy/benchmark/) | n=210 机器人域基准：manifest + 三方输出 + 打分脚本 |
| [eval/accuracy/driving/](eval/accuracy/driving/) | **n=184 驾驶域基准**（§1.11）+ [pareto/](eval/accuracy/driving/pareto/) 分辨率 Pareto（§1.12） |
| [eval/vla_gate4/](eval/vla_gate4/) | Alpamayo-1.5-10B 可行性闸门：README + 实测结果 |
| [scripts/orin/](scripts/orin/) | 设备端脚本归档（构建/评测/功耗/闸门 4） |
| [cloud/](cloud/README.md) | Kaggle 免费云量化/导出管线 |

## 评测资产（三套，全部带标答、可复现）

| 资产 | 规模 | 来源 | 用途 |
|---|---|---|---|
| 机器人域 MC 基准 | n=210（robovqa 110 + robofail 100） | 官方 Cosmos-Reason1-Benchmark（repo 自带视频的全部） | 精度主基准（§1.2） |
| **驾驶域 MC 基准** | n=184（8 类 × 23，答案字母配平） | NVIDIA PhysicalAI-AV `ood_reasoning`（人工核验标注） | 驾驶域精度（§1.11）+ 分辨率 Pareto（§1.12） |
| CVM 轨迹基线 | 合成占位 | legacy 项目迁移 | VLA 轨道对照（暂缓） |

## 快速复现（Orin，量化产物已在设备）

```bash
A=/home/vision/chenrunbin/alpamayo-edge          # dog；goat 为 /home/nvidia/...，布局见 docs/device_layout.md
LI=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_inference
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so

# 性能（纯文字路径）
/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_bench \
  --engineDir $A/engines/int4/llm --mode decode --pastKVLen 512 --iterations 200 --warmup 3

# 驾驶域评测（n=184，INT4；输入构造见 eval/accuracy/driving/README.md）
$LI --inputFile $A/bench/ood_eval/ood_input_dog.json \
    --engineDir $A/engines/int4/llm --multimodalEngineDir $A/engines/int4/visual \
    --outputFile $A/bench/ood_eval/out_int4.json

# 分辨率 Pareto（len16384 引擎；功率窗口对齐采集）
bash $A/scripts/run_pareto.sh int4_1080 $A/engines/int4_len16384/llm \
  $A/engines/int4_len16384/visual $A/bench/ood_eval/ood_input_1080.json $A/bench/ood_eval/out.json
```

完整构建配方（含 `-DEMBEDDED_TARGET=jetson-orin` 的 FMHA 坑）、len16384 引擎构建、
云端量化管线见 [docs/orin_build_notes.md](docs/orin_build_notes.md) 与 [cloud/README.md](cloud/README.md)。

## 诚实边界（摘要，全文见站点）

- 精度口径是**相对退化**（量化+引擎+后端的总部署差距，非纯量化误差）；n=184/n=210 的
  MDE ≈ 7 pts，更小的差异测不出。
- 驾驶域标注是 clip 级事件聚类（非逐帧感知标签）；train split 无法排除 CR2 训练见过
  ⇒ 不声称"未见数据上的绝对能力"，跨精度对照不受影响。
- 无轨迹预测输出（ADE/FDE 属 VLA 轨道，受阻点见结论 7）；无多相机实测（token 账见站点 §1.10）。
- INT4@原生 1080p 有 16% 空输出的运行时缺陷（INT8 全干净，PROBLEMS J1，机理未明）。

## 许可与致谢

模型与工具链：NVIDIA Cosmos-Reason2-8B / TensorRT-Edge-LLM / PhysicalAI-AV 数据集。
本项目为个人研究，全部数字可复现（复现路径见各评测目录 README）。
