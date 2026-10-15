# Alpamayo-Edge：自动驾驶 VLM 的边缘量化部署实证

> 把 **NVIDIA Alpamayo 1.5（现役开源 AV 栈）的 VLM backbone——Cosmos-Reason2-8B** 用
> **TensorRT-Edge-LLM** 量化（INT4 AWQ / INT8 SmoothQuant）部署到 **Jetson AGX Orin**，
> 给出精度、延迟、能耗、内存、可靠性的**全套实测证据**，回答"边缘到底能到什么程度"。
> **零预算 · 零训练 · 全 Orin 实测。**

📊 **呈现站点（论文式结构，图 1–21 / 表 1–13）**：<https://ranpin.github.io/alpamayo-edge/>

## 现状（2026-10-15）

**规划内工作全部收官**：P0 驾驶域数据四道闸门全过、两个正式实验（驾驶域三精度评测、
分辨率 Pareto）完成、VLA 轨道可行性闸门（Alpamayo-1.5-10B）实测关闭。
无进行中事项；可选加固项见 [docs/next_plan.md](docs/next_plan.md) 的 P2。

## 核心结论（全部实测，细节与图表在站点）

1. **量化是边缘落地的必需项，不是优化项**：FP16 引擎在 32GB Orin 上 build（峰值 54.8G）
   与加载（~30G）双双 OOM；INT4（4.6G）/ INT8（8.3G）可落地。
2. **量化无显著精度损失——机器人域 + 驾驶域双证据**：
   官方基准 n=210：FP16 76.2% / INT8 75.2% / INT4 73.8%；
   自建驾驶域基准 n=184（PhysicalAI-AV 人工核验标注）：51.6% / 47.8% / 50.5%。
   六组对比 McNemar 全部不显著。**缩小模型不免费**（2B 显著掉 11 pts）⇒ 激进量化 + 保留大模型。
3. **选型随负载形状翻转**：decode 主导（长文本生成）选 INT4（快 1.61×、能效高 64%、省显存 42%）；
   prefill 主导（看图答短答案——驾驶场景问答正是此类）选 **INT8**（n=210 实测快 1.43×）。
4. **分辨率买得到正确率，拐点在 720p**：INT8 47.8 → 54.9（+7.1 pts, p=0.047）→ 56.0%
   （+8.2 pts, p=0.011；720p→1080p 仅 +1.1 不显著）；代价 E2E 0.50→3.45→11.26 s、
   能耗 27.5→202→650 J/请求。INT8 在每一档同时赢 INT4 的速度/能耗/正确率。
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
| [docs/site/](docs/site/) | 站点源码（gh-pages 分支部署；figures 符号链接至 docs/figures/） |

## 评测资产（三套，全部带标答、可复现）

| 资产 | 规模 | 来源 | 用途 |
|---|---|---|---|
| 机器人域 MC 基准 | n=210（robovqa 110 + robofail 100） | 官方 Cosmos-Reason1-Benchmark（repo 自带视频的全部） | 精度主基准（§1.2） |
| **驾驶域 MC 基准** | n=184（8 类 × 23，答案字母配平） | NVIDIA PhysicalAI-AV `ood_reasoning`（人工核验标注） | 驾驶域精度（§1.11）+ 分辨率 Pareto（§1.12） |
| CVM 轨迹基线 | 合成占位 | 父仓库迁移 | VLA 轨道对照（暂缓） |

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
