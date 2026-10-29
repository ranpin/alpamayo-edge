# Alpamayo-Edge

**NVIDIA Physical AI 模型在 Jetson AGX Orin 上的边缘量化部署与全套实测研究**——一套 TensorRT-Edge
管线部署多代模型：Cosmos-Reason2-8B（INT4/INT8 量化五轴实测）、Cosmos3-Edge（跨代精度对比）、
Alpamayo-R1（AV 轨迹预测）。零预算、零训练、全 Orin 实测。

🛣️ **在线 Demo · Alpamayo-R1 轨迹预测**（真实帧叠加 · 真实 Orin 推理）：<https://ranpin.github.io/alpamayo-edge/r1-trajectory/>
🎬 **在线 Demo · Cosmos3-Edge 2B 驾驶场景理解**：<https://ranpin.github.io/cosmos3-edge/cosmos3-demo/>（Cosmos3 部署与 policy 动作链已拆至独立仓 [cosmos3-edge](https://github.com/ranpin/cosmos3-edge)）

📊 **呈现站点（论文式 · 细节首选入口）**：[CR2-8B 量化部署](https://ranpin.github.io/alpamayo-edge/) ｜ [Cosmos3 跨代对比](https://ranpin.github.io/alpamayo-edge/cosmos3.html)

## 核心结论（一句话版）

> 全部数字、图表与统计口径在[呈现站点](https://ranpin.github.io/alpamayo-edge/)，方法与复现在 [docs/](docs/)；此处只留结论本身。

1. **量化是边缘落地的必需项，不是优化项**——FP16 在 32GB Orin build/加载双 OOM，INT4（4.6G）/ INT8（8.3G）才落地。
2. **量化基本无精度损失**（TOST 等价检验，δ=7pts）；**缩小模型不免费**（2B 显著掉点）⇒ 部署策略 = 激进量化 + 保留大模型。
3. **选型随负载形状翻转**——长生成选 INT4（decode 快 1.61×），看图答短答案选 INT8（prefill 快 1.43×）。
4. **分辨率买得到正确率，性价比拐点 720p**（拿到 +8.2pts 里的 +7.1，成本仅 1080p 的 1/3）。
5. **部署定位 = 事件级场景理解（0.1–1 Hz），非帧率级感知**；量化侧性能已到硬件天花板（decode 带宽 153 ≈ 上限 152 GB/s）。
6. **VLA 轨道（Alpamayo-1.5-10B）在 32GB Orin 不可行**（FP16-only 22.16G、无量化路径）⇒ 受阻点明确而非判死。
7. **Alpamayo-R1 全链跑通、minADE 与官方基线一致**——minADE6@6.4s = 0.804 m vs 官方 0.821 m（落在 95% CI 内）。
8. **Cosmos3-Edge 跨代验证「缩小不免费、量化几乎免费」**——2B reasoner n=184 驾驶域 42.4%，显著输给上一代 8B 量化版（详见 [cosmos3-edge](https://github.com/ranpin/cosmos3-edge) 与[跨代对比](https://ranpin.github.io/alpamayo-edge/cosmos3.html)）。

## 仓库布局

```
├── docs/            技术文档：结果报告 / 方法论 / 问题记录(A–M) / 构建日志 / 规划
│   ├── figures/       图片唯一真源
│   └── site/          呈现站点源码（index.html + cosmos3.html + r1-trajectory/；部署于 gh-pages）
├── eval/            评测资产：n=210 / n=184 驾驶域基准、分辨率 Pareto、VLA 闸门、roofline、kernel、cosmos3 跨代、r1
├── cloud/           Kaggle 免费云量化/导出管线
├── scripts/orin/    设备端脚本归档（构建 / 评测 / 功耗 / 闸门）
├── configs/         设备环境配置（orin.env）
├── data/            驾驶域评测集（manifest 入 git；帧为可重建大件，gitignore）
├── outputs/         量化产物归档（tgz 大件 gitignore，SHA256 与日志入 git）
└── legacy/          前身项目：Qwen3-4B 文本轨迹预测实验（由此转向本项目）
```

## 文档导航

| 文档 | 内容 |
|---|---|
| **[呈现站点](https://ranpin.github.io/alpamayo-edge/)** | **论文式呈现（细节首选入口）**：摘要 → 核心结果 → 方法细节 → 思考讨论 |
| [docs/edge_deploy_status.md](docs/edge_deploy_status.md) | 结果报告：链路、体积、性能 / 功耗 / 精度 / Pareto / VLA 闸门全套指标 |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | 方法与可复现性：模型 / 数据 / 校准集来源、指标定义与测法、评测资产 |
| [docs/PROBLEMS.md](docs/PROBLEMS.md) | 统一问题记录（A–M）：现象 → 根因 → 解决 → 教训 |
| [docs/orin_build_notes.md](docs/orin_build_notes.md) | 工程日志：Orin 构建配方、快速复现命令、FMHA 崩溃根因 |
| [docs/next_plan.md](docs/next_plan.md) | 规划与决策：闸门、与 Alpamayo 家族的关系、明确不做清单 |
| [eval/](eval/) | 评测资产（各带 README）：n=210 / n=184 基准、Pareto、VLA 闸门、cosmos3 跨代、r1 |

> docs/ 下另有 device_layout / plan / FINDINGS / M0_setup / M5_report 等专项与历史文档。

## 许可与致谢

模型与工具链：NVIDIA Cosmos-Reason2-8B / TensorRT-Edge-LLM / PhysicalAI-AV 数据集。本项目为个人研究，
全部数字可复现（复现路径见 [docs/orin_build_notes.md](docs/orin_build_notes.md) 与各评测目录 README）。
