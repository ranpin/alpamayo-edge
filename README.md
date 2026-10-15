# Alpamayo-Edge

**自动驾驶 VLM（Cosmos-Reason2-8B = NVIDIA Alpamayo 1.5 的 backbone）在 Jetson AGX Orin 上的
量化部署与全套实测研究**——INT4(AWQ)/INT8(SmoothQuant) 双精度引擎，精度、延迟、能耗、内存、
可靠性五轴证据，零预算、零训练、全 Orin 实测。

📊 **呈现站点（论文式，图 1–21 / 表 1–13）**：<https://ranpin.github.io/alpamayo-edge/>
📁 **项目详情**：[alpamayo-edge/README.md](alpamayo-edge/README.md)

## 一分钟版本

| 问题 | 实测答案 |
|---|---|
| 8B 驾驶 VLM 能上 32GB Orin 吗？ | **只有量化能**：FP16 build/加载双 OOM；INT4 4.6G / INT8 8.3G 落地 |
| 量化掉多少精度？ | **统计上不掉**：机器人域 n=210 与驾驶域 n=184 双基准、六组 McNemar 全不显著 |
| INT4 还是 INT8？ | **看负载**：长生成选 INT4（decode 快 1.61×）；看图答短答案选 INT8（prefill 快 1.43×，且驾驶域各分辨率档全胜） |
| 分辨率值不值？ | **值，到 720p 为止**：INT8 +7.1 pts（p=0.047）；1080p 再 +1.1（ns）但成本 3.2× |
| 能进 AV 感知环路吗？ | **不能，也不该**：真实分辨率 0.1–1 Hz ⇒ 定位事件级场景理解（风险解释/日志挖掘/驾驶员问答） |
| 还有优化空间吗？ | **量化侧已穷尽**：decode 带宽达硬件可达上限（153.1≈152.3 GB/s）、手写 kernel 负结果、31min 零降频 |
| VLA（轨迹输出）呢？ | **Alpamayo-1.5-10B 在 32GB 级不可行**（FP16-only 22.16G）；64GB 机 CPU 前向已验证可跑通 |

## 仓库布局

```
├── alpamayo-edge/     ★ 本项目全部内容（代码/文档/评测资产/站点源码）
│   ├── README.md        项目总览（核心结论、文档导航、快速复现）
│   ├── docs/            结果报告 / 方法论 / 问题记录(A–J) / 构建日志 / 规划
│   ├── eval/            评测资产：n=210 基准、n=184 驾驶域基准、分辨率 Pareto、VLA 闸门、roofline、kernel
│   ├── cloud/           Kaggle 免费云量化/导出管线
│   ├── scripts/orin/    设备端脚本归档
│   └── site/            呈现站点源码（部署于 gh-pages 分支）
│
└── （根目录其余文件）   前身项目遗留：基于 Qwen3-4B 的文本轨迹预测实验（2026-06~07，
                        main 分支），结论为"真实行人数据上 CVM 基线反超纯文本 LLM"，
                        由此转向本项目（部署 AV 栈的真实感知推理层而非用 LLM 回归轨迹）
```

## 许可与致谢

模型与工具链：NVIDIA Cosmos-Reason2-8B / TensorRT-Edge-LLM / PhysicalAI-AV 数据集。
本项目为个人研究，全部数字可复现（复现路径见各评测目录 README）。
