# Alpamayo-Edge

**NVIDIA Alpamayo 家族 AV 轨迹 VLA 在 Jetson AGX Orin 上的边缘部署与同口径评测**——
**R1**（10B，主打：FP16 全链跑通 + backbone INT4 量化近乎免费）、**1.5**（repack 跨代对照，
诚实口径：工作未完整收官、导航引导假设被否定）、**Alpamayo2-Edge**（2.65B，2026-11-10 发布，
轨迹域「缩小免费」实证）。全部 Orin 实机实测。零预算 · 零训练。

🛣️ **在线 Demo · R1 轨迹预测**（真实帧叠加 + 滚动预测视频 + FP16/INT4 三线对比 · 真实 Orin 推理）：<https://ranpin.github.io/alpamayo-edge/r1-trajectory/>

📊 **呈现站点（家族总览）**：<https://ranpin.github.io/alpamayo-edge/>

> **仓库沿革**：本仓原为 Cosmos-Reason2-8B 边缘量化部署项目（CR2 篇）。2026-11-15 按 NVIDIA 模型
> 家族重组——CR2/Cosmos3 全套迁入姊妹仓 [cosmos-edge](https://github.com/ranpin/cosmos-edge)，
> 本仓聚焦 Alpamayo 家族。方法学体系（五轴评测 · roofline · 预注册闸门 · 等价检验）建立在 CR2 篇，
> [CR2 报告页](https://ranpin.github.io/cosmos-edge/cr2.html) §1.13–1.14 按时间线收录 R1 INT4 与
> 2-Edge 两章（资产在本仓 `eval/r1/`）。两仓共享设备资产与 TRTEdge 0.10.1 构建脚本。

## 核心结论

1. **R1 全链跑通、minADE 与官方基线一致**——三引擎 FP16（llm 14.15G / visual 1.08G / action 4.25G），
   minADE6@6.4s = 0.804 m vs 官方 0.821 m（落在 95% CI 内）。
2. **R1 backbone INT4 AWQ 量化近乎免费**——同口径 0.800 vs FP16 0.804 m（Δ=−0.004 m，比预注册阈值
   0.1 m 小 25 倍），decode 2.52×、引擎 3.11× 小；「量化几乎免费」从 CR2 推理轨道延伸到 VLA 轨迹轨道
   （详见 [eval/r1/int4_backbone.md](eval/r1/int4_backbone.md)）。
3. **跨代对比 R1 胜 Alpamayo-1.5、导航引导非原因**——1.5 经 repack（纯 config 改写）跑上 64GB Orin，
   同口径 minADE6@6.4s R1 0.804 vs 1.5 1.815 m（+126%）；oracle 导航仅 −1%（within-noise），
   「1.5 需要导航」假设被语义 + 量化两个独立测量否定。诚实口径：1.5 侧为跨代对照，工作未完整收官
   （详见 [eval/r1/crossgen_a15.md](eval/r1/crossgen_a15.md)）。
4. **轨迹域「缩小免费」：Alpamayo2-Edge（2.65B）与 R1（10B）统计平手、点估计略优**——同口径
   0.734 vs 0.804 m（配对 p=0.61、10 优/8 劣）；2-Edge 免量化直跑 Orin（BF16 权重 5.3G、PyTorch 零优化）。
   与推理域缩小显著输（Cosmos3 2B vs CR2 8B 量化版，p=0.0115，见 cosmos-edge）对照 ⇒
   **缩小代价域依赖：推理域不免费、轨迹域免费**（详见 [eval/r1/crossgen_a2e.md](eval/r1/crossgen_a2e.md)）。
5. **Alpamayo-1.5-10B 在 32GB Orin 不可部署**（闸门 4 实测：FP16-only 22.16G、官方 ~24GB VRAM、
   无量化路径）⇒ 受阻点明确而非判死（详见 [eval/vla_gate4/README.md](eval/vla_gate4/README.md)）。
6. **action 专家撞 K6 同源墙**（TRT≥11-only 算子），分解补丁第三次打通（PROBLEMS §M，
   前两次在 cosmos-edge 的 Cosmos3 reasoning/policy）。

## 快速开始

R1 全链复现（下载→补丁→对拍→导出→构建→推理→评分）见
[scripts/orin/README.md](scripts/orin/README.md) 与 [eval/r1/README.md](eval/r1/README.md)；
2-Edge 五件套（取数/环境/补丁/推理/评分）在 `eval/r1/a2e/`。需 Jetson AGX Orin 实机，无消费级一键入口。

## 仓库布局

```
├── eval/r1/         R1 评测资产：INT4 backbone、1.5/2-Edge 跨代、dgural GT 叠放 demo、seed 敏感性（各带 README）
├── eval/vla_gate4/  闸门 4：Alpamayo-1.5-10B 边缘可行性实测
├── cloud/           Kaggle 免费云导出/量化（R1 backbone INT4 + AV 域校准对照臂；CR2 管线已迁 cosmos-edge）
├── scripts/orin/    设备端脚本归档（R1 全链 / 1.5 repack / gate4；CR2 与 Cosmos3 脚本已迁 cosmos-edge）
├── docs/            PROBLEMS §M / device_layout（两仓设备布局）/ phase_v_v0_recon
├── configs/         设备环境配置（orin.env）
├── outputs/         R1 量化产物归档（tgz 大件 gitignore，SHA256 入 git）+ CVM 基线
└── legacy/          前身项目：Qwen3-4B 文本轨迹预测实验（由此转向本项目）
```

## 文档导航

| 文档 | 内容 |
|---|---|
| **[呈现站点](https://ranpin.github.io/alpamayo-edge/)** | 家族总览（R1 / 跨代 / 两仓地图） |
| [eval/r1/README.md](eval/r1/README.md) | R1 评测总览：全链、INT4、跨代、demo |
| [docs/PROBLEMS.md](docs/PROBLEMS.md) | 工程问题记录 §M（§A–L 在 [cosmos-edge](https://github.com/ranpin/cosmos-edge/blob/main/docs/PROBLEMS.md)） |
| [docs/device_layout.md](docs/device_layout.md) | 两台 Orin 设备布局（两仓资产同一布局） |
| [docs/phase_v_v0_recon.md](docs/phase_v_v0_recon.md) | Phase V（VLA 轨道）侦察记录 |

## 许可与数据来源

- 代码 **MIT**。
- 模型：`nvidia/Alpamayo-R1-10B`、`nvidia/Alpamayo-1.5-10B`、`nvidia/Alpamayo2-Edge`（OpenMDW-1.1）。
- demo 帧取自 dgural 公开样本（PhysicalAI-AV 公开 100-clip 子集）；minADE 评测用 chunk3116 子集
  （18 clips × 6 seeds，逐 clip 原始数据涉 gated 仓库，不入库）。
