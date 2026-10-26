# 文件放在哪：各设备目录布局（2026-07-27 整理 · 2026-10-21 归位 · 2026-10-26 复核）

> 本项目横跨 **1 台 Mac + 2 台 Jetson Orin + Kaggle 云**，产物合计约 **240 GB**。
> 两台 Orin 都是**共享账号**（多人共用 `vision` / `nvidia` 登录），所以我的文件统一收在
> **`<home>/chenrunbin/alpamayo-edge/`** 下，各目录内附 `INDEX.md`。此前散落在家目录顶层，容易被误删/误认。
>
> **2026-10-21 归位**：dog 上散在家目录顶层的 `homebot_wm/`（另一项目）已收进 `chenrunbin/`；
> goat 上可重建/可重下的大件（闸门 4 权重 21 G + Cosmos3 policy ONNX 1.3 G）已搬到 dog，
> 与 dog 重复的 Cosmos3 reasoning ONNX 4.6 G 已在 goat 删除（删前 `rsync -c` 逐字节校验）。
> goat 可用空间因此从 23 G 回到 45 G。

## 一、总览

| 位置 | 路径 | 大小 | 角色 | 索引 |
|---|---|---|---|---|
| **本地 Mac** | `~/projects/alpamayo-edge/`（git 仓库，分支 `main`；2026-10-15 由 `qwen-trajectory-prediction` 改名） | ~2 G（tgz 归档后） | 代码 / 文档 / 站点 | 本仓库 `README.md` |
| **orin-dog** `vision@30.245.40.99` | `/home/vision/chenrunbin/alpamayo-edge/` | **171 G** | **主部署机**（32GB 模组）：INT4/INT8/2B 建引擎 + 全部实测；**ONNX 主存放地**（58 G）+ 闸门 4 权重（21 G，2026-10-21 从 goat 搬入） | 机上 `INDEX.md` |
| **orin-goat** `nvidia@30.245.40.73` | `/home/nvidia/chenrunbin/alpamayo-edge/` | **68 G** | **FP16 基线机 + Phase W 机**（64GB）：dog 上 build 不下的 FP16 在此建成；Cosmos3-Edge 全部引擎在此构建与推理 | 机上 `INDEX.md` |
| orin-elk `vision@30.245.40.123` | `/home/vision/chenrunbin/` | 12 K（空） | **无本项目文件** | — |
| pc-3070 `vision@30.245.40.164` | `/home/vision/chenrunbin/` | 104 K（`agentguide`，他项目） | **无本项目文件** | — |
| **scm** `chenrunbin.crb@100.82.100.41` | `/workspace/projects/alpamayo-edge-archive/`（2026-10-15 建成：**7 个 ONNX tgz 共 39 G + SHA256SUMS，全量 `sha256sum -c` 通过**） | 12 TB 空闲 | **归档盘**：**不能下载** —— `huggingface.co` TCP 443 不可达（`pypi.org` 通 ⇒ 白名单受限出网）、且无 ffmpeg/ffprobe。Mac/dog 的大件冗余已在归档验证后删除 | `SHA256SUMS` |
| **Kaggle 云** | 私有 kernel `alpamayo-edge-export` / `-2b` / `-lmhead` | — | 量化 + ONNX 导出 + FP16 参考；**输出即时下载，云端不留** | `cloud/README.md` |
| **GitHub** | `ranpin/alpamayo-edge`（2026-10-09 由 `qwen-trajectory-prediction` 改名） | — | `main`（唯一开发分支）· `gh-pages`（[站点](https://ranpin.github.io/alpamayo-edge/)：`index.html` CR2-8B 篇 + `cosmos3.html` Phase W 篇 + `cosmos3-demo/`） | — |

### goat ↔ dog 免密通道（2026-10-21 建立）

goat 的 `~/.ssh/id_ed25519`（注释 `goat-to-dog-transfer`）公钥已入 dog 的 `authorized_keys`，
两端 rsync 3.2.7，实测 **~111 MB/s**（21 G 用 3 分 10 秒）。大文件互传与校验口径：

```bash
# 在 goat 上推（或反向拉，同一把 key 两个方向都能用）
rsync -a --partial <src>/ vision@30.245.40.99:/home/vision/chenrunbin/alpamayo-edge/<dst>/
# 搬完逐字节校验：空输出 = 完全一致，确认后才删源
rsync -acn --itemize-changes <src>/ vision@30.245.40.99:<dst>/
```

## 二、两台 Orin 的统一分类

同一套子目录名，两台一致：

| 子目录 | 内容 | dog | goat | 可重建？ |
|---|---|---|---|---|
| `engines/` | TensorRT 引擎（**sm_87 专用**，两 Orin 互通，不可跨架构） | **92 G**（7 个 int8 变体各 8.9 G + 3 个 int4 各 6.8 G + 8b_lmhead 4.9 G + 2b 2.7 G + 2b_lmhead 1.5 G） | **43 G**（**fp16 16 G** + int8 8.9 G + int4 6.8 G + **cosmos3_policy_droid 6.1 G + cosmos3_edge 5.5 G**） | ✅ 由 `onnx/` 经 `llm_build` / `cosmos3_*_build` |
| `onnx/` | 建引擎的输入（云端量化导出 + Phase W 本地导出） | **58 G**（fp16 16 G / int8 9.9 G / int8_domain 8.9 G / lmhead 8.1 G / int4 6.8 G / **cosmos3_edge 4.6 G** / 2b 2.7 G / **cosmos3_policy_droid 1.3 G**） | **空**（2026-10-21 已搬/删至 dog） | ✅ 原由 `tarballs/` 解包；Cosmos3 由 `trtedge-0.10.1` 导出 |
| `models/` | HuggingFace 原始权重（Phase W） | — | **18 G**（`Cosmos3-Edge` + `Cosmos3-Edge-Policy-DROID`） | ✅ 可重下（hf-mirror） |
| `trtedge-0.10.1/` | **Phase W 自建 TRTEdge**（Cosmos3-Edge 需 0.10.x；含 `build_orin/` 与自生成的 CuTe DSL sm_87+cuda12 预编译件） | 673 M | **6.4 G**（完整构建） | ✅ 由 `scripts/orin/w0b_build_trtedge_0101.sh` 重建 |
| `tarballs/` | 传输/迁移打包 | 已清空（2026-10-15 归档 scm 验证后删，回收 32 G） | 已删（2026-10-12，回收 40 G） | ✅ **纯冗余，可删** |
| `bench/` | n=210 基准 + `ood_eval/`（驾驶域 n=184：448/720/1080 三档帧 + 输入/输出/日志） | 407 M | 33 M（`ood_eval/` 曾用于 FP16 n=184） | ⚠️ 输出是实测记录 |
| `alpamayo_gate4/` | 闸门 4（Alpamayo-1.5-10B 可行性）产物 | **21 G**（`Alpamayo-1.5-10B/` 权重，2026-10-21 从 goat 搬入） | **33 M**（只剩证据：`gate4_result.json/npz`、`meta.npz`、`precheck.log`、`gate4_forward.py`、`frames/`、`processor/`） | ✅ 权重可重下（公开非 gated，须 `HF_HUB_DISABLE_XET=1`）；见 `eval/vla_gate4/README.md` |
| `alpamayo1.5/` | NVlabs 仓库 clone + `a1_5_venv`（py3.12 / torch2.8 CPU） | — | 1.3 G | ✅ 可重 clone |
| `io/` | 各次推理的请求/响应 JSON | 216 K | — | ❌ 实测记录 |
| `logs/` | 构建 / 功耗 / 运行原始日志 | 13 M | 3.1 M | ❌ 实测记录 |
| `prof/` | Nsight Systems 剖析 | 6.6 M | — | ❌ 实测记录 |
| `kernels/` | 手写 W4A16 GEMV 微基准 | 1.1 M | — | ❌ 结果已入 git |
| `scripts/` | 设备端脚本（功耗采样、建引擎+bench、Phase W 全链） | 72 K | 52 K | ❌ 已入 git（`scripts/orin/`） |
| `trtedge/` | TRTEdge **0.9.0** 已编译二进制（第一阶段用） | 用共享安装 `/home/vision/TensorRT-Edge-LLM/`（**不在我目录**） | 92 M（从 dog 拷来） | ✅ |

> ⚠️ **Cosmos3 的 ONNX 与引擎现在分处两机**：ONNX 只在 dog（`onnx/cosmos3_*`），引擎只在 goat
> （`engines/cosmos3_*`）。goat 若要重建 Cosmos3 引擎，需先用上面的免密通道从 dog 把 ONNX 拉回来。

### dog 上保留的两个兼容软链
`/home/vision/engines` → `chenrunbin/alpamayo-edge/engines`
`/home/vision/bench` → `chenrunbin/alpamayo-edge/bench`

原因：**已入 git 的运行记录里写死了旧绝对路径**（如
`eval/accuracy/benchmark/orin_int4_outputs_n210.json` 里的 `/home/vision/bench/frames/robovqa_0_0.jpg`、
`eval/vision/io/worst_req.json`）。这些是实测记录，改写就不再忠实，所以**留软链而不改记录**，
老命令与老记录都能原样重跑。确认无人依赖后可删软链。

### 不属于本项目、整理时未动的
- `/home/vision/TensorRT-Edge-LLM/`（**共享框架安装 0.9.0**，`~/.bashrc` 的 `EDGELLM_PLUGIN_PATH` 指向它；
  移动会让所有人的推理崩，见 [`PROBLEMS.md`](PROBLEMS.md) B2/F1）
- dog `~/chenrunbin/` 下的**其他个人项目**：`homebot_wm` 2.0 G（2026-10-21 归位进来）、
  `qwen-trajectory-prediction` 7.0 G、`vlm` 3.2 G、`llama.cpp` 1.4 G、`memolink` 307 M、
  `edge-agent-memory` 147 M、`deps` 37 M
- `/home/vision/llama.cpp`（2026-07-07 的纯净上游克隆，无构建产物、无模型 ⇒ **无法归属**，未动）
- 他人文件：`ywz/` `ywz.log` `zwx/` `inference_images/` `activevision.log` `carcontrol.log` `find.log` `NV_*.sh`
- **dog 原 `/home/chenrunbin/`（工作代码 aadkcore + agent_group，remote 指向内网 GitLab）已于
  2026-10-21 删除**——与个人项目分属两类、不宜混放；删前已核 0 进程 / 0 systemd / `/opt/agentcore` 无引用。
  岚图工作代码现只在 scm 机。

## 三、本地 Mac 的分类

本地目录 **`~/projects/alpamayo-edge/`**（2026-10-15 由 `qwen-trajectory-prediction` 改名；
同日仓库结构提升：原 `alpamayo-edge/` 子目录内容即仓库根，前身项目挪入 `legacy/`）：

```
~/projects/alpamayo-edge/          # 仓库根 = 项目本身
├── cloud/        Kaggle / Colab 量化导出脚本（入 git）
├── docs/         METHODOLOGY / PROBLEMS / 本文 / figures/（图片唯一真源）/ site/（发布站点源 → gh-pages，其 figures 为符号链接）
├── eval/         全部评测脚本 + 整理后的结果与原始 I/O（入 git）
│   ├── accuracy/ 精度：探针集 + n=210 基准 + 驾驶域 n=184 + 分辨率 Pareto + 校准集消融 + 统计校正
│   ├── cosmos3/  Phase W：reasoner n=184 跨代对比 + policy 帧敏感性复测（含 repro 脚本与证据）
│   ├── vla_gate4/ Alpamayo-1.5-10B 可行性闸门产物
│   ├── vision/   视觉编码器扫描（含 io/ 逐字请求响应）
│   ├── kernels/  手写 GEMV 微基准源码 + 结果
│   ├── perf/     跨设备加速比
│   └── profile/  Nsight 汇总 CSV
├── scripts/orin/ 设备端脚本归档（入 git；含 Phase W 的 W0–W2 全链与两个分解补丁）
├── legacy/       前身项目（Qwen3-4B 文本轨迹预测，冻结）
├── .claude/      skill edge-quantize-deploy（入 git）
└── outputs/      **大件 gitignore**，仅本地：
    ├── *.tgz             自产量化权重 7 件 39 G —— **已归档至
    │                     scm:/workspace/projects/alpamayo-edge-archive/ 并全量 sha256 验证，
    │                     本地副本已删（2026-10-15）**
    ├── *.sha256          校验和（小溯源件）
    └── archive/          原始日志（orin_logs/ goat_logs/ orin_prof/）+ MANIFEST.md（SHA256）
```

**入 git 与只在本地的分界**：小而经过整理、被文档引用的 → git；原始大件与原始日志 → `outputs/`（本地 + MANIFEST）。
设备专用可重建大件（TRT 引擎）→ 只在设备上，不拉回本地（见 [`../outputs/archive/MANIFEST.md`](../outputs/archive/MANIFEST.md)）。

## 四、单点风险与磁盘

| 东西 | 只存在于 | 丢了怎么办 |
|---|---|---|
| **FP16 引擎** 16 G | **仅 orin-goat** `engines/fp16` | ✅ 可重建：ONNX 在 **dog `onnx/fp16`** + scm 归档（`edge_fp16_llm.tgz`，SHA256SUMS 校验通过），但**必须在 ≥64 GB 内存的机器上** build（32GB 机峰值 54.8 G 会 OOM）⇒ 只能在 goat 建 |
| **Cosmos3 引擎** 11.6 G | **仅 orin-goat** `engines/cosmos3_*` | ✅ 可重建：ONNX 在 **dog `onnx/cosmos3_*`**（需先拉回 goat），构建脚本 `scripts/orin/w1_goat_engine_build.sh` / `w2_policy_rebuild_decomposed.sh` |
| **Cosmos3 ONNX** 5.9 G | **仅 orin-dog** `onnx/cosmos3_*` | ✅ 可重导出：权重在 goat `models/`，导出需先打分解补丁（`apply_decomposed_attn_patch.py` / `apply_policy_decomposed_patch.py`） |
| **闸门 4 权重** 21 G | **仅 orin-dog** `alpamayo_gate4/Alpamayo-1.5-10B` | ✅ 可重下（公开非 gated）；证据小件在 goat `alpamayo_gate4/`（33 M） |
| INT4/INT8/2B 引擎 | 两台 Orin | ✅ 由 ONNX 重建（sm_87 机器即可；ONNX 在 dog `onnx/` + scm 归档双份） |
| 全部原始日志 / 请求响应 / 剖析 | 已于 2026-07-27 **全部拉回本地** | ✅ 本地 + MANIFEST |
| 自产量化 ONNX | **scm 归档（唯一 tgz 副本，7 件 39 G，sha256 全验）** + dog 解包目录 | ✅ 最坏情况重跑 Kaggle 量化 |

**磁盘（2026-10-26 实测）**：

| 设备 | 分区 | 总量 | 已用 | 可用 | 备注 |
|---|---|---|---|---|---|
| orin-goat | `/`（nvme0n1p1） | 446 G | 379 G | **45 G（90%）** | 共享机，他人占用会波动；2026-10-19 曾低至 12 G |
| orin-dog | `/home`（nvme0n1p1） | 916 G | 366 G | **504 G（43%）** | 宽松，大件优先放这里 |
| orin-dog | `/`（mmcblk0p1） | 57 G | 24 G | 31 G（45%） | 系统盘，别往这儿放大件 |
| 本地 Mac | — | — | — | — | 仓库 ~2 G（tgz 已归档删除） |

> **规划口径**：需要 **大内存**（FP16 build、10B 级引擎）→ 只能 goat，但要先看 `df -h /` 是否够盘；
> 需要 **大磁盘**（下载权重、存 ONNX）→ dog。goat 盘紧时按本文 §一 的免密通道把可重建大件挪到 dog。
