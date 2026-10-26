# 前身项目训练机副本 · 存储审计（53 G）

> **状态：未处置，待决定**（2026-10-27 审计，用户决定"先都不动，写清楚再说"）
> **位置**：pc-3070 `vision@30.245.40.164:/data/tmp/chenrunbin/projects/qwen-trajectory-prediction/`
> **性质**：alpamayo-edge 前身项目（Qwen3-4B 文本轨迹预测，2026-06~07）的**训练机工作副本**，
> **非 git 仓库**、无远端、独份。

## 一、这是什么

前身项目在 pc-3070（RTX 3070 8G）上做 QLoRA 训练与量化，本目录是当时的工作现场：
训练产物、合并模型、GGUF 量化件、base 权重、conda 环境、处理后数据、训练日志。

项目**方向已关闭**：结论是"真实行人数据上 CVM 基线反超纯文本 LLM"，据此放弃"用 LLM 回归轨迹"
路线、转向部署真实 AV 栈（即现在的 alpamayo-edge）。该结论已写入
[`alpamayo-edge` 主 README](https://github.com/ranpin/alpamayo-edge) 的仓库布局节。

## 二、与 GitHub 仓库 `legacy/` 的关系：**互补，不是重复**

仓库侧 `alpamayo-edge/legacy/`（24 M，78 文件，git 跟踪）是 2026-10-15 冻结前身项目时**整理过的权威副本**。
两边装的是不同东西：

| | 仓库 `legacy/`（24 M） | 本目录（53 G） |
|---|---|---|
| 代码 `scripts/` | 23 个文件 | **同一套**（逐名核对一致） |
| **评测结论** | ✅ **8 个 `eval_results_*.json`** | ❌ **一个都没有**（`find` 实测为空） |
| `demo/` | ✅ 有 | ❌ 无 |
| `docs/technical.md` | ✅ 有 | ❌ 无 |
| 文档版本 | `docs/PRD.md` 等是**冻结时改写过的版本** | 顶层 `PRD.md`/`README.md` 是**原始版本**（sha256 与仓库不同） |
| 训练产物 | ❌ 无 | ✅ 全部在此 |

⇒ **知识（代码 + 结论 + 配方）已完整在仓库；本目录独有的是"训练出来的权重"。**
两边都不是对方的超集，删任何一边都会丢东西——但丢的东西价值差别极大，见下。

**sha256 差异实证**（前 16 位）：

| 文件 | 仓库 `legacy/` | 本目录 |
|---|---|---|
| PRD.md | `ade1efa3cfbd3e1c`（在 `docs/`） | `8c2e1053a71512fd`（在顶层） |
| environment.yml | `fcdb0a9248be8e97` | `061c4087bfd68ec6` |
| requirements.txt | `add04a076f382cbe` | `6fde93c4494077bd` |

## 三、53 G 逐项可再生性

| 内容 | 体积 | 可再生？ | 怎么再生 |
|---|---|---|---|
| 3× `checkpoint-500-merged/`（base+LoRA 合并全量模型） | **22.8 G** | ✅ | base 权重 + adapter 合并，分钟级 |
| `.conda/`（项目内 conda 环境） | **~14 G** | ✅ | `environment.yml` 重建 |
| `models/qwen`（Qwen3-4B base 权重） | **7.6 G** | ✅ | HuggingFace 重下 |
| 3× `*.gguf`（Q4_K_M 量化件） | **7.2 G** | ✅ | 仓库 `scripts/training/quantize_gguf.py` 从合并模型转 |
| 3× `optimizer.pt` | **792 M** | ❌ 但**无价值** | 只对"续训"有用，而该方向已关闭 |
| `data/processed` | **104 M** | ✅ | 仓库 `scripts/data_prep/preprocess.py` 从 raw 重跑 |
| **3× `adapter_model.safetensors`（LoRA 本体，各 132 M）** | **396 M** | ❌ **不可替代** | 重训要 GPU 小时 |
| 训练元数据（`args.json`/`trainer_state.json`/`logging.jsonl`/`runs/`/`logs/`） | **~1 M** | ❌ **不可替代** | 唯一训练记录 |

**结论：53 G 中 99.2%（约 52.6 G）是可再生、可重下、或已无价值的；真正不可替代的只有约 400 M。**

三次训练的时间戳（目录名即 ms-swift 的 run id）：

| run | 时间 | 说明 |
|---|---|---|
| `qwen3-4b-synthetic-lora-world/v0-20260617-113214` | 2026-06-17 | 合成数据 + world |
| `qwen3-4b-synthetic-lora/v0-20260707-144526` | 2026-07-07 | 合成数据 |
| `qwen3-4b-vehicle-lora/v0-20260707-165433` | 2026-07-07 | vehicle（NGSIM） |

另：`outputs/qwen3-4b-awq/` 只有 16 K（仅 `args.json`）——**AWQ 量化未产出成品**，
与 `logs/awq_quantize.log`（7.6 K）对应，是一次未完成的尝试。

## 四、处置选项（待定，本次未执行任何删除）

| 方案 | 动作 | 释放 | 代价 |
|---|---|---|---|
| **A 抽芯删壳**（审计推荐） | 只留 3× adapter（396 M）+ 训练元数据（~1 M），删其余 | **~52.6 G** | 合并模型/GGUF/base/环境需时重建（均有仓库脚本） |
| B 全量归档 scm | 53 G 打 tar + sha256 传 scm 归档盘（12 TB） | 53 G（本机） | 归档 99% 可再生数据，浪费；scm 不可下载 |
| C 全删 | 知识已在仓库，连 400 M 也删 | 53 G | 失去"不重训直接复算负结果"的能力 |
| D 不动（**当前状态**） | 保持原样 | 0 | pc-3070 `/data` 仍剩 309 G，不挤 |

**若选 A/C，那 400 M 孤本建议先异地备份**（拉回 Mac 或传 scm），因为本目录非 git、无远端、独份。

## 五、审计方法（可复核）

- 体积：`du -sh` 含隐藏目录（**注意 `du -sh *` 会漏掉 `.conda`，本次 14 G 差额即由此暴露**）
- 可再生性：逐项核对仓库 `legacy/scripts/` 是否有对应生成脚本（`quantize_gguf.py`/`preprocess.py`/`train_qlora.sh` 均在）
- 孤本判定：`find . -name "eval_results*.json"` 在本机返回空 ⇒ 评测结论不在本机
- 版本差异：`sha256sum` 逐文件对比仓库 `legacy/`
- checkpoint 内部：`ls -la` 确认 `checkpoint-500/` = adapter(132 M) + optimizer(264 M) + 元数据，
  `checkpoint-500-merged/` = 两个 safetensors 分片（4.97 G + 3.08 G）

审计日期 2026-10-27 · 审计人 chenrunbin (ranpin) + Qwen Code
