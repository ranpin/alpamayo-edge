# 分辨率 Pareto（448 / 720p / 原生 1080p × 正确率 × 延迟 × 能耗）

P0 正式实验 ②（2026-10-10）。§1.11 已在 448 档建立驾驶域精度基线；§1.10 给了延迟预算的
解析模型。本实验回答缺失的大轴：**分辨率到底买不买得到正确率，代价是什么**。

## 设计

- **6 帧固定、只变分辨率**（任务输入与 §1.11 完全一致，唯一变量是每帧像素数）：
  - 448 档：长边 448（448×252，processor 对齐 448×256）= 112 token/帧，6 帧 672
  - 720p 档：长边 1280（1280×720）= 880–920 token/帧（processor 32 对齐方向在 720 上是平局，
    实测以运行为准），6 帧 ≈ 5.3–5.5k
  - 1080p 档：原生 1920×1080 逐字节复制（不重编码），processor 对齐 1920×1088 = 2040 token/帧，
    6 帧 12,240（prompt 合计 ≈ 12.4k）
- **引擎**：len4096 装不下 720p/1080p 的 6 帧 ⇒ 新建 len16384 引擎
  （`scripts/orin/build_len16384.sh`）：
  - `engines/int4_len16384/llm`（dog 建，峰值 25.7G，4 分钟）+ `visual`（maxImageTokens 16384 /
    perImage 2048，恰好 ≥ 2040）
  - `engines/int8_len16384/llm`（dog 建，峰值 29.45G 恰好挤进 29G——**外推 ~37G 被实测推翻**，
    INT8 build 内存对长度几乎不敏感：len4096 29.7G → len16384 29.45G）
- **运行**：`scripts/orin/run_pareto.sh` —— llm_inference + tegrastats 功率窗口对齐采集
  （tegrastats 在引擎加载完、第一个请求开始处理后才启动 ⇒ 采样窗口 ≈ 纯处理阶段）。
- **计时**：逐请求 E2E 从 run 日志的逐请求时间戳解析（每请求恰 1 条 "Processing vision inputs"
  毫秒时间戳行；`parse_pareto.py`）。E2E ≈ TTFT + 1–2 个 decode token（MC 答案是单字母）。
- **能耗**：J/请求 = 处理窗口平均总功率（VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0）× E2E。
- 生成参数与 §1.11 逐位相同（贪心、max_generate_length 12、batch 1）。

## 复现

```bash
# 本地：帧（不入 git，可确定性重建）
.venv/bin/python eval/accuracy/driving/build_ood_input.py \
    --long-edge 1280 --out-dir data/ood_eval/frames_720 \
    --frames-root /home/vision/chenrunbin/alpamayo-edge/bench/ood_eval/frames_720 \
    --json-out eval/accuracy/driving/pareto/ood_input_720.json
.venv/bin/python eval/accuracy/driving/build_ood_input.py \
    --native-copy --out-dir data/ood_eval/frames_1080 \
    --frames-root /home/vision/chenrunbin/alpamayo-edge/bench/ood_eval/frames_1080 \
    --json-out eval/accuracy/driving/pareto/ood_input_1080.json
# 448 档复用 §1.11 的 ood_input_dog.json（帧在 bench/ood_eval/frames/）

# 设备：建引擎 → 跑档（dog）
bash scripts/orin/build_len16384.sh
bash scripts/orin/pareto_int4_driver.sh     # smoke(1080p 1 题) → 448 → 720 → 1080
bash scripts/orin/pareto_int8_driver.sh     # int8 448 复测 → G3 构建 → 720 → 1080

# 本地：打分 + 计时/能耗 + 图
.venv/bin/python eval/accuracy/driving/score_ood.py \
    --runs t448=pareto/orin_int4_448_outputs_n184.json \
           t720=pareto/orin_int4_720_outputs_n184.json \
           t1080=pareto/orin_int4_1080_outputs_n184.json \
    --reference t448 --out pareto/RESULTS_pareto_int4.json
.venv/bin/python eval/accuracy/driving/pareto/parse_pareto.py \
    --tags int4_448 int4_720 int4_1080 --out pareto_timing_power_int4.json
.venv/bin/python eval/accuracy/driving/pareto/pareto_viz.py
```

## 文件

| 文件 | 说明 |
|---|---|
| `orin_{int4,int8}_{448,720,1080}_outputs_n184.json` | 设备原始输出（实测记录） |
| `run_pareto_*.log` / `pw_pareto_*.log` | 运行日志（逐请求时间戳）/ tegrastats 功率采样 |
| `RESULTS_pareto_{int4,int8}.json` | 打分结果（McNemar 参考 = 448 档） |
| `pareto_timing_power_{int4,int8}.json` | E2E / 功率 / J-per-请求 |
| `parse_pareto.py` / `pareto_viz.py` | 日志解析 / 图 21 生成 |
| `orin_int4_1080_retry30_outputs.json` | 空输出现象的 30 题重跑（cap 64，见 PROBLEMS J1） |

## 结果（2026-10-10，全部完成）

| 档位 | INT8 正确率（vs 448） | INT8 E2E / 能耗 | INT4 正确率（vs 448） | INT4 E2E / 能耗 |
|---|---|---|---|---|
| 448 | 47.8% | 0.50 s / 27.5 J | 50.5% | 0.68 s / 45.1 J |
| 720p | **54.9%（+7.1，p=0.047）** | 3.45 s / 202 J | 53.8%（+3.3，p=0.392） | 5.50 s / 345 J |
| 原生 1080p | **56.0%（+8.2，p=0.011；vs 720p +1.1，p=0.774）** | 11.26 s / 650 J | 44.0%†（−6.5，p=0.111） | 15.99 s / 966 J |

**结论：分辨率买得到正确率，拐点在 720p；INT8 每档同时赢 INT4 的速度/能耗/正确率。**
† INT4@1080p 受空输出缺陷污染（见下）。站点 §1.12（图 21 / 表 13）。

## 诚实边界

- **INT4@1080p 有 30/184（16.3%）空输出**（不可解码 token，计为答错）——**INT4(AWQ) 特有**
  的运行时边界现象（INT8 同帧同题 552/552 全干净），非分辨率-精度权衡本身；30 题孤立重跑
  12 题恢复 ⇒ 部分依赖运行时累积状态；干净 154 题上 INT4@1080p=52.6%（同题 INT8=54.5%）。
  详见 PROBLEMS J1（机理未明，token id 级取证未做）。
- INT8@1080p 首次运行（KV 预留 32768）在 ~140/184 题被 OOM 杀（tegrastats 实测 RAM 从第一题
  起 28.6G 边缘、无泄漏、末段尖峰 30.4G）；**KV 预留降到 16448（精确省 2.41G，不影响计算）
  重建后 184/184 跑完，全程 26.2G**。上表 INT8@1080p 即 KV16448 引擎的结果。PROBLEMS J2（已解决）。
- FP16 不在 Pareto 内（32GB dog 不可部署；goat 的 FP16 引擎是 len1024，建 len16384 需 ~62G
  build 内存且 goat 与其他用户共享）。
- 720p 档 token 数依赖 processor 对 720 的 32 对齐平局取舍（880 或 920/帧），不影响任何结论。
- 统计口径：预注册 4 次 vs-448 McNemar；Bonferroni α=0.0125 下 1080p p=0.011 存活、
  720p p=0.047 不存活；vs-720p 检验为事后分析。n=184 ⇒ 每档 MDE ≈ 7 pts。
