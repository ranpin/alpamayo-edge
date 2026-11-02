# dgural R1-only 叠 GT demo · 复现管线

Alpamayo-R1 预测轨迹 + GT 真值轨迹叠回 **dgural 公开真实帧**的全链复现脚本。
含两种形态：静态叠加图（已上线）与滚动预测视频（4 段 × 18.7s，资产已就绪、待发布）。
**线上 demo**：<https://ranpin.github.io/alpamayo-edge/r1-trajectory/>（三板块：演示 / 评测结果 / 核心方法）。

## 数据源

- **图像帧**：dgural 公开样本 `dgural/PhysicalAI-Autonomous-Vehicles-Sample`（NVIDIA PhysicalAI-AV 的公开 100-clip 子集，即 gated chunk 0）。
- **GT 轨迹 / 相机标定 / 逐帧时间戳**：gated `nvidia/PhysicalAI-Autonomous-Vehicles`（`labels/egomotion` / `calibration/` / camera zip 内 `*.timestamps.parquet`）。
- **帧↔egomotion 同步**用 gated 逐帧时间戳权威校准（HTTP Range 部分下载，非按 fps 推算）。

## 管线（按序）

| # | 脚本 | 跑在哪 | 作用 |
|---|---|---|---|
| 1 | `cand_select.py` | Mac | 候选 clip 普查（窗内转弯强度 + 帧亮度），定 6 clip |
| 2 | `dl_cams.py` | Mac | 下载 4 路相机视频（dgural 公开样本） |
| 3 | `fetch_ts_multi.py` | Mac | HTTP Range 部分下载 gated 相机 zip 的逐帧时间戳 → 精确帧↔egomotion 映射 |
| 4 | `extract_multi.py` | Mac | 抽 16 帧/clip（4 相机×4 时刻）+ 算 ego 历史 + GT 轨迹 + 建 R1 批量输入 |
| 5 | goat 推理（命令见下） | goat | TRTEdge `action_inference` 批量推理 6 clip |
| 6 | `integrate_multi.py` | goat | `import compute_minade` 官方口径积分 `(accel,κ)`→BEV 轨迹 + ADE/FDE |
| 7 | `gen_web_assets.py` | Mac | 生成网页叠加图（GT绿+R1蓝，鱼眼标定投影，1280×720 JPG） |
| 8 | `build_page.py` | Mac | 生成 demo 页（DATA 驱动 → `docs/site/r1-trajectory/`） |
| 9 | `verify_r1t.py` / `verify_r1t_live.py` | Mac | Playwright 本地/线上验证（明暗双主题、DOM 计数、图片加载、零 console 错误） |

## goat R1 推理命令（步骤 5）

```bash
A=/home/nvidia/chenrunbin/alpamayo-edge; T=$A/trtedge-0.10.1; GD=$A/eval_data/dgural_multi
cd $T
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
timeout 600 ./build_orin/examples/multimodal/action_inference \
    --engineDir $A/engines/alpamayo_r1/llm \
    --multimodalEngineDir $A/engines/alpamayo_r1 \
    --checkpointDir $A/models/Alpamayo-R1-10B \
    --inputFile $GD/r1_multi_in.json \
    --outputFile $GD/r1_multi_out.json \
    --noiseSeed 5
```

输入 `r1_multi_in.json` = 6 个 request（每个 16 帧 camera-major + 16 点 ego 历史 + temperature=0）；
三引擎（llm/visual/action）建于 Orin sm_87，action 专家用分解注意力补丁打穿 TRT≥11 算子墙（见根 README Phase R）。

## 工作目录约定

脚本内联路径是本次会话的工作目录 `/tmp/cdd_poc`（静态版）/ `/tmp/r1vid/{roll,ego,calib}`（滚动版）——macOS 会定期清理——与仓库/goat 绝对路径；
复现时请把各脚本里的这些路径改成你的工作目录/仓库位置。中间数据（`clip_frame_map.json` /
`multi_clip_data.json` / `r1_multi_pred.json` 等）不入库，按上表脚本重新生成。
gated 数据下载脚本（`dl_cams.py` / `fetch_ts_multi.py` / `cand_select.py`）需环境变量
`HF_TOKEN`（Hugging Face 访问令牌，`export HF_TOKEN=...`，勿硬编码进脚本）。

## 诚实口径（与 demo 页一致）

- 单噪声种子（seed=5、temperature=0）确定性推理，**非**官方 minADE6（6 采样取最优）。
- 6 clip 均为「6.4s 预测窗内明显转弯」的困难样本、非随机抽样，误差天然高于官方全集。
- dgural 样本 README 明示「不适合 benchmarking / 统计分析」——数字仅为逐例行为展示，不构成代表性精度结论。
- 官方 Alpamayo-R1 在 644 clip 报告 minADE6@6.4s≈0.79m，口径不同，不可与本 demo 直接比较。

## 投影要点

- 相机是**鱼眼多项式模型**（`fw_poly` 角度→半径，非针孔），标定取 gated `calibration/` 该 clip 的实测内参/外参。
- GT 带真实路面高程 z；R1 的 2D 输出按 GT 路面高程 `z(x)` 插值赋 z，使两者都贴合路面。
- 详细配方见记忆 `physicalai-av-fisheye-calibration`（内参 schema + 投影公式 + 坐标系自检）。

## 滚动预测视频版（2026-11-02）

把静态叠加图扩展成「边前进边预测」：Alpamayo-R1-10B 每 6s（=实测单次推理耗时，稳态 ~5.3s）重预测一次，
4 clip × 3 锚点（t₀=1.5/7.5/13.5s）= 12 条滚动预测，每段 ~18.7s（dgural 视频仅 ~20s）。
绿 GT 与蓝 R1 **同锚 t₀**（同窗 [t₀, t₀+6.4]，重叠=精度），随车一起被吃掉、每 6s 一起刷新。
首版用 oracle 绿（从当前帧 t 的实时未来），蓝尖会落后绿尖=预测年龄（0→6s、刷新归零），观感读作延迟，弃用。
CoT 不叠入视频：CoT 文本与轨迹头方向可能解耦（ecafce84_t13.5 CoT 说 "Turn right"，轨迹与 GT 均左转）。

### 管线（按序）

| # | 脚本 | 跑在哪 | 作用 |
|---|---|---|---|
| 1 | `fetch_ts_roll.py` | Mac | HTTP Range 取 4 clip 逐帧时间戳（同 `fetch_ts_multi.py` 配方） |
| 2 | `extract_roll.py` | Mac | 每锚点抽 4 相机×4 帧 + 16 点 ego 历史 + 64 点 GT → 建 12 requests |
| 3 | goat 推理 | goat | 同上 `action_inference` 命令，输入/输出改 `roll_in.json`/`roll_out.json`（12 requests） |
| 4 | `integrate_roll.py` | Mac | numpy 移植积分（estimate_v0 + 单车模型）；**先对拍 goat torch 结果 6 clip（max\|Δ\|=1.3e-05m）才放行** |
| 5 | `render_roll.py` | Mac | 同锚世界锚定逐帧重投影 + HUD（速度、重预测闪烁、本段 ADE@6.4s、事件级标签）→ 4×18.7s mp4 |
| 6 | `verify_roll.py` / `diag_lag.py` | Mac | 投影对齐自检（锚点恒等 ~1e-13px + 路面落点方向）/ 落后诊断（同锚 vs oracle 绿） |

### 逐锚点误差（单 seed、开环 6.4s、无导航输入）

ADE@6.4s = 同窗 64 点预测与 GT 的平均欧氏距离；FDE = 6.4s 终点距离；终点差 = pred−gt（t₀ ego 系，纵向 + = 过伸、横向 + = 偏左）。

| clip | 场景 | t₀(s) | v₀(m/s) | ADE@6.4s(m) | FDE(m) | 终点纵差(m) | 终点横差(m) | 误差形态 |
|---|---|---|---|---|---|---|---|---|
| 0dd3eb7f | 直行跟车 | 1.5 | 6.7 | 2.38 | 6.43 | -6.4 | +0.5 | 纵向欠伸 |
| 0dd3eb7f | 直行跟车 | 7.5 | 9.0 | 1.04 | 2.05 | -2.0 | +0.3 | 基本贴合 |
| 0dd3eb7f | 直行跟车 | 13.5 | 11.7 | 1.19 | 4.35 | -4.3 | +0.3 | 纵向欠伸 |
| 1f7ef408 | 变道 | 1.5 | 15.2 | 2.66 | 3.59 | +2.7 | +2.4 | 小幅偏差 |
| 1f7ef408 | 变道 | 7.5 | 15.6 | 1.65 | 3.17 | -2.8 | +1.5 | 基本贴合 |
| 1f7ef408 | 变道 | 13.5 | 11.4 | 2.13 | 8.05 | +4.3 | +6.8 | 欠转（切弯） |
| 1fcf2fef | 右转 | 1.5 | 3.6 | 0.60 | 1.08 | +0.4 | +1.0 | 小幅偏差 |
| 1fcf2fef | 右转 | 7.5 | 3.4 | 2.42 | 8.08 | -4.1 | +6.9 | 欠转（切弯） |
| 1fcf2fef | 右转 | 13.5 | 6.7 | 1.19 | 2.99 | -3.0 | -0.2 | 基本贴合 |
| ecafce84 | 左转·红灯起停 | 1.5 | 0.0 | 0.88 | 5.65 | -5.6 | +0.0 | 纵向欠伸 |
| ecafce84 | 左转·红灯起停 | 7.5 | 3.5 | 0.88 | 4.73 | +3.3 | -3.3 | 欠转（切弯） |
| ecafce84 | 左转·红灯起停 | 13.5 | 8.2 | 2.16 | 3.06 | -2.0 | +2.3 | 小幅偏差 |

12 段平均：ADE@6.4s=1.60m、FDE=4.44m。

### 发现

1. **误差三形态**：欠转（切弯）3/12（均为转弯锚点，终点横差 3.3~6.9m，方向对但幅度欠估）；纵向欠伸 4/12（低估加速，含 ecafce84_t1.5 红灯预测停住而 GT 蠕行 5.5m）；基本贴合/小幅偏差 5/12。误差随时域增长（最大偏差多在 6.4s 终点端）。
2. **CoT 退化↔欠转相关**（与静态 demo 叙事一致）：两个最差欠转锚点的 CoT 都退化成泛化跟车——1fcf2fef_t7.5 右转中说 "Adjust speed to keep safe distance"（横差 +6.9m）、1f7ef408_t13.5 右转中说 "Keep distance to the lead vehicle"（+6.8m）；对照 1fcf2fef_t1.5 CoT 点出 "Turn right" → ADE@6.4s=0.60m 近乎贴合。
3. **排除管线系统偏差**：镜像特征（pred_lat≈−gt_lat）0/12；同一积分+GT 管线在 chunk-3116 复现官方基线（18 clips × 6 seeds minADE6=0.804m vs 官方 0.821m，95% CI 内，见 `eval/r1/minade_chunk3116.csv`）；numpy vs torch 对拍 1.3e-05m；渲染锚点恒等 ~1e-13px。几米偏移是真实单 seed 模型误差，非管线 bug。
4. **已知诚实瑕疵**：0dd3eb7f_t1.5 欠伸（FDE 6.43m）⇒ 视频里蓝线 t≈7s 被车提前吃完、刷新前 ~0.5s 消失，属预测误差可视化、非渲染 bug。

### 口径补充（在上述「诚实口径」之外）

- 每锚点单样本：官方 minADE 为 6-seed 采样取最小；seed 敏感性实测（`eval/r1/r1_seed_sensitivity.py`，W4）示 seed 单独贡献的 ADE 变化为 ego 历史的 3.84×，单样本天然差于 6-seed min。
- 开环 6.4s ≠ 闭环控制：真实驾驶靠高频重规划修正误差，R1 在 Orin 单次 ~5.7s（≈0.18Hz），在自己的预测时域内完不成一次修正——这是「事件级非实时」定位的直观证据，视频 HUD 已显著标注。
