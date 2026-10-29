# dgural R1-only 叠 GT demo · 复现管线

Alpamayo-R1 预测轨迹 + GT 真值轨迹叠回 **dgural 公开真实帧**的全链复现脚本。
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

脚本内联路径是本次会话的工作目录 `/tmp/cdd_poc`（macOS 会定期清理）与仓库/goat 绝对路径；
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
