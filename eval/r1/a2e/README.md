# Alpamayo2-Edge 跨代臂 3 · 复现脚本

实验文档（结论 / 口径 / 环境偏差与补丁披露 / 诚实边界）见 [`../crossgen_a2e.md`](../crossgen_a2e.md)。

脚本顺序：

| # | 脚本 | 跑在哪 | 作用 |
|---|---|---|---|
| 1 | `fetch_a2e.py` | Mac | HTTP Range 部分下载 6 路相机 chunk_3116 zip，取 18 clips 的 mp4+timestamps → ffmpeg 抽 22 帧/clip（需 `~/.hf_token`，gated） |
| 2 | `a2e_build_env2.sh` | goat | py3.10 venv：本地 torch/torchvision wheel + 清华镜像装 transformers 5.17 等 + clone 官方包 |
| 3 | `official_pkg_orin.patch` | goat | 官方包（commit `02f55a7`）三处 Orin/py3.10 补丁，`git apply` 即用 |
| 4 | `a2e_run.py` | goat | 内存构造输入（镜像官方 loader 契约）+ 批量推理（官方 `run_data`，6 候选） |
| 5 | `a2e_score.py` | goat | minADE6@6.4s/@5s 评分（对照 `eval_data/gt_chunk3116.json`）→ CSV + 聚合 |
| 6 | `a2e_perf.py` | goat | 性能专项：三段拆解埋点 + 内存轮询 + 单候选对照（结果 `perf_results.json`，功耗原始采样 `perf_tegrastats.log`） |

dgural demo 四线对比臂（实验文档见 [`../dgural_demo/README.md`](../dgural_demo/README.md)「Alpamayo2-Edge 四线对比版」节）：

| # | 脚本 | 跑在哪 | 作用 |
|---|---|---|---|
| 7 | `fetch_dgural_a2e.py` | Mac | dgural 8 clips 取数：6 路相机 chunk_0000 zip（含 ZIP64 解析）+ 标定/egomotion/fw 视频/真实 nav → 14 锚点 × 22 帧 |
| 8 | `a2e_run_dgural.py` | goat | dgural 14 锚点推理 + best-of-6 选择（CoT 三层嵌套需递归展平）→ 两个预测 JSON |
| 9 | `gen_web_assets_a2e.py` | Mac | 四线静态叠加图（2-Edge 品红点线带白描边） |
| 10 | `render_roll_a2e.py` | Mac | 四线滚动视频 + HUD（t₀=1.5s 锚无 2-Edge 线，如实缺省） |
| 11 | `gen_contactsheet_a2e.py` | Mac | 联络表重建 + 4 视频海报（t=9s 帧） |

⚠️ 输入帧 / GT / nav 来自 gated + license:other 数据集：只在机上跑，不入库。
脚本内联路径为本次会话工作目录（`/tmp/a2e`、`/home/nvidia/chenrunbin/a2e`），复现时改成你的位置。
