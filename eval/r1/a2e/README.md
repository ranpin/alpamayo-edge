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

⚠️ 输入帧 / GT / nav 来自 gated + license:other 数据集：只在机上跑，不入库。
脚本内联路径为本次会话工作目录（`/tmp/a2e`、`/home/nvidia/chenrunbin/a2e`），复现时改成你的位置。
