# Phase V · V0 侦察备忘（2026-10-15）

> 闸门 V0 的产出。结论先行：**Thor fork 路径可行，移植面有界**；D0（CPU 保底 demo）的数据依赖
> 有 Kaggle workaround，不被 Mac 的 HF 网络问题卡死。

## 1. fork 基本盘（kimsunguk0/tensorrt-edge-llm-for-alpamayo）

| 项 | 实测 |
|---|---|
| 许可证 | **Apache-2.0**（与 TRTEdge 基座一致）⇒ 可移植、可再分发（保留声明） |
| 基座 | TRTEdge **v0.6.0**（我们 dog 上是 **v0.9.0**，commit 1ac0f2b）⇒ 需版本适配 |
| 最后更新 | 2026-08-26（fork HEAD 9512bcb 2026-06-29 "need to refactoring"） |
| 目标硬件 | Jetson **Thor**（引擎文件名 `*_mxfp8_thor.plan`、`alpa1.5_visual_fp8_rebuild`）|

## 2. 移植面（已定位，规模有界）

**C++ 自定义模块（4 个，共 ~2,400 行，自包含度高）**：
- `cpp/runtime/alpamayoFmRuntime.{h,cpp}`（1,532 行）——FM one-step 引擎的 TRT 运行时；
  依赖仅 `runtime/llmRuntimeUtils.h` + NvInfer + STL
- `cpp/runtime/alpamayoPostVlmRuntime.{h,cpp}`（733 行）——VLM 后编排（CoT→traj token→FM）；
  依赖 `runtime/llmInferenceRuntime.h` + nlohmann/json
- `cpp/common/deltaTrajectoryTokenizer.{h,cpp}`（147 行）——轨迹 delta 分词器
- `cpp/common/npyUtils.{h,cpp}`——npy 读写

**llm_inference.cpp 挂钩**：新增 CLI（`--fmEngine`、`--alpamayoPostVlmRuntime`、
`--alpamayoFmUsePrefillKv`（prefill KV 直喂 FM、跳过 CoT decode 的优化路径）、`--alpamayoNavCfg`、
`--persistentServer`（引擎常驻 + stdin JSON 命令流））+ 编排调用。

**Python 侧**：`tensorrt_edgellm/onnx_export/fm_export.py` + `scripts/export_fm.py`（FM one-step
ONNX 导出，V3 直接复用）；`jetson_live_infer_alpamayo15.py` + `planner_live/`（live 服务层，
demo 可参考但非必需）。

**文档**：`docs/live_dataset_alpamayo_integration_summary.md`（韩文；request bank 格式 =
frames + ego history + GNSS 时间戳 → 输入 JSON，chunk 化组织）+ `docs/alpamayo_portfolio_report_*.md`。

## 3. 主要风险（按大小）

1. **v0.6.0 → v0.9.0 API 漂移**：集中在 `llmInferenceRuntime.h` / `llmRuntimeUtils.h` 的接口
   （KV cache 访问、runtime 生命周期）。V4 开工前先做两版头文件 diff（V0.5）。
2. **dtype 差异**：fork 在 Thor 上用 **mxfp8**（FM 引擎与 visual）；Orin sm_87 无 fp8 ⇒
   FM 引擎与 visual 都要走 **FP16**（fm_export.py 的 dtype 选项待核；visual 我们本就有 FP16 先例）。
   体积账：FM FP16 ≈ 4.6G（2.3B），仍装得下。
3. **VLM 词表**：Alpamayo VLM = CR2 backbone + traj/special token 扩展 ⇒ ONNX 导出（V1）时
   embedding/lm_head 尺寸非标准值，Kaggle 管线（AWQ/SQ 校准）对 resize 后词表的兼容性待验证。

## 4. D0（CPU 保底 demo）数据路径

**阻塞**：Mac → huggingface.co 自 2026-10-15 晨起不可达（HTTP 000/超时；gh、pypi、scm、Orin 均正常
⇒ 疑似 DNS/路由对 HF 单独失效）。ego 历史 + GT 轨迹在 gated 全集 parquet，公开 sample 只有 mp4。

**Workaround（不需要修网络）**：Kaggle kernel 拉数据——Kaggle 侧有外网、HF token 在私有 kernel
`mamihlapinatapai/alpamayo-edge-export` 的 `_HF_FALLBACK`（2026-07-31 验证过 PhysicalAI-AV 访问权），
大产物拉回管线 `cloud/pull_kernel_output.py` 现成。流程：kernel 拉 N clips 的 ego/GT parquet +
（dgural sample 的）4 路 mp4 → kernel output → Mac → scp goat。

**D0 其余依赖已就绪**：goat 上 Alpamayo 权重（22G，字节校验过）+ alpamayo1.5 repo + venv +
gate4 前向脚本（改多 clip 循环即可）；渲染（轨迹叠加 + BEV）在 Mac 本地做（matplotlib/ffmpeg）。

## 5. V0 判定

**通过（有条件）**：许可证 ✓、移植面有界 ✓、数据路径有 workaround ✓。
条件项：V0.5（两版 TRTEdge 头文件 diff）在 V4 开工前完成；V1 先验证词表扩展过导出管线。
按闸门表推进，D0 与 V1 可并行（互不依赖）。
