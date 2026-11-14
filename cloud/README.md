# 一次性导出/量化（免费云）—— Alpamayo 家族篇

8GB 3070 装不下 10B，这一步在免费云一次性完成，产物（ONNX，几 GB）下载后 scp 到 Orin。

> ✅ **实际采用并跑通的脚本**：[`kaggle_r1_vlm_int4.py`](kaggle_r1_vlm_int4.py)——把 R1 的 `vlm.*`（750 张量，
> 剥前缀后名字集合==CR2-8B、形状除 vocab 155697 外全同）repack 成 qwen3_vl 格式 checkpoint（CR2 config 模板改 vocab_size
> + Qwen3-VL-8B-Instruct 公开 tokenizer，**全程免 HF token**：R1 gated=False），钉 TRTEdge **v0.10.1**（与 goat 运行时同版；
> 补丁 A needle 须含 0.10.1 新增的 `low_cpu_mem_usage=True,` 行），export 带 `--max-kv-cache-capacity 4096`（与已部署引擎一致）。
> v2 起不安装 TRTEdge 包本体（scikit-build-core metadata 在 Kaggle py3.13 镜像崩、`pip -q` 吞报错），直装 tools-extra 依赖 pin
> + PYTHONPATH 源码树运行，entry point 经 `python -c` 调用；[kernel-metadata.json](kernel-metadata.json) 已入库，`kaggle kernels push -p cloud/` 即用。
> [`av/`](av/) 是 AV 域校准对照臂（kernel `mamihlapinatapai/r1-vlm-int4-av`，INT4-AV 与 INT4 通用校准的
> 精度对照，结论：两臂差全在噪声内 ⇒ 校准域不敏感）。

本脚本复用的 CR2 量化管线（`kaggle_cosmos_only.py` 及三处 TRTEdge `quantize.py` 补丁）已迁至
[cosmos-edge 仓 cloud/](https://github.com/ranpin/cosmos-edge/tree/main/cloud)。HF token 走**环境变量**，勿硬编码。

## 交给 Orin

产物 scp 到 goat（R1 全链在 64GB orin-goat 跑）：设备布局见
[docs/device_layout.md](https://github.com/ranpin/alpamayo-edge/blob/main/docs/device_layout.md)。

## ⚠️ 风险

- **HF 门控**：R1 gated=False 免 token；PhysicalAI-AV 是门控仓库，国内用 `HF_ENDPOINT=hf-mirror.com` + token。
- **Kaggle 元数据**：必须 `"machine_shape":"NvidiaTeslaT4"` 才拿到 T4×2（默认单 P100 会崩）。
