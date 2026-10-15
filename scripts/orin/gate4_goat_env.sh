#!/bin/bash
# 闸门 4 G4a/G4b（goat）：清磁盘 + alpamayo1.5 环境搭建。
# 磁盘：删 tarballs/ 40G —— 解包对应物已于 2026-10-10 逐一核实存在
#   （onnx/int4 6.8G、onnx/int8 9.9G、onnx/fp16 16G、engines/fp16 16G），
#   device_layout.md 明文"纯冗余，可删以回收 40G"。
# 环境：uv（astral.sh 可达）+ py3.12（uv 从 github releases 拉 standalone）
#   + torch 2.8.0 aarch64 CPU wheel（pypi.org 可达；CUDA wheel 源
#   pypi.jetson.org 与 developer.download.nvidia.com 均不可达 ⇒ 本闸门走 CPU）。
#   flash-attn 跳过（需 nvcc 编 CUDA 版；README 明示有 SDPA fallback）。
A=/home/nvidia/chenrunbin/alpamayo-edge
echo "=== 1. free disk $(date +%T) ==="
ls -lh $A/tarballs/
rm -f $A/tarballs/edge_artifacts_int4_awq.tgz $A/tarballs/edge_artifacts_int8_sq.tgz \
      $A/tarballs/edge_fp16_llm.tgz $A/tarballs/fp16_engine.tar
df -h /home | tail -1

echo "=== 2. install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version || { echo UV_INSTALL_FAILED; exit 1; }

echo "=== 3. clone alpamayo1.5 ==="
cd $A || exit 1
rm -rf alpamayo1.5
git clone --depth 1 https://github.com/NVlabs/alpamayo1.5.git || { echo CLONE_FAILED; exit 1; }
cd alpamayo1.5

echo "=== 4. venv py3.12 + sync (skip flash-attn) $(date +%T) ==="
uv venv a1_5_venv --python 3.12 || { echo VENV_FAILED; exit 1; }
source a1_5_venv/bin/activate
uv sync --active --no-install-package flash-attn || { echo SYNC_FAILED; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" || { echo TORCH_IMPORT_FAILED; exit 1; }
python -c "import physical_ai_av; print('physical_ai_av OK')" || echo PHYSAI_WARN
echo "G4B_DONE $(date +%T)"
