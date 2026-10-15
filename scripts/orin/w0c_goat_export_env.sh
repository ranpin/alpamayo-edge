#!/bin/bash
# W0c-env：goat 上搭 TRTEdge v0.10.1 的 python 导出环境（export 是 CPU 任务，
# 依赖硬 pin：torch==2.13.0 / transformers==5.14.1 / onnx==1.19.0，pypi 可达）。
# 与 a1_5_venv（Alpamayo gate4 用）隔离，避免依赖冲突。
A=/home/nvidia/chenrunbin/alpamayo-edge
cd "$A" || exit 1
export PATH="$HOME/.local/bin:$PATH"

echo "=== clone TRTEdge v0.10.1 $(date +%T) ==="
if [ -d trtedge-0.10.1/.git ]; then echo "clone exists, reuse"; else
  rm -rf trtedge-0.10.1
  git clone --depth 1 --branch v0.10.1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git trtedge-0.10.1 \
    || { echo CLONE_FAILED; exit 1; }
fi
cd trtedge-0.10.1 || exit 1

echo "=== venv + install [export] extras $(date +%T) ==="
uv venv export_venv --python 3.12 || { echo VENV_FAILED; exit 1; }
source export_venv/bin/activate
uv pip install -e ".[export]" > "$A/logs/w0c_pip_install.log" 2>&1 \
  || { echo INSTALL_FAILED; tail -15 "$A/logs/w0c_pip_install.log"; exit 1; }

python -c "import torch, transformers, onnx; print('torch', torch.__version__, '| transformers', transformers.__version__, '| onnx', onnx.__version__)" \
  || { echo IMPORT_FAILED; exit 1; }
which tensorrt-edgellm-export && tensorrt-edgellm-export --help 2>&1 | head -8
echo "W0C_ENV_DONE $(date +%T)"
