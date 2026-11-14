#!/bin/bash
# goat 环境构建（续）：torch/torchvision 用 Mac 并行下载校验过的本地 wheel，其余走清华镜像
cd /home/nvidia/chenrunbin/a2e || exit 1
venv/bin/pip cache purge 2>/dev/null | true
venv/bin/pip install --no-cache-dir wheels/torch-2.8.0-cp310-cp310-linux_aarch64.whl wheels/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl || exit 1
venv/bin/pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==5.17.0 "accelerate>=1.12.0" "huggingface-hub>=1.0,<2" "safetensors>=0.8.0" \
  "einops>=0.8.1" "hydra-core>=1.3.2" "numpy>=2.0" "pillow>=11.0" "scipy==1.15.3" "colorlog>=6.0" || exit 1
[ -d alpamayo2-edge ] || git clone --depth 1 https://github.com/NVlabs/alpamayo2-edge.git || exit 1
cd alpamayo2-edge || exit 1
sed -i 's/requires-python = ">=3.12,<3.13"/requires-python = ">=3.10"/' pyproject.toml
../venv/bin/pip install --no-cache-dir -e . --no-deps || exit 1
../venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
../venv/bin/python -c "import transformers; print('transformers', transformers.__version__)" || exit 1
../venv/bin/python -c "import alpamayo2_edge; from alpamayo2_edge.sample import validate_sample; print('pkg OK')" || echo PKG_IMPORT_FAIL
../venv/bin/pip cache purge 2>/dev/null
echo BUILD2_DONE
