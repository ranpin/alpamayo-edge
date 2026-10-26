#!/bin/bash
# Phase R：Alpamayo-R1 action 专家「只导 + 只建」最小验证全链（goat）。
#
# 目的：回答「Orin sm_87 / JetPack 6.2 / TRT 10.7 上，Alpamayo-R1 的 turnkey
# action 路径到底跑不跑得起来」。**不碰 10B llm 引擎构建**（FP16-only 且内存风险高）。
#
# 2026-10-26 实测结果（本脚本各步骤的判据即来自这次跑通）：
#   export : 峰值 RSS 10.5 G、最低 MemAvailable 37 G、约 5 分钟
#            → model.onnx 6,314,629 B（图）+ model.onnx.data 4,561,240,064 B（外置权重）
#   census : Attention=0 RotaryEmbedding=0 TensorScatter=0 KVCacheUpdate=0（墙已绕过）
#            ScatterElements=72 Softmax=36（= num_hidden_layers 36，每层 k/v 各一次 scatter）
#   build  : Engine generation 89.9 s、峰值 CPU 12088 MiB / GPU 4349 MiB、0 error
#            → action.engine 4,565,059,244 B
#
# 前置：先跑 apply_alpamayo_action_decomposed_patch.py 打补丁，
#       并用 test_alpamayo_action_decomposed.py 对拍（16/16 必须全过）再导出。
set -u
A=/home/nvidia/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1
CKPT=$A/models/Alpamayo-R1-10B
ONNX=$A/onnx/alpamayo_r1
ENG=$A/engines/alpamayo_r1

export PATH=/usr/local/cuda/bin:$PATH
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export USE_DECOMPOSED_ALPAMAYO=1     # 分解补丁门控（独立于 cosmos3 的 USE_DECOMPOSED_POLICY）
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
PY=$T/export_venv/bin/python

mkdir -p "$A/logs" "$A/onnx" "$A/engines"
cd "$T" || exit 1

# --- [0] 权重（21 G，5 shard）。hf CLI 不在系统 PATH，只在 export_venv/bin ---
if [ ! -f "$CKPT/model-00005-of-00005.safetensors" ]; then
  echo "=== [0] download Alpamayo-R1-10B $(date +%T) ==="
  "$T/export_venv/bin/hf" download nvidia/Alpamayo-R1-10B --local-dir "$CKPT" \
      > "$A/logs/r1_download.log" 2>&1
  echo "  download exit=$?"
fi
grep -o '"model_type": *"[^"]*"' "$CKPT/config.json"   # 必须是 alpamayo_r1，否则 export 会拒

# --- [1] 补丁 + 数值对拍（对拍不过就不要往下走，导出/建引擎很贵）---
echo "=== [1] patch + numerical validation $(date +%T) ==="
$PY "$A/scripts/apply_alpamayo_action_decomposed_patch.py"
USE_DECOMPOSED_ALPAMAYO=1 $PY "$A/scripts/test_alpamayo_action_decomposed.py" \
    || { echo "对拍未全过，中止"; exit 1; }

# --- [2] 只导 action（--skip-llm --skip-visual）---
# --max-kv-cache-capacity 必须与后续 llm_build 的 --maxKVCacheCapacity 一致。
echo "=== [2] export action-only $(date +%T) ==="
awk '/MemAvailable/{print "  MemAvailable before: "int($2/1024)" MB"}' /proc/meminfo
./export_venv/bin/tensorrt-edgellm-export "$CKPT" "$ONNX" \
    --skip-llm --skip-visual --max-kv-cache-capacity 4096 \
    > "$A/logs/r1_export.log" 2>&1
echo "  export exit=$? $(date +%T)"
tail -6 "$A/logs/r1_export.log"

# --- [3] 算子普查：证明墙已绕过、分解确实落进图里 ---
echo "=== [3] ONNX op census $(date +%T) ==="
$PY - "$ONNX/action/model.onnx" <<'PYEOF'
import collections, sys, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
c = collections.Counter(n.op_type for n in m.graph.node)
wall = {k: c.get(k, 0) for k in
        ("Attention", "RotaryEmbedding", "TensorScatter", "KVCacheUpdate")}
print("  total nodes:", sum(c.values()))
print("  TRT-11-only wall ops (must all be 0):", wall)
print("  decomposed evidence: ScatterElements=%d Softmax=%d MatMul=%d"
      % (c.get("ScatterElements", 0), c.get("Softmax", 0), c.get("MatMul", 0)))
assert sum(wall.values()) == 0, "墙算子仍在图里，补丁未生效"
PYEOF
[ $? -ne 0 ] && { echo "census 断言失败，中止"; exit 1; }

# --- [4] 只建 action 引擎 ---
echo "=== [4] action_build $(date +%T) ==="
awk '/MemAvailable/{print "  MemAvailable before: "int($2/1024)" MB"}' /proc/meminfo
./build_orin/examples/multimodal/action_build \
    --onnxDir "$ONNX/action" --engineDir "$ENG" --maxBatchSize 6 \
    > "$A/logs/r1_action_build.log" 2>&1
echo "  build exit=$? $(date +%T)"
grep -E "Engine generation completed|Peak memory|built successfully" \
    "$A/logs/r1_action_build.log"

echo "=== artifacts ==="
stat -c "  %s  %n" "$ONNX/action/model.onnx" "$ONNX/action/model.onnx.data" \
                   "$ENG/action/action.engine" 2>/dev/null
df -h / | tail -1
echo "R1_ACTION_MINVERIFY_DONE $(date +%T)"
