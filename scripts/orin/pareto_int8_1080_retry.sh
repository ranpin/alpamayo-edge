#!/bin/bash
# INT8 1080p 第二次尝试：KV 预留 32768 → 16448 token（精确省 2.41 GB 运行时内存，
# 147,456 B/token = 36 层 × 8 KV 头 × 128 维 × K+V × fp16；不影响计算，只影响预留）。
#
# 第一次（KV32768，2026-10-10 15:47–16:14）死于 ~140/184 题：tegrastats RAM 实测
# 从第一题起就在 28.6 GB 边缘（比 INT4 1080p 的 25.1 GB 高 3.5 GB ≈ 引擎体积差），
# 无泄漏（全程平稳），最后尖峰 30.4 GB 超物理上限被 OOM 杀。
# 引擎本体保留在 engines/int8_len16384（KV32768，运行时 OOM 的实物证据），
# 新引擎 engines/int8_len16384_kv16448 用于本次重试。
A=/home/vision/chenrunbin/alpamayo-edge
cd "$A" || exit 1
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH

echo "=== rebuild int8_len16384_kv16448 $(date +%T) ==="
( while true; do free -m | awk '/Mem/{print $3}'; sleep 2; done > /tmp/mem_int8_kv16448.log ) &
MON=$!
/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_build \
  --onnxDir "$A/onnx/int8/Cosmos-8B-int8_sq-onnx/llm" --engineDir "$A/engines/int8_len16384_kv16448/llm" \
  --maxInputLen 16384 --maxBatchSize 1 --maxKVCacheCapacity 16448 \
  > "$A/logs/build_int8_len16384_kv16448.log" 2>&1
rc=$?
kill $MON 2>/dev/null
echo "build exit=$rc peak=$(sort -n /tmp/mem_int8_kv16448.log | tail -1)MB $(date +%T)"
if [ $rc -ne 0 ]; then echo BUILD_FAILED; tail -5 "$A/logs/build_int8_len16384_kv16448.log"; exit 1; fi
du -sh "$A/engines/int8_len16384_kv16448/llm"

echo "=== int8_1080 retry $(date +%T) ==="
bash scripts/run_pareto.sh int8_1080 engines/int8_len16384_kv16448/llm engines/int4_len16384/visual \
  bench/ood_eval/ood_input_1080.json bench/ood_eval/out_pareto_int8_1080.json
echo "INT8_1080_RETRY_DONE $(date +%T)"
