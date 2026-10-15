#!/bin/bash
# INT8 侧分辨率 Pareto：
#   ① 448 档能耗复测（现有 engines/int8 + engines/int4/visual，带 tegrastats 功率窗口）
#      —— 2026-10-09 的 n=184 INT8 运行没采功率；顺带做贪心可复现性校验（同引擎同帧，
#      输出应与 orin_int8_outputs_n184.json 逐条一致）。
#   ② G3 尝试性构建 int8_len16384：INT4 len16384 实测峰值 25.7G（比 len4096 的 18.6G
#      高 7.1G），INT8 len4096 实测峰值 29.7G ⇒ 外推 ~37G > dog 总内存 29G，预期被
#      OOM killer 杀（exit 137，7 月 FP16 build 同模式：杀 builder 自身、机器无恙）。
#      OOM 也是实测结论：INT8 高分辨率轴被 build 内存墙挡住，写成结论而非外推。
#   ③ 若 ② 意外成功：补跑 INT8 720/1080 两档（视觉引擎用 int4_len16384/visual，fp16 共享）。
A=/home/vision/chenrunbin/alpamayo-edge
cd "$A" || exit 1
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH

echo "=== INT8 448 energy rerun $(date +%T) ==="
bash scripts/run_pareto.sh int8_448 engines/int8/llm engines/int4/visual \
  bench/ood_eval/ood_input_dog.json bench/ood_eval/out_pareto_int8_448.json \
  || echo "int8_448 rerun FAILED rc=$?"

echo "=== G3: INT8 len16384 build attempt $(date +%T) ==="
( while true; do free -m | awk '/Mem/{print $3}'; sleep 2; done > /tmp/mem_int8_len16384.log ) &
MON=$!
/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_build \
  --onnxDir "$A/onnx/int8/Cosmos-8B-int8_sq-onnx/llm" --engineDir "$A/engines/int8_len16384/llm" \
  --maxInputLen 16384 --maxBatchSize 1 --maxKVCacheCapacity 32768 \
  > "$A/logs/build_int8_len16384.log" 2>&1
rc=$?
kill $MON 2>/dev/null
echo "int8_len16384 exit=$rc peak=$(sort -n /tmp/mem_int8_len16384.log | tail -1)MB $(date +%T)"

if [ $rc -eq 0 ]; then
  du -sh "$A/engines/int8_len16384/llm"
  echo "=== INT8 720/1080 tiers $(date +%T) ==="
  bash scripts/run_pareto.sh int8_720 engines/int8_len16384/llm engines/int4_len16384/visual \
    bench/ood_eval/ood_input_720.json bench/ood_eval/out_pareto_int8_720.json \
    || echo "int8_720 FAILED rc=$?"
  bash scripts/run_pareto.sh int8_1080 engines/int8_len16384/llm engines/int4_len16384/visual \
    bench/ood_eval/ood_input_1080.json bench/ood_eval/out_pareto_int8_1080.json \
    || echo "int8_1080 FAILED rc=$?"
else
  echo "G3 NOT PASSED: int8_len16384 build exit=$rc (预期 OOM=137)；INT8 轴止于 448 档"
  rm -rf "$A/engines/int8_len16384"   # 清掉半截产物
fi
echo "INT8_PARETO_DRIVER_DONE $(date +%T)"
