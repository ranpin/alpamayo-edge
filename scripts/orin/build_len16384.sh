#!/bin/bash
# 分辨率 Pareto 用：建 maxInputLen=16384 / batch=1 / KV=32768 引擎 + maxImageTokens=16384 视觉引擎。
# 老引擎（int4/int8/*_len4096）一律不动，保住已发布数字的可比性。
#
# 预注册闸门（2026-10-10，见 docs/next_plan.md P0-2）：
#   G1 int4_len16384 + visual_len16384 在 dog 建成（len4096 时 INT4 峰值 18.6G，
#      长度 4× 按实测斜率 +0.6G/4× 估计 ~19-21G < 29G 总内存）。
#      不过 → 改 goat（available 28G）构建后拷回（同 sm_87 + 同版 TRT，引擎互通）。
#   G3 int8_len16384 为「尝试性构建」：len4096 峰值实测 29.7G 几乎撞顶 32G 机，
#      len16384 外推 30-32G ⇒ 可能被 OOM killer 杀（exit 137，7 月 FP16 build 同模式，
#      杀的是 builder 自身、机器无恙）。OOM 也是实测结论：INT8 高分辨率档判为
#      build 内存墙阻塞，Pareto 的 INT8 轴只保留 448 档。故 int8 排在最后，
#      且启动前须确认 dog 上无其他大进程。
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
B=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm
V=/home/vision/TensorRT-Edge-LLM/build_orin/examples/multimodal
L=$A/logs; mkdir -p "$L"

run_build() {   # tag  cmd...
  tag=$1; shift
  echo "=== BUILD $tag start $(date +%T) ==="
  ( while true; do free -m | awk '/Mem/{print $3}'; sleep 2; done > /tmp/mem_$tag.log ) &
  MON=$!
  "$@" > "$L/build_$tag.log" 2>&1
  rc=$?
  kill $MON 2>/dev/null
  echo "  $tag exit=$rc  build峰值内存=$(sort -n /tmp/mem_$tag.log | tail -1) MB  用时到 $(date +%T)"
}

run_build int4_len16384 $B/llm_build \
  --onnxDir $A/onnx/int4/Cosmos-8B-int4_awq-onnx/llm --engineDir $A/engines/int4_len16384/llm \
  --maxInputLen 16384 --maxBatchSize 1 --maxKVCacheCapacity 32768
du -sh $A/engines/int4_len16384/llm

run_build visual_len16384 $V/visual_build \
  --onnxDir $A/onnx/int4/Cosmos-8B-int4_awq-onnx/visual --engineDir $A/engines/int4_len16384 \
  --maxImageTokens 16384 --maxImageTokensPerImage 2048
du -sh $A/engines/int4_len16384/visual

echo "INT4_BUILDS_DONE $(date +%T)"

if [ "$1" = "--with-int8" ]; then
  run_build int8_len16384 $B/llm_build \
    --onnxDir $A/onnx/int8/Cosmos-8B-int8_sq-onnx/llm --engineDir $A/engines/int8_len16384/llm \
    --maxInputLen 16384 --maxBatchSize 1 --maxKVCacheCapacity 32768
  du -sh $A/engines/int8_len16384/llm 2>/dev/null
fi

echo "ALL_BUILDS_DONE $(date +%T)"
