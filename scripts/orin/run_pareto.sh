#!/bin/bash
# 分辨率 Pareto 单档运行：llm_inference + tegrastats 功率窗口对齐采集。
# usage: run_pareto.sh <tag> <engineLlmDir> <visualDir> <inputJson> <outJson>
# 例:    run_pareto.sh int4_1080 engines/int4_len16384/llm engines/int4_len16384/visual \
#          bench/ood_eval/ood_input_1080.json bench/ood_eval/out_pareto_int4_1080.json
#
# 功率口径：tegrastats 在 run 日志出现第一条 "Processing vision inputs"（= 引擎加载完、
# 第一个请求开始处理）后才启动，llm_inference 退出即停 ⇒ 采样窗口 ≈ 纯处理阶段，
# 不含引擎加载（加载是磁盘/CPU 为主，会稀释平均功率）。
# 逐请求 E2E 从 run 日志的逐请求时间戳解析（每请求恰 1 条 vision inputs 行，
# 2026-10-09 n=184 日志已验证该结构），解析在 Mac 侧做。
TAG=$1; ENG=$2; VIS=$3; IN=$4; OUT=$5
A=/home/vision/chenrunbin/alpamayo-edge
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
LI=/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_inference
cd "$A" || exit 1
PW=$A/logs/pw_pareto_${TAG}.log
RUN=$A/bench/ood_eval/run_pareto_${TAG}.log
rm -f "$PW" "$RUN"

echo "PARETO $TAG launch $(date +%T) epoch=$(date +%s.%N)"
"$LI" --inputFile "$IN" --engineDir "$ENG" --multimodalEngineDir "$VIS" --outputFile "$OUT" > "$RUN" 2>&1 &
LIPID=$!

# 等第一个请求开始处理（最多 300 s，防引擎加载失败导致死等）
for _ in $(seq 1 600); do
  grep -q "Processing vision inputs" "$RUN" 2>/dev/null && break
  kill -0 $LIPID 2>/dev/null || break
  sleep 0.5
done
if ! kill -0 $LIPID 2>/dev/null; then
  echo "PARETO $TAG EARLY_EXIT llm_inference died before/at first request; tail:"
  tail -5 "$RUN"
  exit 1
fi
echo "PARETO $TAG proc_start epoch=$(date +%s.%N)"
tegrastats --interval 250 > "$PW" 2>/dev/null &
TSPID=$!

wait $LIPID; rc=$?
echo "PARETO $TAG proc_end epoch=$(date +%s.%N) exit=$rc"
kill $TSPID 2>/dev/null
grep -E "Processing complete" "$RUN" | tail -1
echo "PARETO $TAG pw_samples=$(wc -l < "$PW") DONE"
exit $rc
