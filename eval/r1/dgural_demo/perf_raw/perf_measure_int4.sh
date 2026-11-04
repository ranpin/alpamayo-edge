#!/bin/bash
# INT4 端到端性能实测——与 perf_measure.sh（FP16）逐字同构，唯一差异：
# 引擎 = alpamayo_r1_int4（INT4 llm + FP16 visual/action），不传 --checkpointDir（权重已烘进引擎，
# ExternalWeightManager bindings 空即早返回，见 int4_backbone.md §三）。同输入（dgural_1f7ef408 单 clip
# + dgural_multi 6 clip）、同 --warmup 1 --dumpProfile、同 seed 5/temp0。
# 「单请求端到端」口径 = 日志 warmup 后 benchmark 段时间戳差（Starting actual benchmark runs →
# Processing complete），与 FP16 表 3 同口径；三段拆解来自 profile JSON 的 stages。
set -u
A=/home/nvidia/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1
ENG=$A/engines/alpamayo_r1_int4
P=$A/perf_int4
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export HF_ENDPOINT=https://hf-mirror.com
cd $T || exit 1
AI=./build_orin/examples/multimodal/action_inference
SINGLE=$A/eval_data/dgural_1f7ef408/r1_dgural_in.json
BATCH=$A/eval_data/dgural_multi/r1_multi_in.json
mkdir -p $P
for f in "$SINGLE" "$BATCH" "$ENG/llm/llm.engine"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done
AVAIL=$(free -m | awk '/Mem/{print $7}')
echo "PERF_INT4_START $(date +%T) avail_RAM=${AVAIL}MB"
[ "$AVAIL" -lt 14000 ] && { echo "ABORT: avail RAM ${AVAIL}MB < 14G"; exit 1; }
for i in 1 2 3; do
  t0=$EPOCHREALTIME
  $AI --engineDir $ENG/llm --multimodalEngineDir $ENG \
    --inputFile $SINGLE --outputFile $P/single_$i.json --noiseSeed 5 \
    --warmup 1 --dumpProfile --profileOutputFile $P/profile_single_$i.json \
    > $P/single_$i.log 2>&1
  rc=$?; t1=$EPOCHREALTIME
  echo "single$i exit=$rc wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f",b-a}')s"
done
echo "batch6 start $(date +%T)"
tegrastats --interval 1000 > $P/tegrastats_batch.log 2>&1 &
TP=$!
sleep 2
t0=$EPOCHREALTIME
$AI --engineDir $ENG/llm --multimodalEngineDir $ENG \
  --inputFile $BATCH --outputFile $P/batch6.json --noiseSeed 5 \
  > $P/batch6.log 2>&1
rc=$?; t1=$EPOCHREALTIME
kill $TP 2>/dev/null
echo "batch6 exit=$rc wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f",b-a}')s"
echo "PERF_INT4_DONE $(date +%T)"
