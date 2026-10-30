#!/bin/bash
# R1 性能测量（2026-10-30，goat）：r1-trajectory demo 页「表 2 · Orin 部署与性能」的数据来源。
# 素材 = dgural 公开帧（r1_dgural_in.json 单请求 + r1_multi_in.json 6-clip 批量），无 license 纠缠。
# 实测结果：单请求 E2E 5.0s（warmup 后 ×3）；TRT profile 分段 prefill 1.73s/3006tok（1740 tok/s）
#   + CoT 解码 1.82s/19tok（10.4 tok/s）+ action 扩散 0.60s；批量 5.7s/请求；推理功耗 32.6W
#   （VDD_GPU_SOC 31.7 + VDD_CPU_CV 0.9，GR3D 82%，tegrastats 批量段均值）；峰值统一内存 25.5G
#   （cuda_free_memory_delta 口径；引擎驻留 19.5G = 14.15+1.08+4.25）。
# 🔴 R1 权重 21G 必须在场（action_inference 运行期需 --checkpointDir）；avail RAM < 22G 时拒跑。
set -u
A=/home/nvidia/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1
ENG=$A/engines/alpamayo_r1
CKPT=$A/models/Alpamayo-R1-10B
P=$A/perf
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export HF_ENDPOINT=https://hf-mirror.com
cd $T || exit 1
AI=./build_orin/examples/multimodal/action_inference
SINGLE=$A/eval_data/dgural_1f7ef408/r1_dgural_in.json
BATCH=$A/eval_data/dgural_multi/r1_multi_in.json
for f in "$SINGLE" "$BATCH" "$CKPT/config.json" "$ENG/llm/llm.engine"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done
AVAIL=$(free -m | awk '/Mem/{print $7}')
echo "PERF_START $(date +%T) avail_RAM=${AVAIL}MB"
[ "$AVAIL" -lt 22000 ] && { echo "ABORT: avail RAM ${AVAIL}MB < 22G"; exit 1; }
mkdir -p $P
for i in 1 2 3; do
  t0=$EPOCHREALTIME
  $AI --engineDir $ENG/llm --multimodalEngineDir $ENG --checkpointDir $CKPT \
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
$AI --engineDir $ENG/llm --multimodalEngineDir $ENG --checkpointDir $CKPT \
  --inputFile $BATCH --outputFile $P/batch6.json --noiseSeed 5 \
  > $P/batch6.log 2>&1
rc=$?; t1=$EPOCHREALTIME
kill $TP 2>/dev/null
echo "batch6 exit=$rc wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f",b-a}')s"
echo "PERF_DONE $(date +%T)"
