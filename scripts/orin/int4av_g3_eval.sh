#!/bin/bash
# G3'：R1 backbone INT4 AWQ 的 AV 域校准对照臂精度评测（chunk 3116，18 clips × 6 seeds，
# 与 FP16 基线 0.804m / MMMU 校准臂 0.800m 同口径）。
# 与 int4_g3_eval.sh 逐字同构；唯一差异：引擎=alpamayo_r1_int4_av（AV 域校准的 INT4 llm
# + FP16 visual/action）。预注册问题：校准数据域（MMMU 通用多模态 vs dgural 公开 AV 帧）
# 是否改变「量化近乎免费」结论。判据沿用：ΔminADE6 < 0.1m 近乎免费 / 0.1-0.5m 量化代价 /
# > 0.5m 不可用（对照 FP16 0.80401m）。
set -u
A=/home/nvidia/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1
ENG=$A/engines/alpamayo_r1_int4_av
PY=$T/export_venv/bin/python
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
cd $T || exit 1
exec > $A/logs/int4av_g3_eval.log 2>&1
echo "G3AV_START $(date +%T)"
for s in 0 1 2 3 4 5; do
  timeout 1200 ./build_orin/examples/multimodal/action_inference --engineDir $ENG/llm --multimodalEngineDir $ENG --inputFile $A/io/r1_eval_in18.json --outputFile $A/io/int4av_eval_s$s.json --noiseSeed $s
  rc=$?
  ok=$(python3 -c "
import json
try:
    d=json.load(open('$A/io/int4av_eval_s$s.json')); r=d.get('responses',[])
    print(sum(1 for x in r if x.get('output_trajectory')), len(r))
except Exception as e: print('ERR', e)
" 2>/dev/null)
  echo "seed $s exit=$rc traj_ok/total: $ok $(date +%T)"
done
echo "=== merge + compute_minade $(date +%T) ==="
python3 - <<'EOF'
import json
A="/home/nvidia/chenrunbin/alpamayo-edge"
resps=[]
for s in range(6):
    d=json.load(open(f"{A}/io/int4av_eval_s{s}.json"))
    resps.extend(d["responses"])
json.dump({"input_file":"r1_eval_in18.json","responses":resps}, open(f"{A}/io/int4av_eval_output.json","w"))
print("merged responses:", len(resps))
EOF
$PY $T/examples/accuracy/scripts/compute_minade.py --input $A/io/r1_eval_input108.json --output $A/io/int4av_eval_output.json --gt $A/eval_data/gt_chunk3116.json --num_traj_samples 6 --csv_out $A/eval_data/minade_chunk3116_int4_av.csv
echo "G3AV_DONE $(date +%T)"
