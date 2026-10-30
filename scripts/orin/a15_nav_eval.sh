#!/bin/bash
# Alpamayo-1.5 6-seed 推理 + merge + compute_minade（goat）。默认跑 nav 变体，可传参跑无nav 基线。
# 用法：a15_nav_eval.sh [input.json] [out_prefix] [csv_out]
#   nav 变体（默认）：a15_nav_eval.sh
#   无nav 基线      ：a15_nav_eval.sh io/r1_eval_in18.json a15_eval eval_data/minade_chunk3116_a15.csv
# 前置：1.5 引擎已 repack（repack_alpamayo15_to_r1.py）+ 拷好 R1 tokenizer 三件套
#       （见 eval/r1/crossgen_a15.md §二）；nav 变体还需先跑 derive_a15_oracle_nav.py 建 io/r1_eval_in18_nav.json。
# 口径与 R1 基线（eval/r1/README.md §五）一致：同 18 clips × 6 seeds、同 tokenizer、同 compute_minade。
set -u
A=/home/nvidia/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1
CKPT=$A/models/Alpamayo-1.5-10B
ENG=$A/engines/alpamayo15_r1fmt
PY=$T/export_venv/bin/python
IN=${1:-$A/io/r1_eval_in18_nav.json}
PFX=${2:-a15_nav_eval}
CSV=${3:-$A/eval_data/minade_chunk3116_a15_nav.csv}
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
cd $T || exit 1
mkdir -p $A/logs
exec > $A/logs/${PFX}.log 2>&1
echo "START $(date +%T) in=$IN pfx=$PFX csv=$CSV"
for s in 0 1 2 3 4 5; do
  timeout 1200 ./build_orin/examples/multimodal/action_inference --engineDir $ENG/llm --multimodalEngineDir $ENG --checkpointDir $CKPT --inputFile $IN --outputFile $A/io/${PFX}_s$s.json --noiseSeed $s
  rc=$?
  ok=$(python3 -c "import json; d=json.load(open('$A/io/${PFX}_s$s.json')); r=d.get('responses',[]); print(sum(1 for x in r if x.get('output_trajectory')), len(r))" 2>/dev/null)
  echo "seed $s exit=$rc traj_ok/total: $ok $(date +%T)"
done
echo "=== merge $(date +%T) ==="
python3 -c "
import json
A='$A'; PFX='$PFX'; IN='$IN'
resps=[]
for s in range(6):
    resps.extend(json.load(open(f'{A}/io/{PFX}_s{s}.json'))['responses'])
json.dump({'input_file':IN,'responses':resps}, open(f'{A}/io/{PFX}_output.json','w'))
print('merged responses:', len(resps))
"
echo "=== compute_minade $(date +%T) ==="
# --input 复用 r1_eval_input108.json：compute_minade 只用其 clip id/顺序（zip 配对响应），不用文本，nav 与无nav 通用
$PY $T/examples/accuracy/scripts/compute_minade.py --input $A/io/r1_eval_input108.json --output $A/io/${PFX}_output.json --gt $A/eval_data/gt_chunk3116.json --num_traj_samples 6 --csv_out $CSV
echo "DONE $(date +%T)"
