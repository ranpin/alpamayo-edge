#!/bin/bash
# INT4 分辨率 Pareto 驱动：smoke(1080p 1 题) → 448 → 720p → 1080p 全量三档。
# smoke 不过（G2：视觉引擎拒收 12,240 图像 token 之类）则中止，不浪费整档时间。
A=/home/vision/chenrunbin/alpamayo-edge
cd "$A" || exit 1
S=scripts/run_pareto.sh
E=engines/int4_len16384
B=bench/ood_eval

python3 - << 'EOF'
import json
d = json.load(open('bench/ood_eval/ood_input_1080.json'))
d['requests'] = d['requests'][:1]
json.dump(d, open('bench/ood_eval/ood_input_smoke1080.json', 'w'), ensure_ascii=False)
print('smoke input: 1 request')
EOF

echo "=== SMOKE 1080p 1-request $(date +%T) ==="
bash $S int4_smoke1080 $E/llm $E/visual $B/ood_input_smoke1080.json $B/out_smoke1080.json \
  || { echo "SMOKE_FAILED — G2 不过，中止 Pareto；见 $B/run_pareto_int4_smoke1080.log"; exit 1; }
python3 -c "
import json
o = json.load(open('$B/out_smoke1080.json'))
r = o['responses'][0]
print('smoke output_text:', repr(r.get('output_text')), 'finish:', r.get('finish_reason'))
"

for tier in 448 720 1080; do
  case $tier in
    448)  IN=$B/ood_input_dog.json ;;
    720)  IN=$B/ood_input_720.json ;;
    1080) IN=$B/ood_input_1080.json ;;
  esac
  echo "=== TIER int4_$tier start $(date +%T) ==="
  bash $S int4_$tier $E/llm $E/visual $IN $B/out_pareto_int4_$tier.json \
    || echo "TIER int4_$tier FAILED rc=$? (继续下一档)"
done
echo "INT4_PARETO_ALL_DONE $(date +%T)"
