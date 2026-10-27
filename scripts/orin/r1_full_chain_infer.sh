#!/bin/bash
# Phase R：Alpamayo-R1 全链推理（VLM + action 两阶段）+ seed 敏感性对照。
#
# 前置：三引擎已建成（见 r1_action_minverify.sh 建 action；llm/visual 见下），
#       且 R1 权重 21G 在场（action_inference 运行期需 --checkpointDir）。
#
# 2026-10-27 实测（goat，16 张 gate4 多相机帧 1920×1080）：
#   单次推理 ~6s（引擎加载另需 ~25s）；GPU 统一内存占用 19919 MiB
#   shared execution context 3740115456 B（base 406719488 / vision 3740115456 / action 158074880）
#   输出 output_text（VLM CoT）+ output_trajectory [64,2] = (accel, kappa) 原始归一化值
#
# ⚠️ 输入图像来自 gated + license:other 数据集，**只在机上跑，不得入库/上站**。
#    本脚本只写路径，不含图像数据。
#
# 🔴 两个必须知道的口径：
#   1) **temperature 必须设 0** 才能隔离 seed 轴——VLM 文本默认 temp=1.0 采样，
#      生成的 CoT token 会进 action 专家的 conditioning，不停掉就量不出干净结论。
#   2) **ego 历史不能用 alpamayo_gate4/meta.npz**——实测其 ego_history_xyz 全零、
#      ego_history_rot 全单位阵（闸门 4 只验加载不验数据），会让 v0=0、积分出
#      「静止车刹车到倒车」的物理荒谬轨迹。改用官方指南 alpamayo.md 的真实轨迹
#      （16 点，隐含 ~9.0 m/s = 33 km/h）。
set -u
A=/home/nvidia/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1
PY=$T/export_venv/bin/python
export EDGELLM_PLUGIN_PATH=$T/build_orin/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
cd "$T" || exit 1
mkdir -p "$A/io" "$A/logs"

FRAMES=$A/alpamayo_gate4/frames
CKPT=$A/models/Alpamayo-R1-10B
ENG=$A/engines/alpamayo_r1

# --- [0] 若 llm/visual 引擎尚未建（action 见 r1_action_minverify.sh）---
# 磁盘账：llm ONNX 15.3G + llm 引擎 14.2G 需共存 ⇒ 建 llm 前先把 21G 权重暂搬 dog，
#         建完删 ONNX 再搬回。实测 llm build 峰值 CPU 31.8G（--maxBatchSize 1）。
if [ ! -f "$ENG/llm/llm.engine" ]; then
  echo "=== export llm+visual（--skip-action）==="
  USE_DECOMPOSED_ALPAMAYO=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
    ./export_venv/bin/tensorrt-edgellm-export "$CKPT" "$A/onnx/alpamayo_r1" \
      --skip-action --max-kv-cache-capacity 4096 > "$A/logs/r1_export_llm_visual.log" 2>&1
  echo "  export exit=$?"
  # ⚠️ HF_ENDPOINT 必须设：export 会去取 Qwen/Qwen3-VL-8B-Instruct 的 config.json，
  #    goat 直连 huggingface.co 不通，漏设会卡在网络重试且 RSS 停在 767MB 不动。
  echo "=== visual_build ==="
  ./build_orin/examples/multimodal/visual_build \
      --onnxDir "$A/onnx/alpamayo_r1/visual" --engineDir "$ENG" \
      --minImageTokens 160 --maxImageTokens 18432 --maxImageTokensPerImage 192 \
      > "$A/logs/r1_visual_build.log" 2>&1
  echo "  visual exit=$? (实测 112s, 峰值 CPU 5.4G)"
  echo "=== llm_build ==="
  ./build_orin/examples/llm/llm_build \
      --onnxDir "$A/onnx/alpamayo_r1/llm" --engineDir "$ENG/llm" \
      --maxInputLen 3424 --maxKVCacheCapacity 4096 --maxBatchSize 1 \
      > "$A/logs/r1_llm_build.log" 2>&1
  echo "  llm exit=$? (实测 172s, 峰值 CPU 31.8G, MemAvailable 最低 8.5G)"
fi

# --- [1] 构造输入 JSON（16 帧 camera-major + 官方指南真实 ego 历史）---
echo "=== [1] build inputs ==="
$PY - <<'PYEOF'
import json, re, os
A="/home/nvidia/chenrunbin/alpamayo-edge"
t=open(f"{A}/trtedge-0.10.1/docs/source/user_guide/examples/vla/alpamayo.md").read()
traj=json.loads(re.search(r'"trajectory":\s*(\[.*?\])\s*\n\s*\}',t,re.S).group(1))
F=f"{A}/alpamayo_gate4/frames"
# 帧序权威来源：alpamayo_gate4/gate4_forward.py 的 CAMS × j=1..4（camera-major）
CAMS=["camera_cross_left_120fov","camera_front_wide_120fov",
      "camera_cross_right_120fov","camera_front_tele_30fov"]
paths=[f"{F}/{cam}_{j}.png" for cam in CAMS for j in range(1,5)]
assert len(paths)==16 and all(os.path.exists(p) for p in paths), "帧缺失"
def mk(traj_list, path):
    json.dump({"batch_size":1,"temperature":0.0,"top_p":1.0,"top_k":50,
      "max_generate_length":128,"requests":[{"messages":[
        {"role":"system","content":[{"type":"text","text":
          "You are a driving assistant that generates safe and accurate actions."}]},
        {"role":"user","content":[{"type":"image","image":p} for p in paths]
          +[{"type":"trajectory","trajectory":traj_list}]
          +[{"type":"text","text":"output the chain-of-thought reasoning of the "
             "driving process, then output the future trajectory."}]}]}]},
      open(path,"w"))
mk(traj, f"{A}/io/r1_seed_real.json")            # 真实 ego 历史（~9 m/s）
mk([[0.0,0.0,0.0]]*16, f"{A}/io/r1_seed_zero.json")  # 全零历史（对照轴）
print("  wrote r1_seed_real.json + r1_seed_zero.json (temperature=0)")
PYEOF

# --- [2] 七配置对照：确定性底噪 / seed 轴 / ego 历史轴 ---
echo "=== [2] run 7 configs ==="
run() {  # name seed input
  t0=$(date +%s)
  timeout 180 ./build_orin/examples/multimodal/action_inference \
      --engineDir "$ENG/llm" --multimodalEngineDir "$ENG" --checkpointDir "$CKPT" \
      --inputFile "$A/io/$3" --outputFile "$A/io/r1_seed_$1.json" \
      --noiseSeed "$2" > "$A/logs/r1_seed_$1.log" 2>&1
  echo "  $1 seed=$2 exit=$? wall=$(( $(date +%s)-t0 ))s"
}
run det_a 5 r1_seed_real.json
run det_b 5 r1_seed_real.json     # 与 det_a 同参 ⇒ 确定性底噪（应逐位相同）
run s0 0 r1_seed_real.json
run s1 1 r1_seed_real.json
run s2 2 r1_seed_real.json
run s3 3 r1_seed_real.json
run h0 5 r1_seed_zero.json        # seed 固定、只换 ego 历史

echo "=== [3] 拉回本地分析 ==="
echo "  scp goat:$A/io/r1_seed_*.json  eval/r1/seed_sensitivity/"
echo "  .venv/bin/python eval/r1/r1_seed_sensitivity.py"
echo "R1_FULLCHAIN_INFER_DONE $(date +%T)"
