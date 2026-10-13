#!/bin/bash
# 闸门 4 驱动（goat）：patch config → 加载 + 单次前向。
# 离线适配（goat 不可达 huggingface.co）：
#  1. config.json 的 vlm_name_or_path 指向 nvidia/Cosmos-Reason2-8B ⇒ 改为本地 staged
#     processor 目录（CR2 tokenizer.json 词表已验证与 Qwen3-VL 完全一致，
#     preprocessor_config.json 与官方 helper 用的 Qwen3-VL-2B-Instruct 逐字节相同）。
#  2. processor 目录补 Qwen3-VL-8B-Instruct 的 config.json（架构与 CR2 实测一致：
#     36 层/hidden 4096/vocab 151936）——_initialize_qwenvl3_vlm 要从该路径读 Qwen3VLConfig。
#  3. base_model.py 的默认 vlm_name_or_path（"Qwen/Qwen3-VL-8B-Instruct"）sed 为本地路径：
#     transformers 的 to_diff_dict 会用【无参默认值】重构一个 config，而该 config 类
#     __init__ 带网络副作用（_build_processor）⇒ 默认值必须离线可解析（上游设计地雷）。
#  4. HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE：任何漏网的远程请求快速失败而非 5 轮重试。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
A=/home/nvidia/chenrunbin/alpamayo-edge
cd "$A/alpamayo1.5" || exit 1
source a1_5_venv/bin/activate

python - << 'EOF'
import json
p = "/home/nvidia/chenrunbin/alpamayo-edge/alpamayo_gate4/Alpamayo-1.5-10B/config.json"
c = json.load(open(p))
proc = "/home/nvidia/chenrunbin/alpamayo-edge/alpamayo_gate4/processor"
if c.get("vlm_name_or_path") != proc:
    print("patch vlm_name_or_path:", c.get("vlm_name_or_path"), "->", proc)
    c["vlm_name_or_path"] = proc
    json.dump(c, open(p, "w"), indent=2)
else:
    print("config already patched")
EOF

echo "=== gate4_forward start $(date +%T) ==="
python "$A/alpamayo_gate4/gate4_forward.py"
rc=$?
echo "=== gate4_forward exit=$rc $(date +%T) ==="
exit $rc
