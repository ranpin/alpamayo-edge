# Kaggle script — Alpamayo-R1 vlm backbone INT4 (AWQ) quantization for TRTEdge.
#
# Clean experiment (G1): quantize R1's OWN AV-trained backbone, not vanilla CR2-8B.
# Path: repack R1 `vlm.*` (750 tensors, name-set == CR2-8B after prefix strip, shapes
# identical except vocab 155697 vs 151936) into a qwen3_vl-format checkpoint, then run
# the PROVEN CR2 quantize->export flow on it.
#
# Design decisions (all recon-verified 2026-11-02):
#   - TRTEdge pinned to v0.10.1 == goat runtime (ONNX->llm_build compatibility).
#   - Patch A needle updated: 0.10.1 added `low_cpu_mem_usage=True,` to the
#     from_pretrained block (July needle from kaggle_cosmos_only.py no longer matches).
#   - No HF token anywhere: R1 is public (gated=False); tokenizer/processor files come
#     from public Qwen/Qwen3-VL-8B-Instruct; CR2 config template embedded below
#     (architecture params only) with vocab_size patched to R1's 155697.
#   - --max-kv-cache-capacity 4096 on export: must match the deployed alpamayo llm
#     engine (alpamayo.md: value must match --maxKVCacheCapacity at engine build).
#   - Disk: /kaggle/working is a 20GB loop; ALL heavy work under /tmp (1.1TB overlay),
#     only the final tgz goes to /kaggle/working (pattern proven in kaggle_cosmos_only.py).
#
# PREREQ: Accelerator = GPU T4 x2, Internet = On.
# RUN AS: pushed via `kaggle kernels push` (see kernel-metadata template in cloud/README).
import os, subprocess, glob, shutil, json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SCRATCH = "/tmp/edge"
ART = f"{SCRATCH}/artifacts"
R1_DIR = f"{SCRATCH}/Alpamayo-R1-10B"
QWEN_FILES = f"{SCRATCH}/qwen3vl_files"
REPACK = f"{SCRATCH}/R1-vlm-qwen3vlfmt"
QCKPT = f"{SCRATCH}/R1-vlm-int4_awq"
os.makedirs(ART, exist_ok=True)


def sh(c):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=True)


def df(tag):
    print(f"--- df [{tag}] ---", flush=True)
    subprocess.run("df -h /kaggle/working /tmp 2>/dev/null | sort -u", shell=True)


# 1) TRTEdge pinned to goat's runtime version
df("start")
sh("git clone --depth 1 --branch v0.10.1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
sh("cd /kaggle/working/TRTEdge && pip -q install '.[tools]'")

# 2) Same three patches as kaggle_cosmos_only.py (A needle updated for 0.10.1).
#    Needles asserted unique so upstream drift can't silently skip a patch.
QP = "/kaggle/working/TRTEdge/tensorrt_edgellm/quantization/quantize.py"
_src = open(QP).read()
_A_old = (
    '                model = factory.from_pretrained(\n'
    '                    model_dir,\n'
    '                    torch_dtype=torch_dtype,\n'
    '                    trust_remote_code=True,\n'
    '                    low_cpu_mem_usage=True,\n'
    '                ).to(device)'
)
_A_new = (
    '                model = factory.from_pretrained(\n'
    '                    model_dir,\n'
    '                    torch_dtype=torch_dtype,\n'
    '                    trust_remote_code=True,\n'
    '                    low_cpu_mem_usage=True,\n'
    '                    device_map="auto",\n'
    '                )'
)
_B_old = '    model.to(torch_dtype)'
_B_new = (
    '    try:\n'
    '        model.to(torch_dtype)\n'
    '    except Exception as _edge_e:\n'
    '        print("[edge-patch] skip model.to(dtype) under device_map:", _edge_e, flush=True)\n'
)
_C_old = '    os.makedirs(output_dir, exist_ok=True)\n'
_C_new = (
    '    _dm = getattr(model, "hf_device_map", None)\n'
    '    if _dm and len({str(v) for v in _dm.values()}) > 1:\n'
    '        try:\n'
    '            from accelerate.hooks import remove_hook_from_module\n'
    '            remove_hook_from_module(model, recurse=True)\n'
    '        except Exception as _edge_e:\n'
    '            print("[edge-patch] remove_hook failed:", _edge_e, flush=True)\n'
    '        model.to("cpu")\n'
    '        try:\n'
    '            model.hf_device_map = {"": "cpu"}\n'
    '        except Exception:\n'
    '            pass\n'
    '        import gc as _gc\n'
    '        _gc.collect(); torch.cuda.empty_cache()\n'
    '        print("[edge-patch] consolidated sharded model -> CPU for export", flush=True)\n'
    '\n'
    '    os.makedirs(output_dir, exist_ok=True)\n'
)
assert _src.count(_A_old) == 1, "PATCH A needle not found — TRTEdge v0.10.1 source drifted"
assert _src.count(_B_old) == 1, "PATCH B needle not found"
assert _src.count(_C_old) == 1, "PATCH C needle not found"
open(QP, "w").write(_src.replace(_A_old, _A_new).replace(_B_old, _B_new).replace(_C_old, _C_new))
subprocess.run(f"python -m py_compile {QP}", shell=True, check=True)
print("[edge-patch] A(device_map)+B(to-dtype)+C(cpu-export) applied on v0.10.1", flush=True)

os.environ["PYTHONPATH"] = "/kaggle/working/TRTEdge"
os.chdir("/kaggle/working")
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", shell=True)

ok = False
try:
    # 3) Downloads — both public, no token
    sh(f"hf download nvidia/Alpamayo-R1-10B --local-dir {R1_DIR}")
    df("after-r1-download")
    sh(f'hf download Qwen/Qwen3-VL-8B-Instruct --include "*.json" "*.txt" --local-dir {QWEN_FILES}')

    # 4) Repack: vlm.* -> strip prefix; CR2 config template + R1 vocab; Qwen3-VL tokenizer files
    import torch  # noqa: F401  (safetensors.torch backend)
    from safetensors.torch import load_file, save_file

    os.makedirs(REPACK, exist_ok=True)
    CR2_CFG = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "image_token_id": 151655,
        "model_type": "qwen3_vl",
        "text_config": {
            "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643,
            "dtype": "bfloat16", "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
            "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 12288,
            "max_position_embeddings": 262144, "model_type": "qwen3_vl_text",
            "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
            "rope_theta": 5000000, "use_cache": True,
            "vocab_size": 155697,  # R1 expanded vocab (traj tokens); CR2 native = 151936
        },
        "tie_word_embeddings": False,
        "transformers_version": "4.57.0.dev0",
        "video_token_id": 151656,
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24], "depth": 27, "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 4304,
            "model_type": "qwen3_vl", "num_heads": 16, "num_position_embeddings": 2304,
            "out_hidden_size": 4096, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
    }
    json.dump(CR2_CFG, open(f"{REPACK}/config.json", "w"), indent=2)
    for f in glob.glob(f"{QWEN_FILES}/*"):
        b = os.path.basename(f)
        if not os.path.isfile(f): continue
        if b in ("config.json", "model.safetensors.index.json"): continue  # ours wins
        shutil.copy(f, f"{REPACK}/{b}")

    idx = json.load(open(f"{R1_DIR}/model.safetensors.index.json"))["weight_map"]
    vlm_by_shard = {}
    for k, s in idx.items():
        if k.startswith("vlm."):
            vlm_by_shard.setdefault(s, []).append(k)
    n_exp = sum(len(v) for v in vlm_by_shard.values())
    assert n_exp == 750, f"expected 750 vlm tensors, got {n_exp}"
    buf, buf_bytes, total, new_map, out_i = {}, 0, 0, {}, 0
    LIMIT = int(4.5e9)

    def flush():
        global buf, buf_bytes, out_i
        if not buf: return
        out_i += 1
        save_file(buf, f"{REPACK}/model-{out_i:05d}.safetensors", metadata={"format": "pt"})
        for k in buf: new_map[k] = f"model-{out_i:05d}.safetensors"
        print(f"[repack] shard {out_i}: {len(buf)} tensors {buf_bytes/1e9:.1f}GB", flush=True)
        buf, buf_bytes = {}, 0

    for s in sorted(vlm_by_shard):
        tens = load_file(f"{R1_DIR}/{s}")
        for k in vlm_by_shard[s]:
            t = tens[k]; nk = k[len("vlm."):]
            buf[nk] = t
            nb = t.numel() * t.element_size(); buf_bytes += nb; total += nb
            if buf_bytes >= LIMIT: flush()
        del tens
    flush()
    assert len(new_map) == 750, f"repacked {len(new_map)} != 750"
    n = out_i
    for i in range(1, n + 1):
        os.rename(f"{REPACK}/model-{i:05d}.safetensors", f"{REPACK}/model-{i:05d}-of-{n:05d}.safetensors")
    final_map = {k: v.replace(".safetensors", f"-of-{n:05d}.safetensors") for k, v in new_map.items()}
    json.dump({"metadata": {"total_size": total}, "weight_map": final_map},
              open(f"{REPACK}/model.safetensors.index.json", "w"), indent=2)
    print(f"REPACK_OK 750 tensors, {n} shards, {total/1e9:.1f}GB -> {REPACK}", flush=True)
    df("after-repack")
    # free the 22GB raw checkpoint — repack is verified complete
    shutil.rmtree(R1_DIR, ignore_errors=True)

    # 5) Quantize (same CLI as proven CR2 run)
    sh(f"tensorrt-edgellm-quantize llm --model_dir {REPACK} --output_dir {QCKPT} --quantization int4_awq")
    print("QUANTIZE_OK", flush=True)
    df("after-quantize")
    shutil.rmtree(REPACK, ignore_errors=True)

    # 6) Export ONNX (KV capacity must match deployed alpamayo llm engine)
    sh(f"tensorrt-edgellm-export {QCKPT} {QCKPT}/onnx --max-kv-cache-capacity 4096")
    print("EXPORT_OK", flush=True)
    df("after-export")
    dest = f"{ART}/R1-vlm-int4_awq-onnx"
    shutil.move(f"{QCKPT}/onnx", dest)
    shutil.rmtree(QCKPT, ignore_errors=True)
    ok = True
except Exception as e:
    print("R1_VLM_INT4_FAILED:", repr(e), flush=True)
finally:
    produced = glob.glob(f"{ART}/*-onnx")
    if produced:
        try:
            shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
            sh("tar -czf /kaggle/working/r1_vlm_int4_onnx.tgz -C " + ART + " " +
               " ".join(os.path.basename(p) for p in produced))
            print("PACKED_OK", [os.path.basename(p) for p in produced], flush=True)
        except Exception as pe:
            print("PACK_FAILED:", pe, flush=True)
    else:
        print("NO_ONNX_PRODUCED", flush=True)
    df("end")

print("DONE ok=" + str(ok), flush=True)
