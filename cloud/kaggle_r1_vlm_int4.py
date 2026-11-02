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
# RUN AS: pushed via `kaggle kernels push` (kernel-metadata.json committed in cloud/).
#
# v2 (2026-11-02): v1 ERRORed at 16s — `pip install '.[tools]'` died inside scikit-build-core's
#   metadata step on Kaggle's Python 3.13 image (real error swallowed by pip -q: "No available
#   output"). Fix: never build/install the TRTEdge package — pip-install the v0.10.1 tools-extra
#   dependency pins directly and run from the source tree via PYTHONPATH (the wheel is
#   python-only anyway: EDGELLM_PYTHON_ONLY_WHEEL=ON; __init__ reads _version.py, no dist-info
#   needed); console scripts invoked via `python -c`. finally now ALWAYS removes
#   /kaggle/working/TRTEdge (v1's errored run left the ~100MB clone in kernel output, making
#   `kaggle kernels output` unusably slow; the run log is retrievable from the API's inline
#   `log` field: GET /api/v1/kernels/output?userName=..&kernelSlug=..).
# v3 (2026-11-02): v2 passed pip/patches/ENV_OK/R1-download but died fetching the Qwen tokenizer
#   files — hub 1.x `hf download` misparses `--include "*.json" "*.txt"` (second pattern leaks
#   into positional filenames; it tried to fetch a literal "*.txt"). Fix: snapshot_download
#   Python API for both fetches; ENV_OK gate now also asserts CUDA compute works (torch resolved
#   to a cu130 build on T4); except re-raises so failures surface as kernel ERROR instead of a
#   misleading COMPLETE (v2's swallowed exit was only caught because the sentinel watched both).
# v4 (2026-11-02): v3 passed downloads AND repack (REPACK_OK 750 tensors/4 shards/17.6GB) but
#   died at quantize-import: Kaggle's preinstalled torchaudio is a cu128 build; transformers
#   5.14.1 audio_utils does a module-level `import torchaudio` whose CUDA check hard-fails
#   against torch 2.13.0+cu130. Fix: uninstall torchaudio (audio unused here; the transformers
#   availability guard then skips the import) + ENV_OK gate now imports both entry-point chains
#   (quantize/export) so import-class landmines detonate BEFORE the 22GB download, not after.
# v5 (2026-11-02): v4 cleared imports/downloads/repack and died in AWQ calibration — CUDA OOM
#   (3.46GiB SDPA alloc). Mechanism: VLM + awq_lite forces the multimodal calibration path
#   (_is_image_blind_calibration, BY DESIGN — text-only calibration "optimizes the image-token
#   ranges away and the model degenerates on image input", so the multimodal recipe must stay);
#   MMMU images go through the processor UNCAPPED, and torch 2.13 SDPA on T4 (sm_75) has only
#   the math backend (flash needs sm_80+), materialising S^2 attention over multi-k visual-token
#   sequences. Fix: patch D caps calibration images to ~400k px (~392 visual tokens) inside
#   _multimodal_calib_dataloader — recipe intact, attention drops to tens of MB.
import os, subprocess, glob, shutil, json, sys

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


def entry(ep, argv):
    # v2: run console-script entry point "mod:fn" from the source tree (package not installed).
    mod, fn = ep.split(":")
    code = f"import sys; sys.argv = {argv!r}; from {mod} import {fn} as _m; _m()"
    print("+", ep, " ".join(argv[1:]), flush=True)
    subprocess.run([sys.executable, "-c", code], check=True)


# 1) TRTEdge pinned to goat's runtime version
df("start")
sh("git clone --depth 1 --branch v0.10.1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
# v2: exact v0.10.1 tools-extra pins (== goat export env), installed directly. The TRTEdge
# package itself is NOT built (scikit-build-core metadata dies on Kaggle py3.13) — the source
# tree + PYTHONPATH is the package. No -q: v1's real error was swallowed by it.
sh("pip install --progress-bar off "
   "torch==2.13.0 torchvision==0.28.0 transformers==5.14.1 accelerate huggingface-hub "
   "onnx==1.19.0 onnxscript==0.7.1 onnx-graphsurgeon==0.6.1 safetensors==0.8.0 numpy==2.2.6 "
   "nvidia-modelopt==0.45.0 datasets==5.0.0 peft==0.19.1 einops==0.8.2 tiktoken==0.13.0 "
   "librosa==0.11.0 soundfile==0.14.0 tqdm==4.69.0 backoff 'cuda-python>=12.6,<14'")
# v4: the image's preinstalled torchaudio is a cu128 build; transformers 5.14.1 audio_utils
# imports it at module level and its CUDA check hard-fails against torch 2.13.0+cu130.
# Audio is unused in this flow; once uninstalled, transformers' availability guard skips it.
sh("pip uninstall -y torchaudio")

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
# v5 patch D: cap calibration image resolution — torch 2.13 SDPA on T4 (sm_75) is
# math-backend-only and materialises S^2; uncapped MMMU images OOM'd v4 (3.46GiB alloc).
_D_old = (
    '    batches: list[dict[str, Any]] = []\n'
    '    for image, question in image_dataset():\n'
    '        messages = [{'
)
_D_new = (
    '    batches: list[dict[str, Any]] = []\n'
    '    for image, question in image_dataset():\n'
    '        # [edge-patch D] Cap calibration image resolution (~400k px -> ~392\n'
    '        # visual tokens): torch 2.13 SDPA on T4 (sm_75) has only the math\n'
    '        # backend and materialises S^2 attention; uncapped MMMU images explode\n'
    '        # to multi-k visual-token sequences -> multi-GiB alloc -> OOM (v4).\n'
    '        # Multimodal recipe stays intact (see _is_image_blind_calibration).\n'
    '        try:\n'
    '            from PIL import Image as _PILImage\n'
    '            if isinstance(image, _PILImage.Image):\n'
    '                _w, _h = image.size\n'
    '                _CAP = 401408\n'
    '                if _w * _h > _CAP:\n'
    '                    _s = (_CAP / (_w * _h)) ** 0.5\n'
    '                    image = image.resize((max(32, int(_w * _s)), max(32, int(_h * _s))))\n'
    '                if not globals().get("_EDGE_D_LOGGED"):\n'
    '                    globals()["_EDGE_D_LOGGED"] = True\n'
    '                    print(f"[edge-patch D] image cap active: {_w}x{_h} -> {image.size}", flush=True)\n'
    '            elif not globals().get("_EDGE_D_LOGGED"):\n'
    '                globals()["_EDGE_D_LOGGED"] = True\n'
    '                print(f"[edge-patch D] WARNING non-PIL image {type(image)} - cap NOT applied", flush=True)\n'
    '        except Exception as _edge_e:\n'
    '            print("[edge-patch] image cap skipped:", _edge_e, flush=True)\n'
    '        messages = [{'
)
assert _src.count(_A_old) == 1, "PATCH A needle not found — TRTEdge v0.10.1 source drifted"
assert _src.count(_B_old) == 1, "PATCH B needle not found"
assert _src.count(_C_old) == 1, "PATCH C needle not found"
assert _src.count(_D_old) == 1, "PATCH D needle not found"
open(QP, "w").write(_src.replace(_A_old, _A_new).replace(_B_old, _B_new).replace(_C_old, _C_new).replace(_D_old, _D_new))
subprocess.run(f"python -m py_compile {QP}", shell=True, check=True)
print("[edge-patch] A(device_map)+B(to-dtype)+C(cpu-export)+D(image-cap) applied on v0.10.1", flush=True)

os.environ["PYTHONPATH"] = "/kaggle/working/TRTEdge"
os.chdir("/kaggle/working")
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", shell=True)
# v2 early gate: source tree must import and versions must match goat's export env
# (fails fast BEFORE the 22GB download if the no-install bypass is broken)
# v3: also assert real CUDA compute — torch resolved to a cu130 build on T4 (sm_75)
# v4: also import both entry-point chains — v3's torchaudio landmine detonated only at
# quantize-import time, after downloads+repack were done; this makes import-class fail fast
sh("""python -c 'import tensorrt_edgellm, torch, transformers; assert torch.cuda.is_available(), "CUDA unavailable"; x = torch.randn(64, 64, device="cuda"); from tensorrt_edgellm.scripts.quantize import main as _q; from tensorrt_edgellm.scripts.export import main as _e; print("ENV_OK trtedge", tensorrt_edgellm.__version__, "torch", torch.__version__, "tfm", transformers.__version__, "cuda_matmul", float((x @ x).sum()), "entry_imports OK")'""")

ok = False
try:
    # 3) Downloads — both public, no token. v3: Python API instead of `hf` CLI —
    #    hub 1.x misparses `--include "*.json" "*.txt"` (second pattern leaks into
    #    positional filenames -> tried to fetch a literal "*.txt" -> v2 failure).
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/Alpamayo-R1-10B", local_dir=R1_DIR)
    df("after-r1-download")
    snapshot_download("Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"],
                      local_dir=QWEN_FILES)

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

    # 5) Quantize (same CLI as proven CR2 run; v2: entry point via python -c)
    entry("tensorrt_edgellm.scripts.quantize:main",
          ["tensorrt-edgellm-quantize", "llm", "--model_dir", REPACK,
           "--output_dir", QCKPT, "--quantization", "int4_awq"])
    print("QUANTIZE_OK", flush=True)
    df("after-quantize")
    shutil.rmtree(REPACK, ignore_errors=True)

    # 6) Export ONNX (KV capacity must match deployed alpamayo llm engine)
    entry("tensorrt_edgellm.scripts.export:main",
          ["tensorrt-edgellm-export", QCKPT, f"{QCKPT}/onnx", "--max-kv-cache-capacity", "4096"])
    print("EXPORT_OK", flush=True)
    df("after-export")
    dest = f"{ART}/R1-vlm-int4_awq-onnx"
    shutil.move(f"{QCKPT}/onnx", dest)
    shutil.rmtree(QCKPT, ignore_errors=True)
    ok = True
except Exception as e:
    print("R1_VLM_INT4_FAILED:", repr(e), flush=True)
    raise  # v3: surface as kernel ERROR — v2 swallowed the exit and Kaggle logged COMPLETE
finally:
    # v2: ALWAYS drop the source clone — v1's errored run left it in kernel output,
    # making `kaggle kernels output` download the whole tree file-by-file
    shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
    produced = glob.glob(f"{ART}/*-onnx")
    if produced:
        try:
            sh("tar -czf /kaggle/working/r1_vlm_int4_onnx.tgz -C " + ART + " " +
               " ".join(os.path.basename(p) for p in produced))
            print("PACKED_OK", [os.path.basename(p) for p in produced], flush=True)
        except Exception as pe:
            print("PACK_FAILED:", pe, flush=True)
    else:
        print("NO_ONNX_PRODUCED", flush=True)
    df("end")

print("DONE ok=" + str(ok), flush=True)
