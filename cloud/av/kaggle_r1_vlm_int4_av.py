# Kaggle script — Alpamayo-R1 vlm backbone INT4 (AWQ), AV-domain calibration arm.
#
# Control experiment (2026-11-03): identical to cloud/kaggle_r1_vlm_int4.py (MMMU arm,
# v5) in EVERY respect except the image calibration dataset — this arm calibrates AWQ
# on real AV driving frames instead of MMMU academic images. Pre-registered follow-up
# variable from the INT4 experiment (eval/r1/int4_backbone.md §五.3): the MMMU arm
# measured minADE6@6.4s 0.80046 vs FP16 0.80401 m (Δ=-0.004m, "nearly free"); this
# arm tests whether AV-domain calibration changes that verdict.
#
# Calibration frames: dgural/PhysicalAI-Autonomous-Vehicles-Sample (PUBLIC HF dataset,
# gated=false — the 100-clip public sample of PhysicalAI-AV; Kaggle downloads it
# directly, no gated data is touched or redistributed). 8 clips (first 8 UUIDs in
# repo order, deterministic) × R1's 4 deployment cameras (cross_left/right_120fov,
# front_tele_30fov, front_wide_120fov — the demo's cl/cr/ft/fw) × 4 frames (fps=0.4)
# = 128 frames == the MMMU arm's 128 samples. Patch D's ~400k-px cap applies
# identically (1920×1080 frames get capped just like large MMMU images).
# Questions: fixed pool of 8 driving questions, cycled (the MMMU arm used each
# image's own academic question; both arms pair image+question through the same
# chat template, so the variable is image domain, not prompt structure).
#
# Injection: patch E registers an "av_frames" image dataset through TRTEdge's own
# calibration-dataset registry (@register("image", "av_frames") module + one import
# line in datasets/__init__.py) and the quantize call passes --image_dataset av_frames.
# The driver logs "Image calibration dataset: av_frames" — verify in the run log.
#
# Everything else (repack recipe, patches A-D, dependency pins, export flags) is
# IDENTICAL to v5 so the only experimental variable is the calibration image domain.
#
# v2 (2026-11-03): v1 died in the FIRST calibration forward (0/128, ~10min in — after every
#   AV-specific step had succeeded: AVFRAMES_OK 128, "Image calibration dataset: av_frames",
#   patch D cap 1920x1080->844x475, 1484 quantizers inserted). Cause: F.conv3d in the vision
#   tower's patch_embed raised CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH — Kaggle's preinstalled
#   nvidia-cudnn-cu12 coexists with torch cu130's nvidia-cudnn-cu13 in the SAME nvidia/cudnn/lib
#   directory and the mixed-version sublibraries got loaded (v5/MMMU arm survived the same recipe
#   => the mixed state is node/order dependent). Fix: uninstall nvidia-cudnn-cu12 + a conv3d
#   probe in the ENV_OK gate replicating the exact patch_embed op (fails at ~4min, before the
#   22GB download, if the env is still broken). Note: v1's script actually ran only 615s; the
#   ~2h RUNNING status was Kaggle status lag — logs are ground truth.
# v3 (2026-11-03): v2's cu12 uninstall did NOT clear the mismatch — the conv3d probe died at
#   ~5min (before the 22GB download, exactly as designed; TRTEdge left in output => the crash
#   was pre-try). v3: (1) force-reinstall the exact nvidia-cudnn-cu13 torch declares + put its
#   lib dir first on LD_LIBRARY_PATH (consistent cu13 sublibrary set); (2) probe is now
#   NON-FATAL and adaptive — if cuDNN conv is still broken, quantize/export run with
#   torch.backends.cudnn.enabled=False (native conv3d; ulp-level numeric difference vs the
#   MMMU arm's cuDNN path — documented caveat, far below AWQ/INT4 granularity); (3) atexit
#   guarantees TRTEdge cleanup even for pre-try crashes (fast CLI log retrieval).
import os, subprocess, glob, shutil, json, sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SCRATCH = "/tmp/edge"
ART = f"{SCRATCH}/artifacts"
R1_DIR = f"{SCRATCH}/Alpamayo-R1-10B"
QWEN_FILES = f"{SCRATCH}/qwen3vl_files"
DGURAL_DIR = f"{SCRATCH}/dgural"
AVFRAMES = f"{SCRATCH}/av_frames"
REPACK = f"{SCRATCH}/R1-vlm-qwen3vlfmt"
QCKPT = f"{SCRATCH}/R1-vlm-int4_awq_av"
os.makedirs(ART, exist_ok=True)

CLIPS = [
    "01d3588e-bca7-4a18-8e74-c6cfe9e996db",
    "038678de-6307-4cd2-a3c9-5d64e1fb8862",
    "0391b4a7-3956-4146-ad2e-98b74e7877fa",
    "0397f34a-b3f3-4e40-ac2b-a9fcd656226c",
    "04c5a3f4-7d2f-499a-ba05-b57194594735",
    "054da32b-9f3d-4074-93ab-044036b679f8",
    "058bf960-f531-4ae6-8b24-455d7683676e",
    "0853ef8f-9d6b-4963-8ea3-fafb7e107ac7",
]
CAMS = {  # short name -> dgural camera file component (R1's 4 deployment cameras)
    "cl": "camera_cross_left_120fov",
    "cr": "camera_cross_right_120fov",
    "ft": "camera_front_tele_30fov",
    "fw": "camera_front_wide_120fov",
}


def sh(c):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, check=True)


def df(tag):
    print(f"--- df [{tag}] ---", flush=True)
    subprocess.run("df -h /kaggle/working /tmp 2>/dev/null | sort -u", shell=True)


def entry(ep, argv, no_cudnn=False):
    # Run console-script entry point "mod:fn" from the source tree (package not installed).
    # v3: no_cudnn disables cuDNN in the subprocess (native conv3d) when the probe found the
    # cuDNN sublibrary set still broken.
    mod, fn = ep.split(":")
    pre = "import torch; torch.backends.cudnn.enabled = False; " if no_cudnn else ""
    code = f"import sys; sys.argv = {argv!r}; {pre}from {mod} import {fn} as _m; _m()"
    print("+", ep, " ".join(argv[1:]), f"(cudnn={'off' if no_cudnn else 'on'})", flush=True)
    subprocess.run([sys.executable, "-c", code], check=True)


# 1) TRTEdge pinned to goat's runtime version + exact dependency pins (identical to v5)
df("start")
sh("git clone --depth 1 --branch v0.10.1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
# v3: guarantee TRTEdge cleanup even for pre-try crashes (v2 died before the try block and
# left the clone in kernel output — CLI log retrieval then crawls ~2k files)
import atexit
atexit.register(lambda: shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True))
sh("pip install --progress-bar off "
   "torch==2.13.0 torchvision==0.28.0 transformers==5.14.1 accelerate huggingface-hub "
   "onnx==1.19.0 onnxscript==0.7.1 onnx-graphsurgeon==0.6.1 safetensors==0.8.0 numpy==2.2.6 "
   "nvidia-modelopt==0.45.0 datasets==5.0.0 peft==0.19.1 einops==0.8.2 tiktoken==0.13.0 "
   "librosa==0.11.0 soundfile==0.14.0 tqdm==4.69.0 backoff 'cuda-python>=12.6,<14'")
sh("pip uninstall -y torchaudio")
# v2: drop the stale cu12 cudnn — it coexists with torch cu130's nvidia-cudnn-cu13 in the
# SAME nvidia/cudnn/lib/ dir; mixed-version sublibraries killed v1's first conv3d
# (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH). "|| true": absent is fine.
sh("pip uninstall -y nvidia-cudnn-cu12 || true")
# v3: v2 proved the cu12 uninstall alone doesn't clear the mismatch — force a consistent cu13
# sublibrary set (reinstall exactly what torch declares) and put it first on the loader path.
sh("""python3 -c "import importlib.metadata as m; rs=[r for r in (m.requires('torch') or []) if 'nvidia-cudnn' in r]; print(rs[0].split(';')[0].strip().replace(' ', '') if rs else '')" > /tmp/cudnn_req.txt; cat /tmp/cudnn_req.txt""")
sh('REQ=$(cat /tmp/cudnn_req.txt); if [ -n "$REQ" ]; then pip install --force-reinstall --no-deps "$REQ"; else pip install --force-reinstall --no-deps nvidia-cudnn-cu13; fi')
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.13/dist-packages/nvidia/cudnn/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
if subprocess.run("which ffmpeg", shell=True).returncode != 0:
    sh("apt-get update -qq && apt-get install -y -qq ffmpeg")
sh("ffmpeg -version | head -1")

# 2) Patches A-D (identical to v5) + patch E (av_frames dataset registration)
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
_D_old = (
    '    batches: list[dict[str, Any]] = []\n'
    '    for image, question in image_dataset():\n'
    '        messages = [{'
)
_D_new = (
    '    batches: list[dict[str, Any]] = []\n'
    '    for image, question in image_dataset():\n'
    '        # [edge-patch D] Cap calibration image resolution: torch 2.13 SDPA on T4\n'
    '        # (sm_75) has only the math backend and materialises S^2 attention;\n'
    '        # uncapped images explode to multi-k visual tokens -> multi-GiB alloc -> OOM.\n'
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

# Patch E: av_frames calibration dataset via TRTEdge's own registry
_AV_DS = """import os

from PIL import Image

from . import register

_DIR = "__AVFRAMES__"
_QUESTIONS = [
    "What should the ego vehicle do next?",
    "Describe the current driving scene and the road layout.",
    "What traffic signals, signs, or road markings are visible?",
    "Is there a lead vehicle ahead, and how close is it?",
    "What maneuver should the ego vehicle perform at the upcoming intersection?",
    "Are there pedestrians or cyclists near the roadway?",
    "How many lanes are there in the ego vehicle's direction of travel?",
    "What is the overall traffic density in this scene?",
]


@register("image", "av_frames")
def av_frames():
    files = sorted(f for f in os.listdir(_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for i, f in enumerate(files):
        yield Image.open(os.path.join(_DIR, f)).convert("RGB"), _QUESTIONS[i % len(_QUESTIONS)]
""".replace("__AVFRAMES__", AVFRAMES)
open("/kaggle/working/TRTEdge/tensorrt_edgellm/quantization/datasets/av_frames.py", "w").write(_AV_DS)
_DS_INIT = "/kaggle/working/TRTEdge/tensorrt_edgellm/quantization/datasets/__init__.py"
_di = open(_DS_INIT).read()
_E_old = "from . import builtin as _builtin"
assert _di.count(_E_old) == 1, "PATCH E needle not found"
open(_DS_INIT, "w").write(_di.replace(_E_old, _E_old + "\nfrom . import av_frames as _av_frames  # noqa: E402,F401"))
print("[edge-patch] A(device_map)+B(to-dtype)+C(cpu-export)+D(image-cap)+E(av_frames dataset) applied on v0.10.1", flush=True)

os.environ["PYTHONPATH"] = "/kaggle/working/TRTEdge"
os.chdir("/kaggle/working")
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", shell=True)
# Early gate: source tree + both entry chains + patch E's dataset import must all load
# (fails fast BEFORE the 22GB download if anything is broken)
sh("""python -c 'import tensorrt_edgellm, torch, transformers; assert torch.cuda.is_available(), "CUDA unavailable"; x = torch.randn(64, 64, device="cuda"); from tensorrt_edgellm.scripts.quantize import main as _q; from tensorrt_edgellm.scripts.export import main as _e; from tensorrt_edgellm.quantization.datasets import get_dataset; assert get_dataset("image", "av_frames").calib_name == "av_frames"; print("ENV_OK trtedge", tensorrt_edgellm.__version__, "torch", torch.__version__, "tfm", transformers.__version__, "cuda_matmul", float((x @ x).sum()), "entry_imports OK", "av_frames registered")'""")
# v3: cuDNN conv3d probe — NON-FATAL and adaptive. If the cu13 reinstall + LD_LIBRARY_PATH
# didn't clear the mismatch, quantize/export run with cudnn disabled (native conv3d; ulp-level
# numeric difference vs the MMMU arm's cuDNN path — documented caveat, far below INT4 granularity).
_PROBE = ('import torch, os, glob; '
          'libs = sorted(glob.glob("/usr/local/lib/python3.13/dist-packages/nvidia/cudnn*/lib/libcudnn*.so*")); '
          'print("cudnn libs:", [os.path.basename(l) for l in libs][:10]); '
          'w = torch.randn(1152, 3, 2, 16, 16, device="cuda", dtype=torch.float16); '
          'x = torch.randn(1, 3, 2, 480, 832, device="cuda", dtype=torch.float16); '
          'y = torch.nn.functional.conv3d(x, w, stride=(2, 16, 16)); torch.cuda.synchronize(); '
          'print("CUDNN_CONV3D_OK", tuple(y.shape), "cudnn", torch.backends.cudnn.version())')
_p = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
CUDNN_OK = (_p.returncode == 0)
print("CUDNN_PROBE:", "OK" if CUDNN_OK else "BROKEN -> quantize/export disable cudnn", flush=True)
print((_p.stdout or "").strip()[-800:], flush=True)
if not CUDNN_OK:
    print((_p.stderr or "").strip()[-1500:], flush=True)

ok = False
try:
    # 3) Downloads — R1 + Qwen tokenizer (identical to v5) + dgural AV videos (public)
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/Alpamayo-R1-10B", local_dir=R1_DIR)
    df("after-r1-download")
    snapshot_download("Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"],
                      local_dir=QWEN_FILES)
    snapshot_download("dgural/PhysicalAI-Autonomous-Vehicles-Sample", repo_type="dataset",
                      allow_patterns=[f"data/{c}.{cam}.mp4" for c in CLIPS for cam in CAMS.values()],
                      local_dir=DGURAL_DIR)
    df("after-dgural-download")

    # 3b) Extract 128 calibration frames: 8 clips x 4 cams x 4 frames (fps=0.4)
    os.makedirs(AVFRAMES, exist_ok=True)
    for c in CLIPS:
        for short, cam in CAMS.items():
            v = f"{DGURAL_DIR}/data/{c}.{cam}.mp4"
            sh(f'ffmpeg -y -loglevel error -i "{v}" -vf fps=0.4 -frames:v 4 -q:v 2 '
               f'"{AVFRAMES}/{c[:8]}_{short}_%d.jpg"')
    n_frames = len(glob.glob(f"{AVFRAMES}/*.jpg"))
    assert n_frames >= 128, f"expected >=128 AV frames, got {n_frames}"
    print(f"AVFRAMES_OK {n_frames} frames ({len(CLIPS)} clips x {len(CAMS)} cams x 4)", flush=True)
    shutil.rmtree(DGURAL_DIR, ignore_errors=True)

    # 4) Repack (identical to v5)
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
    shutil.rmtree(R1_DIR, ignore_errors=True)

    # 5) Quantize with AV-domain calibration (the ONLY difference from the MMMU arm)
    entry("tensorrt_edgellm.scripts.quantize:main",
          ["tensorrt-edgellm-quantize", "llm", "--model_dir", REPACK,
           "--output_dir", QCKPT, "--quantization", "int4_awq",
           "--image_dataset", "av_frames"],
          no_cudnn=not CUDNN_OK)
    print("QUANTIZE_OK", flush=True)
    df("after-quantize")
    shutil.rmtree(REPACK, ignore_errors=True)

    # 6) Export ONNX (identical to v5)
    entry("tensorrt_edgellm.scripts.export:main",
          ["tensorrt-edgellm-export", QCKPT, f"{QCKPT}/onnx", "--max-kv-cache-capacity", "4096"],
          no_cudnn=not CUDNN_OK)
    print("EXPORT_OK", flush=True)
    df("after-export")
    dest = f"{ART}/R1-vlm-int4_awq_av-onnx"
    shutil.move(f"{QCKPT}/onnx", dest)
    shutil.rmtree(QCKPT, ignore_errors=True)
    ok = True
except Exception as e:
    print("R1_VLM_INT4_AV_FAILED:", repr(e), flush=True)
    raise  # surface as kernel ERROR, not a misleading COMPLETE
finally:
    # Always drop the source clone — errored runs must not bloat kernel output
    shutil.rmtree("/kaggle/working/TRTEdge", ignore_errors=True)
    produced = glob.glob(f"{ART}/*-onnx")
    if produced:
        try:
            sh("tar -czf /kaggle/working/r1_vlm_int4_av_onnx.tgz -C " + ART + " " +
               " ".join(os.path.basename(p) for p in produced))
            print("PACKED_OK", [os.path.basename(p) for p in produced], flush=True)
        except Exception as pe:
            print("PACK_FAILED:", pe, flush=True)
    else:
        print("NO_ONNX_PRODUCED", flush=True)
    df("end")

print("DONE ok=" + str(ok), flush=True)
