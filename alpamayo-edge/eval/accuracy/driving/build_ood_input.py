#!/usr/bin/env python3
"""Build the driving-domain (OOD n=184) inference inputs from data/ood_eval/manifest.json.

Two artifacts, mirroring the n=210 robovqa/robofail benchmark protocol exactly:

1. Scaled frames: native 1920x1080 -> long edge 448 (448x252), flat-named
   ood_{i:03d}_{j}.jpg. Same 448-long-edge口径 as bench/frames/robovqa_*.jpg
   (the runtime processor does the 32-alignment 448x252 -> 448x256 = 112 tok/frame).
2. llm_inference input JSON: greedy (temperature 0, top_k 1, top_p 1),
   max_generate_length 12, batch_size 1 -- identical generation params to
   bench/orin_input.json (n=210).

The prompt template is the n=210 one, with the manifest's question and 8 options:

    You are shown 6 frames sampled in order from a video.
    {question}
    Options:
    A. {...}
    ...
    H. {...}
    Answer with ONLY the single letter of the correct option.

Frame paths inside the JSON are absolute and device-specific, so build one JSON
per device (--frames-root). Frames themselves are device-independent: build once,
scp the same tarball everywhere.

Usage:
    .venv/bin/python alpamayo-edge/eval/accuracy/driving/build_ood_input.py \
        --out-dir alpamayo-edge/data/ood_eval/frames_448 \
        --frames-root /home/vision/chenrunbin/alpamayo-edge/bench/ood_eval/frames \
        --json-out alpamayo-edge/eval/accuracy/driving/ood_input_dog.json
"""
import argparse
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OOD = os.path.join(REPO, "data", "ood_eval")

PROMPT_TEMPLATE = (
    "You are shown 6 frames sampled in order from a video.\n"
    "{question}\n"
    "Options:\n{options}\n"
    "Answer with ONLY the single letter of the correct option."
)


def scale_frames(items, out_dir, long_edge=448, native_copy=False):
    os.makedirs(out_dir, exist_ok=True)
    made, skipped = 0, 0
    for i, item in enumerate(items):
        for j, rel in enumerate(item["frames"]):
            dst = os.path.join(out_dir, f"ood_{i:03d}_{j}.jpg")
            if os.path.exists(dst):
                skipped += 1
                continue
            if native_copy:
                # byte-identical copy of the harvested native frame (no re-encode)
                with open(os.path.join(OOD, rel), "rb") as src, open(dst, "wb") as out:
                    out.write(src.read())
            else:
                im = Image.open(os.path.join(OOD, rel)).convert("RGB")
                w, h = im.size
                s = long_edge / max(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
                im.save(dst, quality=90)
            made += 1
    sizes = {Image.open(os.path.join(out_dir, f)).size
             for f in os.listdir(out_dir) if f.endswith(".jpg")}
    print(f"frames: {made} made, {skipped} existed; sizes {sizes}")
    return sizes


def build_json(items, frames_root, json_out):
    requests = []
    for i, item in enumerate(items):
        opts = item["options"]
        options = "\n".join(f"{L}. {opts[L]}" for L in sorted(opts))
        text = PROMPT_TEMPLATE.format(question=item["question"], options=options)
        content = [{"type": "image",
                    "image": f"{frames_root}/ood_{i:03d}_{j}.jpg"}
                   for j in range(len(item["frames"]))]
        content.append({"type": "text", "text": text})
        requests.append({"messages": [{"role": "user", "content": content}]})
    doc = {"batch_size": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
           "max_generate_length": 12, "requests": requests}
    with open(json_out, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"wrote {json_out}: {len(requests)} requests")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(OOD, "frames_448"))
    ap.add_argument("--frames-root", required=True,
                    help="absolute frames dir ON THE DEVICE that will run inference")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--long-edge", type=int, default=448)
    ap.add_argument("--native-copy", action="store_true",
                    help="byte-copy the harvested native frames instead of resizing "
                         "(used for the native-1080p Pareto tier)")
    a = ap.parse_args()

    man = json.load(open(os.path.join(OOD, "manifest.json")))
    items = man["items"]
    assert len(items) == man["n_items"] == 184, len(items)
    scale_frames(items, a.out_dir, a.long_edge, native_copy=a.native_copy)
    build_json(items, a.frames_root.rstrip("/"), a.json_out)


if __name__ == "__main__":
    main()
