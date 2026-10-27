import pandas as pd, numpy as np, subprocess, zipfile, os, sys, io
A="/home/nvidia/chenrunbin/alpamayo-edge"; E=f"{A}/eval_data"
FF=f"{A}/frames_venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2"
T0=5_100_000
# camera order MUST match gate4 camera_indices [0,1,2,6]
CAMS=[("camera_cross_left_120fov","camera_cross_left_120fov.chunk_3116.zip"),
      ("camera_front_wide_120fov","camera_front_wide_120fov.chunk_3116.zip"),
      ("camera_cross_right_120fov","camera_cross_right_120fov.chunk_3116.zip"),
      ("camera_front_tele_30fov","camera_front_tele_30fov.chunk_3116.zip")]
clips=[l.strip() for l in open(f"{E}/selected_clips.txt") if l.strip()]
targets=[T0-300000,T0-200000,T0-100000,T0]   # t0-0.3 .. t0
zips={cam:zipfile.ZipFile(f"{E}/chunk3116/camzip/{zf}") for cam,zf in CAMS}
os.makedirs(f"{E}/frames",exist_ok=True)
nok=0; nbad=[]
for ci,clip in enumerate(clips):
    os.makedirs(f"{E}/frames/{clip}",exist_ok=True)
    for cam,zf in CAMS:
        z=zips[cam]
        mp4n=f"{clip}.{cam}.mp4"; tsn=f"{clip}.{cam}.timestamps.parquet"
        try:
            ts=pd.read_parquet(io.BytesIO(z.read(tsn)))["timestamp"].values.astype(np.int64)
        except KeyError:
            nbad.append((clip,cam,"no ts")); continue
        idxs=[int(np.argmin(np.abs(ts-t))) for t in targets]
        offs=[int(ts[i]-t) for i,t in zip(idxs,targets)]
        if any(abs(o)>20000 for o in offs): nbad.append((clip,cam,f"off={offs}"))
        # extract mp4 to temp, then the 4 frames
        mp4b=z.read(mp4n)
        tmp=f"/tmp/x_{clip[:8]}_{cam}.mp4"
        open(tmp,"wb").write(mp4b)
        sel="+".join(f"eq(n\\,{i})" for i in idxs)
        for j in range(4):
            out=f"{E}/frames/{clip}/{cam}_{j+1}.png"
        subprocess.run([FF,"-y","-i",tmp,"-vf",f"select='{sel}'","-vsync","0",
                        f"{E}/frames/{clip}/{cam}_%d.png"],
                       capture_output=True)
        os.remove(tmp)
        got=len([f for f in os.listdir(f"{E}/frames/{clip}") if f.startswith(cam)])
        nok+= 1 if got==4 else 0
        if got!=4: nbad.append((clip,cam,f"got {got}"))
    print(f"  [{ci+1}/{len(clips)}] {clip[:8]} done",flush=True)
print("OK clips fully extracted:",nok,"/",len(clips))
if nbad: print("issues:",nbad[:10])
