#!/usr/bin/env python3
"""为滚动 demo 的 4 个 clip 取 gated 逐帧时间戳（HTTP Range 部分下载，免下 2GB 整包）。
复用 dgural_demo/fetch_ts_multi.py 的 zip 中央目录解析逻辑，改成 4 clip + 只存 timestamps.npy。
精确帧↔egomotion 同步用（朴素 frame=t×30 会错 0.03-0.17s / 1-2m）。
"""
import subprocess, struct, io, zlib, os, glob
import pandas as pd, numpy as np

TOK = os.environ.get("HF_TOKEN", "")  # 从环境变量读，勿硬编码
URL = "https://hf-mirror.com/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip"
SIZE = 2050553236
EGO = '/tmp/r1vid/ego'
OUT = '/tmp/r1vid/roll/ts'
CLIPS = ['ecafce84', '1fcf2fef', '0dd3eb7f', '1f7ef408']
full = {os.path.basename(f).split('.')[0][:8]: os.path.basename(f).split('.')[0] for f in glob.glob(f'{EGO}/*.parquet')}

_n = [0]
def fetch(start, end):
    _n[0] += 1; out = f"/tmp/zrm_{_n[0]}.bin"
    for a in range(3):
        r = subprocess.run(["curl","-sL","--max-time","90","-r",f"{start}-{end}","-H",f"Authorization: Bearer {TOK}",URL,"-o",out,"-w","%{http_code}"],capture_output=True,text=True)
        if os.path.exists(out) and os.path.getsize(out) > 0 and r.stdout.strip() in ('206','200'):
            d = open(out,"rb").read(); os.remove(out); return d
    d = open(out,"rb").read() if os.path.exists(out) else b''
    if os.path.exists(out): os.remove(out)
    return d

tail = fetch(SIZE-256*1024, SIZE-1); i = tail.rfind(b'\x50\x4b\x05\x06')
cd_size = struct.unpack('<I', tail[i+12:i+16])[0]; cd_off = struct.unpack('<I', tail[i+16:i+20])[0]
cd = fetch(cd_off, cd_off+cd_size-1)
pos = 0; entries = {}
while pos+46 <= len(cd) and cd[pos:pos+4] == b'\x50\x4b\x01\x02':
    method = struct.unpack('<H', cd[pos+10:pos+12])[0]; csize = struct.unpack('<I', cd[pos+20:pos+24])[0]
    fnl = struct.unpack('<H', cd[pos+28:pos+30])[0]; exl = struct.unpack('<H', cd[pos+30:pos+32])[0]; cml = struct.unpack('<H', cd[pos+32:pos+34])[0]
    loff = struct.unpack('<I', cd[pos+42:pos+46])[0]; name = cd[pos+46:pos+46+fnl].decode('utf-8','replace')
    entries[name] = (method, csize, loff); pos += 46+fnl+exl+cml
print('zip 中央目录 entries:', len(entries))
os.makedirs(OUT, exist_ok=True)

def get_ts(cid8):
    cid = full[cid8]; tgt = f'{cid}.camera_front_wide_120fov.timestamps.parquet'
    if tgt not in entries: print(cid8, 'NOT FOUND in zip'); return None
    method, csize, loff = entries[tgt]; lh = fetch(loff, loff+30+300)
    lfnl = struct.unpack('<H', lh[26:28])[0]; lexl = struct.unpack('<H', lh[28:30])[0]
    data = fetch(loff+30+lfnl+lexl, loff+30+lfnl+lexl+csize-1)
    raw = zlib.decompress(data, -15) if method == 8 else data
    return pd.read_parquet(io.BytesIO(raw))['timestamp'].values.astype(np.int64)

for cid8 in CLIPS:
    vt = get_ts(cid8)
    if vt is None: continue
    np.save(f'{OUT}/{cid8}_fw_ts.npy', vt)
    df = pd.read_parquet(f'{EGO}/{full[cid8]}.egomotion.parquet').sort_values('timestamp')
    et = df['timestamp'].values.astype(np.int64)
    vt_rel = (vt - et[0]) / 1e6
    print(f'{cid8}: {len(vt)} 帧, sync_off={vt_rel[0]:+.4f}s, 视频覆盖 egomotion [{vt_rel[0]:.1f}, {vt_rel[-1]:.1f}]s, 帧率~{len(vt)/max(vt_rel[-1]-vt_rel[0],1e-6):.1f}fps')
print('TS_DONE')
