import subprocess, struct, io, zlib, os, json
import numpy as np, pandas as pd

TOK = open(os.path.expanduser('~/.hf_token')).read().strip()
BASE = "https://hf-mirror.com/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main/camera"
# 官方 loader _CAMERAS 顺序与索引（load_physical_aiavdataset.py）
CAMS = [("camera_cross_left_120fov", 0), ("camera_front_wide_120fov", 1),
        ("camera_cross_right_120fov", 2), ("camera_rear_left_70fov", 3),
        ("camera_rear_right_70fov", 5), ("camera_front_tele_30fov", 6)]
FRONT = (1, 6)
T0 = 5_100_000
FRONT_OFF = [-2_000_000, -1_000_000, -500_000, -100_000, 0]
OTHER_OFF = [-1_000_000, -100_000, 0]
clips = [l.strip() for l in open('/tmp/a2e/selected_clips.txt') if l.strip()]
OUT = '/tmp/a2e/frames'
FFMPEG = '/opt/homebrew/bin/ffmpeg'
os.makedirs(OUT, exist_ok=True)

_n = [0]
def fetch(url, start, end, retries=3):
    _n[0] += 1
    out = f"/tmp/a2e/_rng_{_n[0]}.bin"
    for _ in range(retries):
        r = subprocess.run(["curl", "-sL", "--max-time", "600", "-r", f"{start}-{end}",
                            "-H", f"Authorization: Bearer {TOK}", url, "-o", out, "-w", "%{http_code}"],
                           capture_output=True, text=True)
        if os.path.exists(out) and os.path.getsize(out) > 0 and r.stdout.strip() in ('206', '200'):
            d = open(out, 'rb').read(); os.remove(out); return d
    if os.path.exists(out): os.remove(out)
    raise RuntimeError(f"fetch fail {url} {start}-{end} rc={r.stdout}")

def head_size(url):
    r = subprocess.run(["curl", "-sIL", "--max-time", "30", "-H", f"Authorization: Bearer {TOK}", url],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith('x-linked-size'):
            return int(line.split(':')[1].strip())
    ls = [l for l in r.stdout.splitlines() if l.lower().startswith('content-length')]
    return int(ls[-1].split(':')[1].strip())

manifest = {}
for cam, cam_id in CAMS:
    url = f"{BASE}/{cam}/{cam}.chunk_3116.zip"
    size = head_size(url)
    print(f"== {cam} (id={cam_id}) zip={size/1e9:.2f}GB", flush=True)
    tail = fetch(url, size - 256*1024, size - 1)
    i = tail.rfind(b'\x50\x4b\x05\x06')
    cd_size = struct.unpack('<I', tail[i+12:i+16])[0]
    cd_off = struct.unpack('<I', tail[i+16:i+20])[0]
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        raise RuntimeError(f"zip64 not handled for {cam}")
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    pos = 0; entries = {}
    while pos + 46 <= len(cd) and cd[pos:pos+4] == b'\x50\x4b\x01\x02':
        method = struct.unpack('<H', cd[pos+10:pos+12])[0]
        csize = struct.unpack('<I', cd[pos+20:pos+24])[0]
        fnl = struct.unpack('<H', cd[pos+28:pos+30])[0]
        exl = struct.unpack('<H', cd[pos+30:pos+32])[0]
        cml = struct.unpack('<H', cd[pos+32:pos+34])[0]
        loff = struct.unpack('<I', cd[pos+42:pos+46])[0]
        name = cd[pos+46:pos+46+fnl].decode('utf-8', 'replace')
        entries[name] = (method, csize, loff)
        pos += 46 + fnl + exl + cml
    print(f"   CD entries: {len(entries)}", flush=True)

    def get_entry(name):
        method, csize, loff = entries[name]
        lh = fetch(url, loff, loff + 29)
        lfnl = struct.unpack('<H', lh[26:28])[0]
        lexl = struct.unpack('<H', lh[28:30])[0]
        data = fetch(url, loff + 30 + lfnl + lexl, loff + 30 + lfnl + lexl + csize - 1)
        return zlib.decompress(data, -15) if method == 8 else data

    offs = FRONT_OFF if cam_id in FRONT else OTHER_OFF
    for clip in clips:
        tsn = f"{clip}.{cam}.timestamps.parquet"
        mp4n = f"{clip}.{cam}.mp4"
        if tsn not in entries or mp4n not in entries:
            print(f"   MISSING {clip[:8]} {cam}", flush=True)
            continue
        ts = pd.read_parquet(io.BytesIO(get_entry(tsn)))['timestamp'].values.astype(np.int64)
        assert np.all(ts[1:] >= ts[:-1]), f"ts unsorted {clip[:8]} {cam}"
        req = np.array([T0 + o for o in offs], dtype=np.int64)
        # 官方 decoder 语义：取 <= 请求时刻的最近帧（validate: timestamps <= requested）
        idxs = [int(np.searchsorted(ts, r, side='right') - 1) for r in req]
        if any(ix < 0 for ix in idxs):
            print(f"   TS_UNDERFLOW {clip[:8]} {cam}", flush=True)
            continue
        actual = [int(ts[ix]) for ix in idxs]
        mp4b = get_entry(mp4n)
        tmp = f"/tmp/a2e/_x_{cam_id}.mp4"
        open(tmp, 'wb').write(mp4b)
        os.makedirs(f"{OUT}/{clip}", exist_ok=True)
        sel = "+".join(f"eq(n\\,{ix})" for ix in idxs)
        subprocess.run([FFMPEG, "-y", "-i", tmp, "-vf", f"select='{sel}'", "-vsync", "0",
                        f"{OUT}/{clip}/{cam}_%d.png"], capture_output=True)
        os.remove(tmp)
        got = sorted([f for f in os.listdir(f"{OUT}/{clip}") if f.startswith(cam + "_")],
                     key=lambda f: int(f.rsplit('_', 1)[1].split('.')[0]))
        manifest.setdefault(clip, {})[cam] = {
            'cam_id': cam_id, 'frame_idxs': idxs, 'actual_ts': actual,
            'req_ts': [int(r) for r in req], 'files': got, 'n': len(got)}
        lag = [int((r - a) / 1000) for r, a in zip(req, actual)]
        print(f"   {clip[:8]} {cam}: {len(got)}/{len(offs)} lag_ms={lag}", flush=True)

json.dump(manifest, open('/tmp/a2e/frames_manifest.json', 'w'), indent=1)
nfull = sum(1 for c, v in manifest.items() if len(v) == 6 and all(x['n'] == len(x['req_ts']) for x in v.values()))
print(f"FETCH_DONE clips_full_6cam={nfull}/{len(clips)}", flush=True)
