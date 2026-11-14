"""dgural 8 clips 的 Alpamayo2-Edge 四线 demo 取数（Mac 侧）。
gated 部分走 hf-mirror + ~/.hf_token（Range 部分下载）；dgural 公开样本匿名可下。
产物：/tmp/r1vid/{calib,ego,a2e_cams,a2e_frames,ts} + nav_dgural.json + manifest_a2e_dgural.json
"""
import subprocess, struct, io, zlib, os, json, zipfile
import numpy as np, pandas as pd

TOK = open(os.path.expanduser('~/.hf_token')).read().strip()
GATED = "https://hf-mirror.com/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main"
DGURAL = "https://hf-mirror.com/datasets/dgural/PhysicalAI-Autonomous-Vehicles-Sample/resolve/main/data"
V = '/tmp/r1vid'
CAMS = [("camera_cross_left_120fov", "cl", 0), ("camera_front_wide_120fov", "fw", 1),
        ("camera_cross_right_120fov", "cr", 2), ("camera_rear_left_70fov", "rl", 3),
        ("camera_rear_right_70fov", "rr", 5), ("camera_front_tele_30fov", "ft", 6)]
FRONT = (1, 6)
FRONT_OFF = [-2_000_000, -1_000_000, -500_000, -100_000, 0]
OTHER_OFF = [-1_000_000, -100_000, 0]
FULL = {  # 8 clips 全 ID（navigation.parquet 核实）
    'ecafce84': 'ecafce84-447d-43a6-aff1-b385fbc71f15',
    '76f4a312': '76f4a312-4b5a-4bce-8499-b249ae70d0a0',
    '46003675': '46003675-4b4e-4c0f-ae54-3f7622bddf6a',
    '1f7ef408': '1f7ef408-e2fc-4f9c-9a54-c81fbb854141',
    'fa83bcb8': 'fa83bcb8-ea31-4dbb-b447-4fb458f5984b',
    '058bf960': '058bf960-f531-4ae6-8b24-455d7683676e',
    '1fcf2fef': '1fcf2fef-279b-4490-adc6-9b7d1ff0118d',
    '0dd3eb7f': '0dd3eb7f-47a1-40d5-a228-d06ea1288839'}
STATIC_T0 = {'ecafce84': 9.0, '76f4a312': 8.0, '46003675': 7.0,
             '1f7ef408': 12.5, 'fa83bcb8': 4.0, '058bf960': 7.0}
ROLL_T0 = [7.5, 13.5]  # t0=1.5 不满足 2-Edge 前相机 -2s 历史窗，弃（页面披露）
FFMPEG = '/opt/homebrew/bin/ffmpeg'

def curl(url, out, token=True, retries=4, maxtime=600):
    for a in range(retries):
        args = ['curl', '-sL', '--max-time', str(maxtime)]
        if token: args += ['-H', f'Authorization: Bearer {TOK}']
        args += [url, '-o', out, '-w', '%{http_code}']
        r = subprocess.run(args, capture_output=True, text=True)
        if os.path.exists(out) and os.path.getsize(out) > 0 and r.stdout.strip() in ('200', '206'):
            return True
    return False

def fetch_range(url, start, end, retries=4):
    for a in range(retries):
        outf = f"{V}/_rng_{os.getpid()}_{a}.bin"
        r = subprocess.run(['curl', '-sL', '--max-time', '900', '-r', f'{start}-{end}',
                            '-H', f'Authorization: Bearer {TOK}', url, '-o', outf, '-w', '%{http_code}'],
                           capture_output=True, text=True)
        if os.path.exists(outf) and os.path.getsize(outf) > 0 and r.stdout.strip() in ('200', '206'):
            d = open(outf, 'rb').read(); os.remove(outf); return d
        if os.path.exists(outf): os.remove(outf)
    raise RuntimeError(f'range fail {url} {start}-{end}')

def head_size(url):
    r = subprocess.run(['curl', '-sIL', '--max-time', '60', '-H', f'Authorization: Bearer {TOK}', url],
                       capture_output=True, text=True)
    ls = [l for l in r.stdout.splitlines() if l.lower().startswith('content-length')]
    return int(ls[-1].split(':')[1].strip())

def zip_entries(url, size):
    tail = fetch_range(url, size - 256*1024, size - 1)
    i = tail.rfind(b'\x50\x4b\x05\x06')
    cd_size = struct.unpack('<I', tail[i+12:i+16])[0]
    cd_off = struct.unpack('<I', tail[i+16:i+20])[0]
    cd = fetch_range(url, cd_off, cd_off + cd_size - 1)
    pos = 0; entries = {}
    while pos + 46 <= len(cd) and cd[pos:pos+4] == b'\x50\x4b\x01\x02':
        method = struct.unpack('<H', cd[pos+10:pos+12])[0]
        csize = struct.unpack('<I', cd[pos+20:pos+24])[0]
        fnl = struct.unpack('<H', cd[pos+28:pos+30])[0]
        exl = struct.unpack('<H', cd[pos+30:pos+32])[0]
        cml = struct.unpack('<H', cd[pos+32:pos+34])[0]
        loff = struct.unpack('<I', cd[pos+42:pos+46])[0]
        name = cd[pos+46:pos+46+fnl].decode('utf-8', 'replace')
        if loff == 0xFFFFFFFF or csize == 0xFFFFFFFF:  # ZIP64（>2GB zip）
            ex = cd[pos+46+fnl:pos+46+fnl+exl]
            i = 0; found = False
            while i + 4 <= len(ex):
                tag = struct.unpack('<H', ex[i:i+2])[0]
                sz = struct.unpack('<H', ex[i+2:i+4])[0]
                if tag == 1:
                    vals = ex[i+4:i+4+sz]; j = 0
                    usize = struct.unpack('<I', cd[pos+24:pos+28])[0]
                    if usize == 0xFFFFFFFF: j += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack('<Q', vals[j:j+8])[0]; j += 8
                    if loff == 0xFFFFFFFF: loff = struct.unpack('<Q', vals[j:j+8])[0]; j += 8
                    found = True; break
                i += 4 + sz
            assert found, f'zip64 extra not found for {name}'
        entries[name] = (method, csize, loff)
        pos += 46 + fnl + exl + cml
    return entries

def zip_read(url, entries, name):
    method, csize, loff = entries[name]
    lh = fetch_range(url, loff, loff + 29)
    lfnl = struct.unpack('<H', lh[26:28])[0]
    lexl = struct.unpack('<H', lh[28:30])[0]
    data = fetch_range(url, loff + 30 + lfnl + lexl, loff + 30 + lfnl + lexl + csize - 1)
    return zlib.decompress(data, -15) if method == 8 else data

# ---------- 0) nav（真实标注，≤t0 最近行在推理侧选） ----------
nav_all = pd.read_parquet('/tmp/a2e/navigation.parquet')
nav_out = {}
for c8, cid in FULL.items():
    sub = nav_all[nav_all.clip_id == cid].sort_values('timestamp')
    nav_out[c8] = {'clip_id': cid, 'rows': [[int(r.timestamp), r.nav_text] for r in sub.itertuples()]}
json.dump(nav_out, open(f'{V}/nav_dgural.json', 'w'), indent=1)
print(f'[0] nav: {", ".join(f"{c8}:{len(v[chr(114)+chr(111)+chr(119)+chr(115)])}" for c8, v in nav_out.items())}', flush=True)

# ---------- 1) 标定（gated，小文件直下） ----------
os.makedirs(f'{V}/calib', exist_ok=True)
for sub, f in [('camera_intrinsics', 'camera_intrinsics.chunk_0000.parquet'),
               ('sensor_extrinsics', 'sensor_extrinsics.chunk_0000.parquet')]:
    assert curl(f'{GATED}/calibration/{sub}/{f}', f'{V}/calib/{f}'), f'calib fail {f}'
print('[1] calib OK', flush=True)

# ---------- 2) egomotion chunk_0000（36MB 全量）→ 8 clips ----------
os.makedirs(f'{V}/ego', exist_ok=True)
ego_zip = f'{V}/_ego_chunk0000.zip'
assert curl(f'{GATED}/labels/egomotion/egomotion.chunk_0000.zip', ego_zip), 'ego zip fail'
with zipfile.ZipFile(ego_zip) as z:
    for c8, cid in FULL.items():
        name = f'{cid}.egomotion.parquet'
        with z.open(name) as src, open(f'{V}/ego/{cid}.egomotion.parquet', 'wb') as dst:
            dst.write(src.read())
os.remove(ego_zip)
print('[2] ego 8 clips OK', flush=True)

# ---------- 3) dgural 公开 front_wide 视频（匿名） ----------
for c8, cid in FULL.items():
    out = f'{V}/{c8}_fw.mp4'
    if os.path.exists(out) and os.path.getsize(out) > 1e6: continue
    assert curl(f'{DGURAL}/{cid}.camera_front_wide_120fov.mp4', out, token=False), f'fw video fail {c8}'
    print(f'[3] fw video {c8} {os.path.getsize(out)//1024//1024}MB', flush=True)

# ---------- 4) 6 路相机 gated Range：mp4 + ts ----------
os.makedirs(f'{V}/a2e_cams', exist_ok=True)
ts_cache = {}
for cam, cshort, cam_id in CAMS:
    url = f'{GATED}/camera/{cam}/{cam}.chunk_0000.zip'
    size = head_size(url)
    entries = zip_entries(url, size)
    print(f'[4] {cam} zip={size/1e9:.2f}GB entries={len(entries)}', flush=True)
    for c8, cid in FULL.items():
        mp4_out = f'{V}/a2e_cams/{c8}_{cshort}.mp4'
        if not (os.path.exists(mp4_out) and os.path.getsize(mp4_out) > 1e6):
            data = zip_read(url, entries, f'{cid}.{cam}.mp4')
            open(mp4_out, 'wb').write(data)
        ts = pd.read_parquet(io.BytesIO(zip_read(url, entries, f'{cid}.{cam}.timestamps.parquet')))['timestamp'].values.astype(np.int64)
        assert np.all(ts[1:] >= ts[:-1])
        ts_cache[(c8, cam_id)] = ts
        np.save(f'{V}/a2e_cams/{c8}_{cshort}_ts.npy', ts)
    if cam == 'camera_front_wide_120fov':  # render_roll 需要的 fw 帧时间戳
        os.makedirs(f'{V}/ts', exist_ok=True)
        for c8 in FULL:
            np.save(f'{V}/ts/{c8}_fw_ts.npy', ts_cache[(c8, 1)])

# ---------- 5) 14 锚点 × 22 帧抽取 ----------
anchors = [(c8, t0, c8) for c8, t0 in STATIC_T0.items()]
for c8 in ['ecafce84', '1fcf2fef', '0dd3eb7f', '1f7ef408']:
    for t0 in ROLL_T0:
        anchors.append((c8, t0, f'{c8}_t{t0}'))
manifest = {}
os.makedirs(f'{V}/a2e_frames', exist_ok=True)
for c8, t0_s, tag in anchors:
    t0 = int(t0_s * 1e6)
    os.makedirs(f'{V}/a2e_frames/{tag}', exist_ok=True)
    ent = {}
    for cam, cshort, cam_id in CAMS:
        ts = ts_cache[(c8, cam_id)]
        offs = FRONT_OFF if cam_id in FRONT else OTHER_OFF
        req = np.array([t0 + o for o in offs], dtype=np.int64)
        idxs = [int(np.searchsorted(ts, r, side='right') - 1) for r in req]
        assert all(ix >= 0 for ix in idxs), f'{tag} {cam} t0 过早'
        actual = [int(ts[ix]) for ix in idxs]
        sel = '+'.join(f'eq(n\\,{ix})' for ix in idxs)
        subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', f'{V}/a2e_cams/{c8}_{cshort}.mp4',
                        '-vf', f"select='{sel}'", '-vsync', '0',
                        f'{V}/a2e_frames/{tag}/{cam}_%d.png'], check=True)
        got = sorted([f for f in os.listdir(f'{V}/a2e_frames/{tag}') if f.startswith(cam + '_')],
                     key=lambda f: int(f.rsplit('_', 1)[1].split('.')[0]))
        assert len(got) == len(offs), f'{tag} {cam} got {len(got)}'
        ent[cam] = {'cam_id': cam_id, 'actual_ts': actual, 'req_ts': [int(r) for r in req], 'files': got}
    manifest[tag] = {'cid8': c8, 'cid': FULL[c8], 't0': t0, 'cams': ent}
    print(f'[5] {tag} 22 frames OK', flush=True)
json.dump(manifest, open(f'{V}/manifest_a2e_dgural.json', 'w'), indent=1)
print(f'FETCH_DGURAL_A2E_DONE anchors={len(manifest)}', flush=True)
