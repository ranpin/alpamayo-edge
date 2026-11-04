#!/usr/bin/env python3
"""滚动预测视频：4 clip × ~19s，R1 每 ~6s（=真实单次推理耗时）重预测一次。
蓝=R1 预测、绿=GT 真值，二者同锚 t₀（同窗 [t₀,t₀+6.4]，重叠=预测精度 ADE），
随车前进被一起"吃掉"、到下个 t₀ 一起刷新 → 锯齿式"边开边预测"。HUD 常显单次推理耗时+事件级非实时+速度。

坐标链（与已验证 gen_vid.py/gen_web_assets.py 同源）：
  pred/gt 在 t₀_k ego 系 → 世界锚定 p_world=p_ego@R(t₀).T+P(t₀)
  → 每帧 t 变当前车体系 p_ego(t)=(p_world−P(t))@R(t) → 相机 (p_ego−t_vc)@R_vc → 鱼眼。
速度取 egomotion vx/vy/vz 真值。CoT 默认不叠（部分锚点文本与轨迹方向矛盾，见 integrate_roll 输出）。
"""
import json, os, sys, glob, subprocess, shutil
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROLL = '/tmp/r1vid/roll'; EGO = '/tmp/r1vid/ego'; CALIB = '/tmp/r1vid/calib'
CAM = 'camera_front_wide_120fov'; FPS = 30
T_START, T_END = 0.8, 19.5
OUT_W, OUT_H = 1280, 720
SHOW_COT = False
GT_COL = (60, 255, 120); R1_COL = (90, 170, 255); I4_COL = (255, 150, 40)
ALL_CLIPS = ['0dd3eb7f', 'ecafce84', '1fcf2fef', '1f7ef408']

pred_data = json.load(open(f'{ROLL}/roll_pred.json'))
pred_int4 = json.load(open(f'{ROLL}/roll_pred_int4.json'))
clip_data = json.load(open(f'{ROLL}/roll_clip_data.json'))
intr = pd.read_parquet(f'{CALIB}/camera_intrinsics.chunk_0000.parquet')
extr = pd.read_parquet(f'{CALIB}/sensor_extrinsics.chunk_0000.parquet')

def font(sz):
    for f in ['/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Medium.ttc',
              '/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()
F_LEG, F_HUD, F_BIG, F_LBL, F_NOTE = font(22), font(25), font(40), font(23), font(19)

def render_clip(cid8, debug=False):
    tags = sorted([k for k in clip_data if k.startswith(cid8 + '_')], key=lambda k: clip_data[k]['t0'])
    assert len(tags) == 3, f'{cid8}: 期望 3 个锚点, 得 {tags}'
    cid = clip_data[tags[0]]['cid']
    fw_mp4 = f'{ROLL}/{cid8}_fw.mp4'
    assert os.path.exists(fw_mp4), f'缺 fw 视频 {fw_mp4}'

    df = pd.read_parquet(f'{EGO}/{cid}.egomotion.parquet').sort_values('timestamp').reset_index(drop=True)
    ts = df['timestamp'].values.astype(np.float64) / 1e6; trel = ts - ts[0]
    P = df[['x', 'y', 'z']].values; V = df[['vx', 'vy', 'vz']].values
    Rall = Rotation.from_quat(df[['qx', 'qy', 'qz', 'qw']].values).as_matrix()  # ego→world
    TEND_DATA = trel[-1]
    def pose_at(t):
        p = np.array([np.interp(t, trel, P[:, k]) for k in range(3)])
        v = np.array([np.interp(t, trel, V[:, k]) for k in range(3)])
        return p, v, Rall[int(np.argmin(np.abs(trel - t)))]

    irow = intr.loc[(cid, CAM)]; erow = extr.loc[(cid, CAM)]
    W0, H0 = int(irow['width']), int(irow['height'])
    cx, cy = float(irow['cx']), float(irow['cy'])
    fwp = np.array([float(irow[f'fw_poly_{k}']) for k in range(5)])
    t_vc = np.array([float(erow[c]) for c in 'xyz'])
    R_vc = Rotation.from_quat([float(erow[c]) for c in ['qx', 'qy', 'qz', 'qw']]).as_matrix()
    assert abs(R_vc @ [0, 0, 1] - [1, 0, 0]).max() < 0.2, f'R_vc 朝向异常'
    def project(p_world, p_t, R_t, W, H):
        pc = ((p_world - p_t) @ R_t - t_vc) @ R_vc          # 世界→当前车体系→相机系
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]; r3 = np.hypot(X, Y)
        th = np.arctan2(r3, Z)
        r = fwp[0] + fwp[1]*th + fwp[2]*th**2 + fwp[3]*th**3 + fwp[4]*th**4
        sx, sy = W / W0, H / H0
        u = np.full(len(p_world), np.nan); v = np.full(len(p_world), np.nan)
        m = (Z > 0.1) & (r3 > 1e-6)
        u[m] = (cx + r[m]*X[m]/r3[m])*sx; v[m] = (cy + r[m]*Y[m]/r3[m])*sy
        return u, v
    def draw_traj(D, p_world, p_t, R_t, W, H, col, bright=False):
        u, v = project(p_world, p_t, R_t, W, H)
        inf = (~np.isnan(u)) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ga, gw, cw = (110, 20, 8) if bright else (80, 15, 6)   # 光晕alpha/宽+核心宽，同已验证 gen_vid.py 口径
        for i in range(len(p_world) - 1):
            if inf[i] and inf[i+1]:
                D.line([(u[i], v[i]), (u[i+1], v[i+1])], fill=col+(ga,), width=gw)
                D.line([(u[i], v[i]), (u[i+1], v[i+1])], fill=col+(245,), width=cw)
        for i in range(0, len(p_world), 8):
            if inf[i]: D.ellipse([u[i]-5, v[i]-5, u[i]+5, v[i]+5], fill=(255,255,255,255), outline=col+(255,), width=2)
        vis = [i for i in range(len(p_world)) if inf[i]]
        if vis: D.ellipse([u[vis[-1]]-11, v[vis[-1]]-11, u[vis[-1]]+11, v[vis[-1]]+11], outline=col+(255,), width=4)
        return int(sum(inf))
    def draw_traj_dashed(D, p_world, p_t, R_t, W, H, col, dash=16, gap=11):
        # INT4 橙虚线：与 FP16 蓝实线最强样式区分，近重叠也能一眼分辨（空隙露蓝）
        u, v = project(p_world, p_t, R_t, W, H)
        inf = (~np.isnan(u)) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uv = np.stack([u, v], 1); N = len(uv); a = 0
        while a < N:
            if not inf[a]: a += 1; continue
            b = a
            while b+1 < N and inf[b+1]: b += 1
            run = uv[a:b+1]
            if len(run) >= 2:
                res = [tuple(run[0])]
                for k in range(1, len(run)):
                    p0 = run[k-1]; p1 = run[k]; dd = float(np.hypot(p1[0]-p0[0], p1[1]-p0[1])); nn = int(max(1, round(dd)))
                    for m in range(1, nn+1): res.append((p0[0]+(p1[0]-p0[0])*m/nn, p0[1]+(p1[1]-p0[1])*m/nn))
                L = len(res); k = 0
                while k < L-1:
                    e = min(k+dash, L-1)
                    if e > k:
                        D.line([res[k], res[e]], fill=(18,20,26,150), width=10)
                        D.line([res[k], res[e]], fill=col+(255,), width=5)
                    k = e+gap
            a = b+1
        vis = [q for q in range(N) if inf[q]]
        if vis:
            ex, ey = uv[vis[-1]]; D.ellipse([ex-10, ey-10, ex+10, ey+10], outline=col+(255,), width=4)

    # 3 条预测 → 世界锚定（pred 赋 GT 路面高程 z；gt 同锚世界化，与 pred 同窗公平对比）
    preds = []
    for tag in tags:
        pd_, cd_ = pred_data[tag], clip_data[tag]
        pd4_ = pred_int4[tag]
        pred = np.array(pd_['pred_xy']); gt = np.array(cd_['gt_xyz'])
        predz = np.interp(pred[:, 0], gt[:, 0], gt[:, 2])
        pred3 = np.hstack([pred, predz[:, None]])
        pred4 = np.array(pd4_['pred_xy']); pred4z = np.interp(pred4[:, 0], gt[:, 0], gt[:, 2])
        pred43 = np.hstack([pred4, pred4z[:, None]])
        p0, _, R0 = pose_at(cd_['t0'])
        preds.append({'t0': cd_['t0'], 'pw': pred3 @ R0.T + p0, 'pw4': pred43 @ R0.T + p0, 'gtw': gt @ R0.T + p0,
                      'cot': pd_['cot'], 'ade6': pd_['ade6'], 'ade6_4': pd4_['ade6'], 'cover': (cd_['t0'], cd_['t0'] + 6.4)})

    vt = np.load(f'{ROLL}/ts/{cid8}_fw_ts.npy').astype(np.int64)
    et = df['timestamp'].values.astype(np.int64); vt_rel = (vt - et[0]) / 1e6
    i_start = int(np.searchsorted(vt_rel, T_START))
    i_end = min(int(np.searchsorted(vt_rel, T_END)) - 1, len(vt_rel) - 1)

    frames_dir = f'{ROLL}/rframes_{cid8}'; out_dir = f'{ROLL}/rout_{cid8}'
    for d in (frames_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', fw_mp4, '-vf',
                    f"select='between(n\\,{i_start}\\,{i_end})'", '-vsync', '0',
                    f'{frames_dir}/f_%04d.png'], check=True)
    fnames = sorted(glob.glob(f'{frames_dir}/f_*.png'))
    print(f'[{cid8}] 帧 {len(fnames)} (video {i_start}..{i_end}, t {vt_rel[i_start]:.2f}..{vt_rel[i_end]:.2f}s) '
          f'锚点 t0={[round(p["t0"],2) for p in preds]}', flush=True)

    prev_t0 = None; flash_until = -1; n_blue = 0; speeds = []
    for oi, fn in enumerate(fnames):
        t = vt_rel[i_start + oi]
        p_t, v_t, R_t = pose_at(t)
        speed = float(np.linalg.norm(v_t)); speeds.append(speed)
        img = Image.open(fn).convert('RGB').resize((OUT_W, OUT_H), Image.LANCZOS)
        W, H = img.size; D = ImageDraw.Draw(img, 'RGBA')
        # 绿GT+蓝R1 同锚 t₀_k（同窗公平对比，重叠=精度），随车一起被吃掉、到下个 t₀ 一起刷新
        active = [pr for pr in preds if pr['t0'] <= t + 1e-6]
        cur = active[-1] if active else None
        if cur is not None:
            if prev_t0 is None or cur['t0'] != prev_t0: flash_until = oi + 18
            draw_traj(D, cur['gtw'], p_t, R_t, W, H, GT_COL)
            nb = draw_traj(D, cur['pw'], p_t, R_t, W, H, R1_COL, bright=(oi < flash_until))
            draw_traj_dashed(D, cur['pw4'], p_t, R_t, W, H, I4_COL)
            if nb > 0: n_blue += 1
            prev_t0 = cur['t0']
        # ---- HUD ----
        D.rectangle([16, 16, 576, 158], fill=(15, 18, 26, 185))               # 图例
        D.line([(30, 38), (64, 38)], fill=GT_COL+(255,), width=6); D.text((72, 26), 'GT 真值 · 同锚 t₀', font=F_LEG, fill=(255,255,255))
        D.line([(30, 68), (64, 68)], fill=R1_COL+(255,), width=6); D.text((72, 56), 'R1 FP16 · 同锚 t₀', font=F_LEG, fill=(255,255,255))
        for xs in range(30, 65, 12): D.line([(xs, 98), (min(xs+7, 64), 98)], fill=I4_COL+(255,), width=5)
        D.text((72, 86), 'R1 INT4 · 同锚 t₀', font=F_LEG, fill=(255,255,255))
        D.text((30, 124), 'ADE@6.4s = 同窗内预测与GT的平均欧氏距离 (m)', font=F_NOTE, fill=(190, 200, 220))
        D.rectangle([W-306, 16, W-16, 100], fill=(15, 18, 26, 185))          # 速度
        D.text((W-294, 20), f'{speed*3.6:5.1f} km/h', font=F_BIG, fill=(255, 220, 120))
        D.text((W-294, 68), f'{speed:4.1f} m/s   t = {t:4.1f} s', font=F_LBL, fill=(255, 255, 255))
        D.rectangle([16, H-46, W-16, H-14], fill=(15, 18, 26, 195))          # 常显延迟标签
        D.text((28, H-40), 'Alpamayo-R1 10B · Jetson Orin · 单次推理 FP16 ~5.7s / INT4 ~4.2s · 每 ~6s 重预测(按FP16节奏) · 事件级非实时',
               font=F_LBL, fill=(205, 214, 232))
        if cur is not None:                                                   # 活跃预测 + 刷新闪烁
            msg = f'#{len(active)} 同锚 t₀={cur["t0"]:.1f}s · ADE@6.4s FP16={cur["ade6"]:.2f}m / INT4={cur["ade6_4"]:.2f}m'
            if oi < flash_until:
                D.rectangle([16, H-88, 940, H-52], fill=(60, 45, 12, 210))
                D.text((28, H-82), f'⟳ 重新预测 · 新锚点 t₀={cur["t0"]:.1f}s · ADE@6.4s FP16={cur["ade6"]:.2f}m / INT4={cur["ade6_4"]:.2f}m', font=F_HUD, fill=(255, 235, 130))
            else:
                D.rectangle([16, H-88, 940, H-52], fill=(15, 18, 26, 185))
                D.text((28, H-82), msg, font=F_HUD, fill=(185, 200, 225))
            if SHOW_COT:
                D.rectangle([16, H-124, W-16, H-92], fill=(15, 18, 26, 170))
                D.text((28, H-118), f'CoT: {cur["cot"][:88]}', font=F_LBL, fill=(220, 220, 220))
        else:
            D.rectangle([16, H-88, 720, H-52], fill=(15, 18, 26, 185))
            D.text((28, H-82), f'R1 首次推理进行中…（待锚定 t₀={preds[0]["t0"]:.1f}s 采帧）', font=F_HUD, fill=(185, 200, 225))
        img.save(f'{out_dir}/v_%04d.png' % oi)
        if debug and oi in (0, i_start and 0):
            pass
    if debug:
        # 自检：锚点帧处 pred 首点应落在画面下半部（车前方近处）
        for pr in preds:
            t = pr['t0']; p_t, _, R_t = pose_at(t)
            u, v = project(pr['pw'][:1], p_t, R_t, OUT_W, OUT_H)
            print(f'    [dbg] t0={t:.2f} pred[0] 投影 u={u[0]:.0f} v={v[0]:.0f} (画面 {OUT_W}x{OUT_H}, 应偏下)', flush=True)
        print(f'    [dbg] 速度 {min(speeds):.1f}~{max(speeds):.1f} m/s | 有蓝帧 {n_blue}/{len(fnames)}', flush=True)

    out_mp4 = f'{ROLL}/{cid8}_roll.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', f'{out_dir}/v_%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out_mp4], check=True)
    print(f'[{cid8}] mp4 {os.path.getsize(out_mp4)//1024}KB → {out_mp4}', flush=True)
    shutil.rmtree(frames_dir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    dbg = '--debug' in args
    clips = [a for a in args if not a.startswith('--')] or ALL_CLIPS
    for c in clips:
        render_clip(c, debug=dbg)
    print('RENDER_DONE', clips)
