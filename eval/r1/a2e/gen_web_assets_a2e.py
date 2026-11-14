#!/usr/bin/env python3
"""四线静态叠加：GT 绿实线(最粗) + R1 FP16 蓝实线 + R1 INT4 橙虚线(深描边)
+ Alpamayo2-Edge 品红点线(白描边)。投影链与 gen_web_assets_int4.py 逐字相同。
cwd 无关：全部绝对路径。输出 /tmp/r1int4/web_assets_a2e/{cid8}.jpg"""
import json, numpy as np, pandas as pd, os
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

R1I = '/tmp/r1int4'; V = '/tmp/r1vid'
mcd = json.load(open(f'{R1I}/multi_clip_data.json'))
pred_fp16 = json.load(open(f'{R1I}/r1_multi_pred.json'))
pred_int4 = json.load(open(f'{R1I}/r1_multi_pred_int4.json'))
pred_a2e = json.load(open(f'{R1I}/r1_multi_pred_a2e.json'))
intr = pd.read_parquet(f'{V}/calib/camera_intrinsics.chunk_0000.parquet')
extr = pd.read_parquet(f'{V}/calib/sensor_extrinsics.chunk_0000.parquet')
CAM = 'camera_front_wide_120fov'
try: F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
except Exception: F = ImageFont.load_default()
try: FC = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 24)
except Exception: FC = F
os.makedirs(f'{R1I}/web_assets_a2e', exist_ok=True)
GT_COL = (60, 255, 120); FP_COL = (90, 170, 255); I4_COL = (255, 150, 40); A2_COL = (255, 70, 150)
DARK = (18, 20, 26, 150); LIGHT = (240, 242, 255, 170)

def draw_solid(D, uv, inf, col, wcore, whalo):
    for i in range(len(uv)-1):
        if inf[i] and inf[i+1]:
            if whalo: D.line([tuple(uv[i]), tuple(uv[i+1])], fill=col+(70,), width=whalo)
            D.line([tuple(uv[i]), tuple(uv[i+1])], fill=col+(245,), width=wcore)

def draw_dashed(D, uv, inf, col, wcore, dash, gap, ocol=None, ow=None):
    N = len(uv); i = 0
    while i < N:
        if not inf[i]: i += 1; continue
        j = i
        while j+1 < N and inf[j+1]: j += 1
        run = uv[i:j+1]
        if len(run) >= 2:
            res = [tuple(run[0])]
            for k in range(1, len(run)):
                p0 = run[k-1]; p1 = run[k]; d = float(np.hypot(p1[0]-p0[0], p1[1]-p0[1])); n = int(max(1, round(d)))
                for m in range(1, n+1): res.append((p0[0]+(p1[0]-p0[0])*m/n, p0[1]+(p1[1]-p0[1])*m/n))
            L = len(res); k = 0
            while k < L-1:
                e = min(k+dash, L-1)
                if e > k:
                    if ocol: D.line([res[k], res[e]], fill=ocol, width=ow)
                    D.line([res[k], res[e]], fill=col+(255,), width=wcore)
                k = e+gap
        i = j+1

for cid8 in mcd:
    cid = mcd[cid8]['cid']
    irow = intr.loc[(cid, CAM)]; erow = extr.loc[(cid, CAM)]
    W0, H0 = int(irow['width']), int(irow['height']); cx, cy = float(irow['cx']), float(irow['cy'])
    fw = np.array([float(irow[f'fw_poly_{k}']) for k in range(5)])
    t_vc = np.array([float(erow[c]) for c in 'xyz']); R_vc = Rotation.from_quat([float(erow[c]) for c in ['qx', 'qy', 'qz', 'qw']]).as_matrix()
    def project(p3):
        pc = (p3-t_vc) @ R_vc; X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]; fr = Z > 0.1; r3 = np.hypot(X, Y)
        th = np.arctan2(r3, Z); r = fw[0]+fw[1]*th+fw[2]*th**2+fw[3]*th**3+fw[4]*th**4
        u = np.full(len(p3), np.nan); v = np.full(len(p3), np.nan); msk = fr & (r3 > 1e-6)
        u[msk] = cx+r[msk]*X[msk]/r3[msk]; v[msk] = cy+r[msk]*Y[msk]/r3[msk]; return u, v
    ge = np.array(mcd[cid8]['gt_xyz'])
    def to3d(pxy):
        pr = np.array(pxy); pz = np.interp(pr[:, 0], ge[:, 0], ge[:, 2]); return np.hstack([pr, pz[:, None]])
    fp3 = to3d(pred_fp16[cid8]['pred_xy']); i43 = to3d(pred_int4[cid8]['pred_xy']); a23 = to3d(pred_a2e[cid8]['pred_xy'])
    img = Image.open(f'{R1I}/frames/{cid8}.png').convert('RGB'); W, H = img.size; sx, sy = W/W0, H/H0
    D = ImageDraw.Draw(img, 'RGBA')
    def prep(p3):
        u, v = project(p3); u, v = u*sx, v*sy
        inf = (~np.isnan(u)) & (u >= -50) & (u < W+50) & (v >= -50) & (v < H+50)
        return np.stack([u, v], 1), inf
    uvg, ig = prep(ge); uvf, iff = prep(fp3); uvi, ii = prep(i43); uva, ia = prep(a23)
    # 底→顶：GT 绿实(最粗) → FP16 蓝实 → INT4 橙虚(深描边) → 2-Edge 品红点(白描边)
    draw_solid(D, uvg, ig, GT_COL, 12, 20)
    draw_solid(D, uvf, iff, FP_COL, 8, 0)
    draw_dashed(D, uvi, ii, I4_COL, 6, 26, 17, ocol=DARK, ow=11)
    draw_dashed(D, uva, ia, A2_COL, 5, 8, 10, ocol=LIGHT, ow=9)
    for uv, inf, col in [(uvg, ig, GT_COL), (uvf, iff, FP_COL), (uvi, ii, I4_COL), (uva, ia, A2_COL)]:
        vis = [k for k in range(64) if inf[k]]
        if vis:
            e = uv[vis[-1]]; D.ellipse([e[0]-10, e[1]-10, e[0]+10, e[1]+10], outline=col+(255,), width=4)
    # 图例（4 条，样式与图上一致）
    D.rectangle([20, 20, 396, 176], fill=(15, 18, 26, 195))
    D.line([(36, 46), (84, 46)], fill=GT_COL+(255,), width=9); D.text((94, 33), "GT 真值", font=FC, fill=(255, 255, 255))
    D.line([(36, 80), (84, 80)], fill=FP_COL+(255,), width=7); D.text((94, 67), "R1 FP16 (10B)", font=FC, fill=(255, 255, 255))
    for xs in range(36, 85, 16): D.line([(xs, 114), (min(xs+9, 84), 114)], fill=I4_COL+(255,), width=6)
    D.text((94, 101), "R1 INT4 (10B)", font=FC, fill=(255, 255, 255))
    for xs in range(36, 85, 9): D.line([(xs, 148), (min(xs+4, 84), 148)], fill=A2_COL+(255,), width=5)
    D.text((94, 135), "Alpamayo2-Edge (2.65B)", font=FC, fill=(255, 255, 255))
    img.resize((1280, 720), Image.LANCZOS).convert('RGB').save(f'{R1I}/web_assets_a2e/{cid8}.jpg', quality=90)
    print(f'{cid8}.jpg ADE6 fp16={pred_fp16[cid8]["ade6"]:.2f} int4={pred_int4[cid8]["ade6"]:.2f} a2e={pred_a2e[cid8]["ade6"]:.2f}')
print('WEB_ASSETS_A2E_DONE')
