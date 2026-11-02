#!/usr/bin/env python3
"""滚动预测积分：12 条 raw output_trajectory (accel,kappa) → BEV pred_xy。
compute_minade.py 的纯 numpy 移植（Mac 无 torch）；先用 multi demo 的 6 clip
对拍 goat torch 结果 r1_multi_pred.json（max|Δ|<1e-3 才放行），再积分滚动 12 条。
roll_clip_data.json 缺 ego_history_rot → 从 egomotion 重算（与 extract_roll.py 同一 ipf 逻辑），
并用存下的 ego_history_xyz 校验重算一致。
"""
import json, glob, os
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

ROLL = '/tmp/r1vid/roll'; EGO = '/tmp/r1vid/ego'

# --- compute_minade.py 常数（逐行同步）---
DT = 0.1
ACCEL_MEAN = 0.029052734375; ACCEL_STD = 0.6796875
CURVATURE_MEAN = 0.0002689361572265625; CURVATURE_STD = 0.026123046875
V_LAMBDA = 1e-06; V_RIDGE = 0.0001

def estimate_v0(hxyz, hrot):
    """3阶 Tikhonov(λ/dt⁶=1.0)+ridge(1e-4) 最小二乘 → v[-1]=t0 时刻速度。
    方程组: v[i]·cosθ + v[i+1]·cosθᵢ₊₁ = (2/DT)·Δxᵢ（梯形位移），θ 来自 rot 的 yaw。"""
    hxyz = np.asarray(hxyz, dtype=np.float64); hrot = np.asarray(hrot, dtype=np.float64)
    dxy = hxyz[1:, :2] - hxyz[:-1, :2]                       # (N,2), N=15
    theta = np.unwrap(np.arctan2(hrot[:, 1, 0], hrot[:, 0, 0]))
    N = len(dxy)
    b = (2 / DT * dxy).reshape(-1)                           # row-major [gx0,gy0,gx1,gy1,...]
    A = np.zeros((2 * N, N + 1))
    r = np.arange(N)
    A[2*r,   r  ] = np.cos(theta[:-1]); A[2*r,   r+1] = np.cos(theta[1:])
    A[2*r+1, r  ] = np.sin(theta[:-1]); A[2*r+1, r+1] = np.sin(theta[1:])
    M = N + 1
    D = np.zeros((M - 3, M))
    D[r[:M-3], r[:M-3]] = -1; D[r[:M-3], r[:M-3]+1] = 3
    D[r[:M-3], r[:M-3]+2] = -3; D[r[:M-3], r[:M-3]+3] = 1
    DTD = (V_LAMBDA / DT**6) * D.T @ D
    lhs = A.T @ A + DTD + V_RIDGE * np.eye(M)
    v = np.linalg.solve(lhs, A.T @ b)
    return float(v[-1])

def integrate(ot, hxyz, hrot):
    """(64,2) 归一化 (accel,kappa) → BEV pred_xy (64,2)，单车模型 trapezoid cumsum。"""
    sa = np.asarray(ot, dtype=np.float64).reshape(64, 2)
    v0 = estimate_v0(hxyz, hrot)
    accel = sa[:, 0] * ACCEL_STD + ACCEL_MEAN
    kappa = sa[:, 1] * CURVATURE_STD + CURVATURE_MEAN
    vel = np.concatenate([[v0], v0 + np.cumsum(accel * DT)])                    # (65,)
    th = np.concatenate([[0.0], np.cumsum(kappa * vel[:-1] * DT)
                         + np.cumsum(kappa * accel * 0.5 * DT**2)])             # (65,)
    hdt = 0.5 * DT
    x = np.cumsum(vel[:-1]*np.cos(th[:-1])*hdt) + np.cumsum(vel[1:]*np.cos(th[1:])*hdt)
    y = np.cumsum(vel[:-1]*np.sin(th[:-1])*hdt) + np.cumsum(vel[1:]*np.sin(th[1:])*hdt)
    return np.stack([x, y], -1), v0

# ============ 1) 对拍：numpy vs goat torch（multi demo 6 clip）============
P = f'{ROLL}/parity'
mout = json.load(open(f'{P}/r1_multi_out.json'))
mcd = json.load(open(f'{P}/multi_clip_data.json'))
mpred = json.load(open(f'{P}/r1_multi_pred.json'))
minp = json.load(open(f'{P}/r1_multi_in.json'))
req_ids = [r['id'] for r in minp['requests']]
print('=== 对拍（numpy 移植 vs torch r1_multi_pred.json）===')
worst = 0.0
for i, resp in enumerate(mout['responses']):
    j = resp.get('request_idx', i)
    rid = resp.get('id') or req_ids[j]
    d = mcd[rid]
    pred, v0 = integrate(resp['output_trajectory'], d['ego_history_xyz'], d['ego_history_rot'])
    ref = np.array(mpred[rid]['pred_xy'])
    dmax = float(np.abs(pred - ref).max())
    worst = max(worst, dmax)
    print(f'  {rid}: max|Δ|={dmax:.2e} m  v0={v0:.4f} (torch {mpred[rid]["v0"]:.4f})')
assert worst < 1e-3, f'对拍失败 max|Δ|={worst:.2e} —— numpy 移植不可用'
print(f'对拍通过：6/6 clip max|Δ|={worst:.2e} m < 1e-3\n')

# ============ 2) 滚动 12 条：重算 rot + 积分 ============
full = {os.path.basename(f).split('.')[0][:8]: os.path.basename(f).split('.')[0]
        for f in glob.glob(f'{EGO}/*.parquet')}

def ipf(cid):
    df = pd.read_parquet(f'{EGO}/{cid}.egomotion.parquet').sort_values('timestamp').reset_index(drop=True)
    ts = df['timestamp'].values.astype(np.float64)/1e6; trel = ts - ts[0]
    P_ = df[['x','y','z']].values
    Rall = Rotation.from_quat(df[['qx','qy','qz','qw']].values).as_matrix()
    def ip(t):
        p = np.array([np.interp(t, trel, P_[:, k]) for k in range(3)])
        return p, Rall[int(np.argmin(np.abs(trel - t)))]
    return ip

out = json.load(open(f'{ROLL}/roll_out.json'))
inp = json.load(open(f'{ROLL}/roll_in.json'))
clip_data = json.load(open(f'{ROLL}/roll_clip_data.json'))
tags = [r['id'] for r in inp['requests']]

result = {}
print('=== 滚动 12 条积分（pred末 vs gt末 看方向；lat+=左）===')
for i, resp in enumerate(out['responses']):
    j = resp.get('request_idx', i)
    tag = tags[j]
    d = clip_data[tag]
    ip = ipf(d['cid'])
    p0, R0 = ip(d['t0'])
    th = d['t0'] - 1.5 + np.arange(16) * 0.1
    hxyz_w = np.array([ip(t)[0] for t in th])
    he = (R0.T @ (hxyz_w - p0).T).T                       # t0 ego 系
    hrot = np.array([R0.T @ ip(t)[1] for t in th])
    # 存下的 t0 是 round(t0a,4)（0.1ms 精度），重算用舍入值 → 位移差≈v×1e-4s≤1e-3m，物理可忽略
    dz = float(np.abs(he - np.array(d['ego_history_xyz'])).max())
    assert dz < 1e-3, f'{tag}: 重算 ego_history 与存下不一致 max|Δ|={dz:.2e}'
    pred, v0 = integrate(resp['output_trajectory'], he, hrot)
    gt = np.array(d['gt_xyz'])[:, :2]
    dd = np.linalg.norm(pred - gt, axis=1)
    cot = resp.get('output_text', '')
    result[tag] = {'pred_xy': pred.tolist(), 'v0': v0, 'cot': cot,
                   'ade3': float(dd[:30].mean()), 'ade6': float(dd.mean()), 'fde': float(dd[-1])}
    print(f'  {tag}: v0={v0:5.2f} ADE3={dd[:30].mean():5.2f} ADE6={dd.mean():5.2f} FDE={dd[-1]:5.2f} '
          f'| pred末=[{pred[-1,0]:6.1f},{pred[-1,1]:+6.1f}] gt末=[{gt[-1,0]:6.1f},{gt[-1,1]:+6.1f}]')
    print(f'      CoT: {cot}')
json.dump(result, open(f'{ROLL}/roll_pred.json', 'w'))
ades = [r['ade6'] for r in result.values()]
print(f'\n12 条平均 ADE@6.4s={np.mean(ades):.2f} m  平均FDE={np.mean([r["fde"] for r in result.values()]):.2f} m')
print('SAVED roll_pred.json')
