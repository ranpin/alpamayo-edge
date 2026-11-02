#!/usr/bin/env python3
"""验证"蓝落后绿"是设计的预测年龄(t−t₀)而非渲染 bug。
判据：落后量随 t−t₀ 线性增长、刷新瞬间归零(≈0) → 设计；刷新时仍恒定落后 → bug。
对 0dd3eb7f 第一个周期(t₀=1.5)逐 0.5s 测：蓝尖=预测末点(t₀+6.4)、绿尖=GT(t+6.3) 的投影像素。
直行路远处→画面高处(v 小)，故 v蓝尖 − v绿尖 >0 = 蓝尖更近(落后)的像素量。"""
import json, glob, os
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

ROLL='/tmp/r1vid/roll'; EGO='/tmp/r1vid/ego'; CALIB='/tmp/r1vid/calib'
CAM='camera_front_wide_120fov'; OUT_W,OUT_H=1280,720
cid8='0dd3eb7f'
pred_data=json.load(open(f'{ROLL}/roll_pred.json')); clip_data=json.load(open(f'{ROLL}/roll_clip_data.json'))
intr=pd.read_parquet(f'{CALIB}/camera_intrinsics.chunk_0000.parquet'); extr=pd.read_parquet(f'{CALIB}/sensor_extrinsics.chunk_0000.parquet')
tags=sorted([k for k in clip_data if k.startswith(cid8+'_')],key=lambda k:clip_data[k]['t0'])
cid=clip_data[tags[0]]['cid']
df=pd.read_parquet(f'{EGO}/{cid}.egomotion.parquet').sort_values('timestamp').reset_index(drop=True)
ts=df['timestamp'].values.astype(np.float64)/1e6; trel=ts-ts[0]; P=df[['x','y','z']].values
Rall=Rotation.from_quat(df[['qx','qy','qz','qw']].values).as_matrix()
def pose_at(t): return (np.array([np.interp(t,trel,P[:,k]) for k in range(3)]), Rall[int(np.argmin(np.abs(trel-t)))])
irow=intr.loc[(cid,CAM)]; erow=extr.loc[(cid,CAM)]
W0,H0=int(irow['width']),int(irow['height']); cx,cy=float(irow['cx']),float(irow['cy'])
fwp=np.array([float(irow[f'fw_poly_{k}']) for k in range(5)])
t_vc=np.array([float(erow[c]) for c in 'xyz']); R_vc=Rotation.from_quat([float(erow[c]) for c in ['qx','qy','qz','qw']]).as_matrix()
def project(pw,p_t,R_t):
    pc=((pw-p_t)@R_t-t_vc)@R_vc; X,Y,Z=pc[:,0],pc[:,1],pc[:,2]; r3=np.hypot(X,Y); th=np.arctan2(r3,Z)
    r=fwp[0]+fwp[1]*th+fwp[2]*th**2+fwp[3]*th**3+fwp[4]*th**4
    u=np.full(len(pw),np.nan); v=np.full(len(pw),np.nan); m=(Z>0.1)&(r3>1e-6)
    u[m]=(cx+r[m]*X[m]/r3[m])*OUT_W/W0; v[m]=(cy+r[m]*Y[m]/r3[m])*OUT_H/H0
    return u,v
# 3 条预测世界锚定
preds=[]
for tag in tags:
    pd_,cd_=pred_data[tag],clip_data[tag]
    pred=np.array(pd_['pred_xy']); gt=np.array(cd_['gt_xyz'])
    pred3=np.hstack([pred,np.interp(pred[:,0],gt[:,0],gt[:,2])[:,None]])
    p0,R0=pose_at(cd_['t0']); preds.append({'t0':cd_['t0'],'pw':pred3@R0.T+p0})
print(f"{'t':>5} {'锚t₀':>5} {'年龄':>5} {'蓝尖时刻':>8} {'绿尖时刻':>8} {'时间落后':>8} {'v蓝尖':>6} {'v绿尖':>6} {'像素落后':>8}")
for t in [1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,7.5,8.0,9.0]:
    p_t,R_t=pose_at(t)
    act=[pr for pr in preds if pr['t0']<=t+1e-6]; cur=act[-1]
    t0=cur['t0']; age=t-t0
    blue_tip_t=t0+6.4; green_tip_t=t+6.3; time_lag=green_tip_t-blue_tip_t
    ub,vb=project(cur['pw'],p_t,R_t)                      # 蓝：整条预测
    gt_w=np.column_stack([np.interp(t+np.arange(64)*0.1,trel,P[:,k]) for k in range(3)])
    ug,vg=project(gt_w,p_t,R_t)                           # 绿：GT 未来
    vb_tip=vb[~np.isnan(vb)][-1] if (~np.isnan(vb)).any() else np.nan   # 最后可见点=尖端
    vg_tip=vg[~np.isnan(vg)][-1] if (~np.isnan(vg)).any() else np.nan
    px_lag=vb_tip-vg_tip
    mark=' ←刷新' if age<0.05 else ''
    print(f"{t:5.1f} {t0:5.1f} {age:5.1f} {blue_tip_t:8.1f} {green_tip_t:8.1f} {time_lag:8.1f} {vb_tip:6.0f} {vg_tip:6.0f} {px_lag:8.0f}{mark}")
print("\n判据：年龄=0(刷新)时像素落后≈0；年龄↑则落后↑ → 设计(世界锚定被吃掉)，非恒定偏移bug")
