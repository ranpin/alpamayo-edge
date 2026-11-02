#!/usr/bin/env python3
"""滚动渲染量化自检（模型无视觉的替代）：
1) 锚点帧恒等：project_world(pred_world, pose(t0)) == 静态 (pred3−t_vc)@R_vc 鱼眼（逐位同源已验证 demo）。
2) 路面落点：各 clip 早/中/晚帧，打印 GT 与活跃 pred 在 2s/5s 前瞻点的投影像素 (u,v)，
   判据：v 落在画面下半（路面区）、远点 v 更小（向上收敛）、u 近中线或按转向方向偏移。
"""
import json, glob, os
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

ROLL='/tmp/r1vid/roll'; EGO='/tmp/r1vid/ego'; CALIB='/tmp/r1vid/calib'
CAM='camera_front_wide_120fov'; OUT_W,OUT_H=1280,720
pred_data=json.load(open(f'{ROLL}/roll_pred.json'))
clip_data=json.load(open(f'{ROLL}/roll_clip_data.json'))
intr=pd.read_parquet(f'{CALIB}/camera_intrinsics.chunk_0000.parquet')
extr=pd.read_parquet(f'{CALIB}/sensor_extrinsics.chunk_0000.parquet')
CLIPS=['0dd3eb7f','ecafce84','1fcf2fef','1f7ef408']

for cid8 in CLIPS:
    tags=sorted([k for k in clip_data if k.startswith(cid8+'_')],key=lambda k:clip_data[k]['t0'])
    cid=clip_data[tags[0]]['cid']
    df=pd.read_parquet(f'{EGO}/{cid}.egomotion.parquet').sort_values('timestamp').reset_index(drop=True)
    ts=df['timestamp'].values.astype(np.float64)/1e6; trel=ts-ts[0]
    P=df[['x','y','z']].values
    Rall=Rotation.from_quat(df[['qx','qy','qz','qw']].values).as_matrix()
    def pose_at(t):
        return (np.array([np.interp(t,trel,P[:,k]) for k in range(3)]),
                Rall[int(np.argmin(np.abs(trel-t)))])
    irow=intr.loc[(cid,CAM)]; erow=extr.loc[(cid,CAM)]
    W0,H0=int(irow['width']),int(irow['height']); cx,cy=float(irow['cx']),float(irow['cy'])
    fwp=np.array([float(irow[f'fw_poly_{k}']) for k in range(5)])
    t_vc=np.array([float(erow[c]) for c in 'xyz'])
    R_vc=Rotation.from_quat([float(erow[c]) for c in ['qx','qy','qz','qw']]).as_matrix()
    def fish(pc,W=OUT_W,H=OUT_H):
        X,Y,Z=pc[:,0],pc[:,1],pc[:,2]; r3=np.hypot(X,Y); th=np.arctan2(r3,Z)
        r=fwp[0]+fwp[1]*th+fwp[2]*th**2+fwp[3]*th**3+fwp[4]*th**4
        sx,sy=W/W0,H/H0; u=np.full(len(pc),np.nan); v=np.full(len(pc),np.nan)
        m=(Z>0.1)&(r3>1e-6); u[m]=(cx+r[m]*X[m]/r3[m])*sx; v[m]=(cy+r[m]*Y[m]/r3[m])*sy
        return u,v
    def proj_world(pw,p_t,R_t): return fish(((pw-p_t)@R_t-t_vc)@R_vc)
    def proj_static(p3): return fish((p3-t_vc)@R_vc)

    print(f'\n===== {cid8} ({cid[:8]}) =====')
    # 1) 锚点恒等
    for tag in tags:
        pd_,cd_=pred_data[tag],clip_data[tag]
        pred=np.array(pd_['pred_xy']); gt=np.array(cd_['gt_xyz'])
        pred3=np.hstack([pred,np.interp(pred[:,0],gt[:,0],gt[:,2])[:,None]])
        p0,R0=pose_at(cd_['t0']); pw=pred3@R0.T+p0
        ur,vr=proj_world(pw,p0,R0); us,vs=proj_static(pred3)
        d=np.nanmax(np.abs(np.vstack([ur-us,vr-vs])))
        print(f'  [恒等] {tag}: project_world@anchor vs static max|Δ|={d:.2e}px  (应≈0)')
    # 2) 路面落点：取 t=3/10/17s
    vt=np.load(f'{ROLL}/ts/{cid8}_fw_ts.npy').astype(np.int64)
    et=df['timestamp'].values.astype(np.int64); vt_rel=(vt-et[0])/1e6
    for tq in (3.0,10.0,17.0):
        i=int(np.argmin(np.abs(vt_rel-tq))); t=vt_rel[i]
        p_t,R_t=pose_at(t)
        # GT 未来 2s/5s
        gt_w=np.column_stack([np.interp(t+np.array([2.0,5.0]),trel,P[:,k]) for k in range(3)])
        ug,vg=proj_world(gt_w,p_t,R_t)
        # 活跃 pred
        act=[tg for tg in tags if clip_data[tg]['t0']<=t+1e-6]
        if act:
            tg=act[-1]; pd_,cd_=pred_data[tg],clip_data[tg]
            pred=np.array(pd_['pred_xy']); gt=np.array(cd_['gt_xyz'])
            pred3=np.hstack([pred,np.interp(pred[:,0],gt[:,0],gt[:,2])[:,None]])
            p0,R0=pose_at(cd_['t0']); pw=pred3@R0.T+p0
            # pred 上离 t 最近 2s/5s 前瞻的点（按覆盖时间索引）
            base=cd_['t0']; i2=int(round((t+2.0-base)/0.1)); i5=int(round((t+5.0-base)/0.1))
            sel=[j for j in (i2,i5) if 0<=j<64]
            ub,vb=proj_world(pw[sel],p_t,R_t) if sel else (np.array([np.nan]),np.array([np.nan]))
            anchor=cd_['t0']
        else:
            ub,vb=np.array([np.nan,np.nan]),np.array([np.nan,np.nan]); anchor=None
        print(f'  t={t:4.1f}s 锚={anchor}: GT2s=({ug[0]:.0f},{vg[0]:.0f}) GT5s=({ug[1]:.0f},{vg[1]:.0f}) '
              f'| R1: '+' '.join(f'({x:.0f},{y:.0f})' for x,y in zip(ub,vb)))
print('\n判据: v∈(360,720)=路面下半区; 同列远点(5s)v<近点(2s)v=向上收敛; 直行 u≈640±120, 左转 u 偏小, 右转 u 偏大')
