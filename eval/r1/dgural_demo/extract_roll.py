#!/usr/bin/env python3
"""滚动 demo 多 t0 抽取：每 clip 取 t0≈1.5/7.5/13.5s（间隔6s=R1真实耗时），
抽 4 相机×4 帧 + 算 ego 历史(16点@10Hz) + GT(64点)，生成 R1 批量输入 roll_in.json。
帧索引用 gated timestamps 精确定位（朴素 t×30 会错 1-2m）。抽出的帧随后 scp 到 goat。
"""
import subprocess, os, glob, json
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

ROLL='/tmp/r1vid/roll'; EGO='/tmp/r1vid/ego'
CLIPS=['ecafce84','1fcf2fef','0dd3eb7f','1f7ef408']
CAMSHORT={'fw':'camera_front_wide_120fov','cl':'camera_cross_left_120fov','cr':'camera_cross_right_120fov','ft':'camera_front_tele_30fov'}
CAM_ORDER=['cl','fw','cr','ft']   # R1 camera-major
T0S=[1.5,7.5,13.5]                 # 间隔6s=R1真实单次耗时
full={os.path.basename(f).split('.')[0][:8]:os.path.basename(f).split('.')[0] for f in glob.glob(f'{EGO}/*.parquet')}

def ipf(cid):
    df=pd.read_parquet(f'{EGO}/{cid}.egomotion.parquet').sort_values('timestamp').reset_index(drop=True)
    ts=df['timestamp'].values.astype(np.float64)/1e6; trel=ts-ts[0]
    P=df[['x','y','z']].values; Rall=Rotation.from_quat(df[['qx','qy','qz','qw']].values)
    def ip(t):
        p=np.array([np.interp(t,trel,P[:,k]) for k in range(3)])
        return p, Rall[int(np.argmin(np.abs(trel-t)))].as_matrix()
    return ip, trel

data={}; os.makedirs(f'{ROLL}/frames_roll',exist_ok=True)
for cid8 in CLIPS:
    cid=full[cid8]
    vt=np.load(f'{ROLL}/ts/{cid8}_fw_ts.npy').astype(np.int64)
    ip,trel=ipf(cid)
    df=pd.read_parquet(f'{EGO}/{cid}.egomotion.parquet').sort_values('timestamp')
    et=df['timestamp'].values.astype(np.int64); vt_rel=(vt-et[0])/1e6  # 帧时间相对 egomotion 起点
    for t0 in T0S:
        i0=int(np.argmin(np.abs(vt_rel-t0))); t0a=float(vt_rel[i0])
        idxs=[int(np.argmin(np.abs(vt_rel-(t0a-dt)))) for dt in (0.3,0.2,0.1,0.0)]
        tag=f'{cid8}_t{t0a:.1f}'
        os.makedirs(f'{ROLL}/frames_roll/{tag}',exist_ok=True)
        for cs,cam in CAMSHORT.items():
            sel='+'.join(f'eq(n\\,{i})' for i in idxs)
            subprocess.run(['ffmpeg','-v','error','-y','-i',f'{ROLL}/{cid8}_{cs}.mp4','-vf',f'select={sel}','-vsync','0',
                            f'{ROLL}/frames_roll/{tag}/{cs}_%d.png'],check=True)
        p0,R0=ip(t0a)
        th=t0a-1.5+np.arange(16)*0.1
        hist=np.array([ip(t)[0] for t in th]); he=(R0.T@(hist-p0).T).T
        gtf=np.array([ip(t)[0] for t in t0a+np.arange(64)*0.1]); ge=(R0.T@(gtf-p0).T).T
        data[tag]={'cid8':cid8,'cid':cid,'t0':round(t0a,4),'frame_idxs':idxs,
                   'ego_history_xyz':he.tolist(),'gt_xyz':ge.tolist()}
        nfr=len(glob.glob(f'{ROLL}/frames_roll/{tag}/*.png'))
        print(f'{tag}: 帧{nfr}/16 he[-1]={np.round(he[-1],2)} GT末=fwd{ge[-1,0]:.1f}/lat{ge[-1,1]:.1f}')
json.dump(data,open(f'{ROLL}/roll_clip_data.json','w'))

# R1 批量输入（帧路径先写 goat 目标路径，帧随后 scp 过去）
GOATBASE='/home/nvidia/chenrunbin/alpamayo-edge/eval_data/roll'
requests=[]
for tag in data:
    paths=[f'{GOATBASE}/frames_roll/{tag}/{cs}_{j}.png' for cs in CAM_ORDER for j in range(1,5)]
    requests.append({'id':tag,'messages':[
        {'role':'system','content':[{'type':'text','text':'You are a driving assistant that generates safe and accurate actions.'}]},
        {'role':'user','content':[{'type':'image','image':p} for p in paths]
            +[{'type':'trajectory','trajectory':data[tag]['ego_history_xyz']}]
            +[{'type':'text','text':'output the chain-of-thought reasoning of the driving process, then output the future trajectory.'}]}]})
json.dump({'batch_size':1,'temperature':0.0,'top_p':1.0,'top_k':50,'max_generate_length':128,'requests':requests},
          open(f'{ROLL}/roll_in.json','w'))
print(f'\nbuilt roll_in.json: {len(requests)} requests (4 clip × 3 t0); 总帧={len(glob.glob(f"{ROLL}/frames_roll/*/*.png"))}')
