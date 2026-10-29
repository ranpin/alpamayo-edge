import subprocess, os, glob, json, shutil
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

full={os.path.basename(f).split('.')[0][:8]:os.path.basename(f).split('.')[0] for f in glob.glob('ego_all/*.parquet')}
cmap=json.load(open('clip_frame_map.json')); CLIPS=list(cmap.keys())
CAMSHORT={'fw':'camera_front_wide_120fov','cl':'camera_cross_left_120fov','cr':'camera_cross_right_120fov','ft':'camera_front_tele_30fov'}
CAM_ORDER=['cl','fw','cr','ft']   # R1 camera-major: cross_left, front_wide, cross_right, front_tele
GOATBASE='/home/nvidia/chenrunbin/alpamayo-edge/eval_data/dgural_multi'

# 归并 1f7ef408 视频到 cand/ 统一命名
if not os.path.exists('cand/1f7ef408_fw.mp4'):
    shutil.copy('dg2.mp4','cand/1f7ef408_fw.mp4')
    for cs,cam in [('cl','camera_cross_left_120fov'),('cr','camera_cross_right_120fov'),('ft','camera_front_tele_30fov')]:
        shutil.copy(f'cams/{cam}.mp4',f'cand/1f7ef408_{cs}.mp4')

shutil.rmtree('frames_multi',ignore_errors=True)
def ipf(cid):
    df=pd.read_parquet(f'ego_all/{cid}.egomotion.parquet').sort_values('timestamp').reset_index(drop=True)
    ts=df['timestamp'].values.astype(np.float64)/1e6; trel=ts-ts[0]
    P=df[['x','y','z']].values; Rall=Rotation.from_quat(df[['qx','qy','qz','qw']].values)
    def ip(t):
        p=np.array([np.interp(t,trel,P[:,k]) for k in range(3)])
        return p, Rall[int(np.argmin(np.abs(trel-t)))].as_matrix()
    return ip

data={}
for cid8 in CLIPS:
    m=cmap[cid8]; cid=m['cid']; t0=m['t0']; idxs=m['frame_idxs']
    os.makedirs(f'frames_multi/{cid8}',exist_ok=True)
    for cs in CAMSHORT:
        sel='+'.join(f'eq(n\\,{i})' for i in idxs)
        subprocess.run(['ffmpeg','-v','error','-y','-i',f'cand/{cid8}_{cs}.mp4','-vf',f'select={sel}','-vsync','0',
                        f'frames_multi/{cid8}/{cs}_%d.png'],check=True)
    ip=ipf(cid); p0,R0=ip(t0)
    th=t0-1.5+np.arange(16)*0.1
    hist=np.array([ip(t)[0] for t in th]); he=(R0.T@(hist-p0).T).T
    hr=np.array([R0.T@ip(t)[1] for t in th])
    gtf=np.array([ip(t)[0] for t in t0+np.arange(64)*0.1]); ge=(R0.T@(gtf-p0).T).T
    data[cid8]={'cid':cid,'t0':t0,'frame_idxs':idxs,'ego_history_xyz':he.tolist(),
                'ego_history_rot':hr.tolist(),'gt_xyz':ge.tolist()}
    nfr=len(glob.glob(f'frames_multi/{cid8}/*.png'))
    print(f'{cid8}: t0={t0} 帧{nfr}/16 he[-1]={np.round(he[-1],2)} GT末=fwd{ge[-1,0]:.1f}/lat{ge[-1,1]:.1f} z末={ge[-1,2]:.2f}')
json.dump(data,open('multi_clip_data.json','w'))

requests=[]
for cid8 in CLIPS:
    d=data[cid8]
    paths=[f'{GOATBASE}/{cid8}/{cs}_{j}.png' for cs in CAM_ORDER for j in range(1,5)]
    requests.append({'id':cid8,'messages':[
        {'role':'system','content':[{'type':'text','text':'You are a driving assistant that generates safe and accurate actions.'}]},
        {'role':'user','content':[{'type':'image','image':p} for p in paths]
            +[{'type':'trajectory','trajectory':d['ego_history_xyz']}]
            +[{'type':'text','text':'output the chain-of-thought reasoning of the driving process, then output the future trajectory.'}]}]})
json.dump({'batch_size':1,'temperature':0.0,'top_p':1.0,'top_k':50,'max_generate_length':128,'requests':requests},
          open('r1_multi_in.json','w'))
print(f'\nbuilt r1_multi_in.json: {len(requests)} requests; 总帧数={len(glob.glob("frames_multi/*/*.png"))}')
