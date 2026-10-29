import subprocess, os, glob, json
import numpy as np, pandas as pd
from PIL import Image

# 候选：cid8, t0(egomotion相对), dhdg, v, fwd  —— 4左4右、速度多样
CAND=[
 ('69cf2443',10.5,-92,7.4,29),('46003675',7.0,91,9.1,48),
 ('ecafce84',9.0,90,7.4,26),('37c25a1d',8.5,-85,9.6,36),
 ('fa83bcb8',4.0,-71,13.2,64),('054da32b',3.5,67,8.9,29),
 ('058bf960',7.0,80,5.7,27),('76f4a312',8.0,-88,8.2,35),
]
full={os.path.basename(f).split('.')[0][:8]:os.path.basename(f).split('.')[0] for f in glob.glob('ego_all/*.parquet')}
TOK=os.environ.get("HF_TOKEN","")  # 从环境变量读，勿硬编码
BASE="https://hf-mirror.com/datasets/dgural/PhysicalAI-Autonomous-Vehicles-Sample/resolve/main/data"
D='cand'; os.makedirs(D,exist_ok=True)
rows=[]
for cid8,t0,dhdg,v,fwd in CAND:
    cid=full[cid8]; mp4=f'{D}/{cid8}_fw.mp4'
    if not (os.path.exists(mp4) and os.path.getsize(mp4)>1e6):
        for a in range(3):
            r=subprocess.run(['curl','-sL','--max-time','120','-H',f'Authorization: Bearer {TOK}',
                              f'{BASE}/{cid}.camera_front_wide_120fov.mp4','-o',mp4],capture_output=True)
            if os.path.exists(mp4) and os.path.getsize(mp4)>1e6: break
    sz=os.path.getsize(mp4)/1e6 if os.path.exists(mp4) else 0
    fr=int(round(t0*30)); png=f'{D}/{cid8}_f{fr}.png'
    subprocess.run(['ffmpeg','-v','error','-y','-i',mp4,'-vf',f'select=eq(n\\,{fr})','-vsync','0','-frames:v','1',png],capture_output=True)
    if os.path.exists(png):
        g=np.array(Image.open(png).convert('L')).astype(float)
        mean,std,over=g.mean(),g.std(),(g>250).mean()*100
    else: mean=std=over=-1
    rows.append((cid8,cid,t0,fr,dhdg,v,fwd,round(sz,1),round(mean,1),round(std,1),round(over,2)))
    print(f'{cid8} t0={t0} frame{fr} dhdg={dhdg:+d} v={v} | {sz:.1f}MB mean={mean:.1f} std={std:.1f} over={over:.2f}%',flush=True)
json.dump([{'cid8':r[0],'cid':r[1],'t0':r[2],'frame':r[3],'dhdg':r[4],'v':r[5],'fwd':r[6],'mean':r[8]} for r in rows],
          open(f'{D}/cand_info.json','w'),indent=1)
print('\n=== 按亮度排序（mean 越高越好，避开<75 的雾/暗帧）===')
for r in sorted(rows,key=lambda x:-x[8]):
    flag='✓亮' if r[8]>=80 else ('~偏暗' if r[8]>=75 else '✗太暗')
    print(f'  {r[0]} mean={r[8]:.1f} {flag}  dhdg={r[4]:+d} v={r[5]} fwd={r[6]}')
