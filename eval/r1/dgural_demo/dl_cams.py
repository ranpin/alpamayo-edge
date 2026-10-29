import subprocess, os, glob
full={os.path.basename(f).split('.')[0][:8]:os.path.basename(f).split('.')[0] for f in glob.glob('ego_all/*.parquet')}
CLIPS=['46003675','ecafce84','058bf960','76f4a312','fa83bcb8']
CAMS=['camera_cross_left_120fov','camera_cross_right_120fov','camera_front_tele_30fov']
short={'camera_cross_left_120fov':'cl','camera_cross_right_120fov':'cr','camera_front_tele_30fov':'ft'}
TOK=os.environ.get("HF_TOKEN","")  # 从环境变量读，勿硬编码
BASE="https://hf-mirror.com/datasets/dgural/PhysicalAI-Autonomous-Vehicles-Sample/resolve/main/data"
D='cand'; os.makedirs(D,exist_ok=True)
n=0
for cid8 in CLIPS:
    cid=full[cid8]
    for cam in CAMS:
        out=f'{D}/{cid8}_{short[cam]}.mp4'
        if os.path.exists(out) and os.path.getsize(out)>1e6:
            print(f'skip {cid8}_{short[cam]}',flush=True); continue
        ok=False
        for a in range(4):
            subprocess.run(['curl','-sL','--max-time','180','-H',f'Authorization: Bearer {TOK}',
                            f'{BASE}/{cid}.{cam}.mp4','-o',out],capture_output=True)
            if os.path.exists(out) and os.path.getsize(out)>1e6: ok=True; break
        sz=os.path.getsize(out)/1e6 if os.path.exists(out) else 0
        n+=1; print(f'[{n}/15] {cid8}_{short[cam]} {sz:.1f}MB {"OK" if ok else "FAIL"}',flush=True)
print('ALL_CAMS_DONE')
