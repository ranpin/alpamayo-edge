import json, numpy as np, pandas as pd, subprocess, os
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

cmap=json.load(open('clip_frame_map.json'))
mcd=json.load(open('multi_clip_data.json'))
pred=json.load(open('r1_multi_pred.json'))
intr=pd.read_parquet('calib/camera_intrinsics.chunk_0000.parquet')
extr=pd.read_parquet('calib/sensor_extrinsics.chunk_0000.parquet')
CAM='camera_front_wide_120fov'
try: F=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",26)
except: F=ImageFont.load_default()
os.makedirs('web_assets',exist_ok=True)

for cid8 in cmap:
    m=cmap[cid8]; cid=m['cid']; ft0=m['frame_t0']
    raw=f'web_assets/{cid8}_raw.png'
    subprocess.run(['ffmpeg','-v','error','-y','-i',f'cand/{cid8}_fw.mp4','-vf',f'select=eq(n\\,{ft0})','-vsync','0','-frames:v','1',raw],check=True)
    irow=intr.loc[(cid,CAM)]; erow=extr.loc[(cid,CAM)]
    W0,H0=int(irow['width']),int(irow['height']); cx,cy=float(irow['cx']),float(irow['cy'])
    fw=np.array([float(irow[f'fw_poly_{k}']) for k in range(5)])
    t_vc=np.array([float(erow[c]) for c in 'xyz']); R_vc=Rotation.from_quat([float(erow[c]) for c in ['qx','qy','qz','qw']]).as_matrix()
    def project(p3):
        pc=(p3-t_vc)@R_vc; X,Y,Z=pc[:,0],pc[:,1],pc[:,2]; fr=Z>0.1; r3=np.hypot(X,Y)
        th=np.arctan2(r3,Z); r=fw[0]+fw[1]*th+fw[2]*th**2+fw[3]*th**3+fw[4]*th**4
        u=np.full(len(p3),np.nan); v=np.full(len(p3),np.nan); msk=fr&(r3>1e-6)
        u[msk]=cx+r[msk]*X[msk]/r3[msk]; v[msk]=cy+r[msk]*Y[msk]/r3[msk]; return u,v
    ge=np.array(mcd[cid8]['gt_xyz']); pr=np.array(pred[cid8]['pred_xy'])
    pr_z=np.interp(pr[:,0],ge[:,0],ge[:,2]); pr3=np.hstack([pr,pr_z[:,None]])
    img=Image.open(raw).convert('RGB'); W,H=img.size; sx,sy=W/W0,H/H0
    D=ImageDraw.Draw(img,'RGBA')
    def draw(p3,col):
        u,v=project(p3); u,v=u*sx,v*sy
        inf=(~np.isnan(u))&(u>=0)&(u<W)&(v>=0)&(v<H)
        for i in range(63):
            if inf[i] and inf[i+1]:
                D.line([(u[i],v[i]),(u[i+1],v[i+1])],fill=col+(80,),width=15)
                D.line([(u[i],v[i]),(u[i+1],v[i+1])],fill=col+(245,),width=6)
        for i in range(0,64,8):
            if inf[i]: D.ellipse([u[i]-6,v[i]-6,u[i]+6,v[i]+6],fill=(255,255,255,255),outline=col+(255,),width=2)
        vis=[i for i in range(64) if inf[i]]
        if vis: D.ellipse([u[vis[-1]]-11,v[vis[-1]]-11,u[vis[-1]]+11,v[vis[-1]]+11],outline=col+(255,),width=4)
    draw(ge,(60,255,120)); draw(pr3,(90,170,255))
    # 角标小图例（自包含）
    D.rectangle([20,20,300,96],fill=(15,18,26,185))
    D.line([(36,44),(72,44)],fill=(60,255,120,255),width=6); D.text((82,30),"GT 真值",font=F,fill=(255,255,255))
    D.line([(36,78),(72,78)],fill=(90,170,255,255),width=6); D.text((82,64),"R1 预测",font=F,fill=(255,255,255))
    img.resize((1280,720),Image.LANCZOS).convert('RGB').save(f'web_assets/{cid8}.jpg',quality=88)
    os.remove(raw)
    print(f'{cid8}.jpg  {os.path.getsize(f"web_assets/{cid8}.jpg")//1024}KB')
print('WEB_ASSETS_DONE')
