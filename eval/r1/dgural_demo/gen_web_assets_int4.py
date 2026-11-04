import json, numpy as np, pandas as pd, os
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

# 三线叠加 v2：GT 真值(绿·实线·最粗) + R1 FP16(蓝·实线·中) + R1 INT4(橙·虚线·带深描边)。
# 用户反馈「三线叠一块看不清」=> 用样式(实/虚)+粗细+颜色三重区分；INT4 虚线压 FP16 实线上、
# 空隙露蓝，即使近重叠也能一眼分辨。投影链与 gen_web_assets.py 逐字相同(已验证 diff=0.00)。
mcd=json.load(open('multi_clip_data.json'))
pred_fp16=json.load(open('r1_multi_pred.json'))
pred_int4=json.load(open('r1_multi_pred_int4.json'))
intr=pd.read_parquet('calib/camera_intrinsics.chunk_0000.parquet')
extr=pd.read_parquet('calib/sensor_extrinsics.chunk_0000.parquet')
CAM='camera_front_wide_120fov'
try: F=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",24)
except: F=ImageFont.load_default()
os.makedirs('web_assets',exist_ok=True)
GT_COL=(60,255,120); FP_COL=(90,170,255); I4_COL=(255,150,40); DARK=(18,20,26,150)

def draw_solid(D,uv,inf,col,wcore,whalo):
    for i in range(len(uv)-1):
        if inf[i] and inf[i+1]:
            if whalo: D.line([tuple(uv[i]),tuple(uv[i+1])],fill=col+(70,),width=whalo)
            D.line([tuple(uv[i]),tuple(uv[i+1])],fill=col+(245,),width=wcore)

def draw_dashed(D,uv,inf,col,wcore,dash,gap,ocol=None,ow=None):
    N=len(uv); i=0
    while i<N:
        if not inf[i]: i+=1; continue
        j=i
        while j+1<N and inf[j+1]: j+=1
        run=uv[i:j+1]
        if len(run)>=2:
            res=[tuple(run[0])]
            for k in range(1,len(run)):
                p0=run[k-1]; p1=run[k]; d=float(np.hypot(p1[0]-p0[0],p1[1]-p0[1])); n=int(max(1,round(d)))
                for m in range(1,n+1): res.append((p0[0]+(p1[0]-p0[0])*m/n, p0[1]+(p1[1]-p0[1])*m/n))
            L=len(res); k=0
            while k<L-1:
                e=min(k+dash,L-1)
                if e>k:
                    if ocol: D.line([res[k],res[e]],fill=ocol,width=ow)
                    D.line([res[k],res[e]],fill=col+(255,),width=wcore)
                k=e+gap
        i=j+1

for cid8 in mcd:
    cid=mcd[cid8]['cid']
    irow=intr.loc[(cid,CAM)]; erow=extr.loc[(cid,CAM)]
    W0,H0=int(irow['width']),int(irow['height']); cx,cy=float(irow['cx']),float(irow['cy'])
    fw=np.array([float(irow[f'fw_poly_{k}']) for k in range(5)])
    t_vc=np.array([float(erow[c]) for c in 'xyz']); R_vc=Rotation.from_quat([float(erow[c]) for c in ['qx','qy','qz','qw']]).as_matrix()
    def project(p3):
        pc=(p3-t_vc)@R_vc; X,Y,Z=pc[:,0],pc[:,1],pc[:,2]; fr=Z>0.1; r3=np.hypot(X,Y)
        th=np.arctan2(r3,Z); r=fw[0]+fw[1]*th+fw[2]*th**2+fw[3]*th**3+fw[4]*th**4
        u=np.full(len(p3),np.nan); v=np.full(len(p3),np.nan); msk=fr&(r3>1e-6)
        u[msk]=cx+r[msk]*X[msk]/r3[msk]; v[msk]=cy+r[msk]*Y[msk]/r3[msk]; return u,v
    ge=np.array(mcd[cid8]['gt_xyz'])
    def to3d(pxy):
        pr=np.array(pxy); pz=np.interp(pr[:,0],ge[:,0],ge[:,2]); return np.hstack([pr,pz[:,None]])
    fp3=to3d(pred_fp16[cid8]['pred_xy']); i43=to3d(pred_int4[cid8]['pred_xy'])
    img=Image.open(f'frames/{cid8}.png').convert('RGB'); W,H=img.size; sx,sy=W/W0,H/H0
    D=ImageDraw.Draw(img,'RGBA')
    def prep(p3):
        u,v=project(p3); u,v=u*sx,v*sy
        inf=(~np.isnan(u))&(u>=-50)&(u<W+50)&(v>=-50)&(v<H+50)
        return np.stack([u,v],1), inf
    uvg,ig=prep(ge); uvf,iff=prep(fp3); uvi,ii=prep(i43)
    # 底→顶：GT 绿实线(最粗) → FP16 蓝实线(中) → INT4 橙虚线(深描边)
    draw_solid(D,uvg,ig,GT_COL,12,20)
    draw_solid(D,uvf,iff,FP_COL,8,0)
    draw_dashed(D,uvi,ii,I4_COL,6,26,17,ocol=DARK,ow=11)
    # 端点圈（6.4s 终点）
    for uv,inf,col in [(uvg,ig,GT_COL),(uvf,iff,FP_COL),(uvi,ii,I4_COL)]:
        vis=[k for k in range(64) if inf[k]]
        if vis:
            e=uv[vis[-1]]; D.ellipse([e[0]-10,e[1]-10,e[0]+10,e[1]+10],outline=col+(255,),width=4)
    # 图例（3 条，样式与图上一致：绿实/蓝实/橙虚）
    D.rectangle([20,20,352,140],fill=(15,18,26,195))
    D.line([(36,46),(84,46)],fill=GT_COL+(255,),width=9); D.text((94,33),"GT 真值",font=F,fill=(255,255,255))
    D.line([(36,80),(84,80)],fill=FP_COL+(255,),width=7); D.text((94,67),"R1 FP16",font=F,fill=(255,255,255))
    for xs in range(36,85,16): D.line([(xs,114),(min(xs+9,84),114)],fill=I4_COL+(255,),width=6)
    D.text((94,101),"R1 INT4",font=F,fill=(255,255,255))
    img.resize((1280,720),Image.LANCZOS).convert('RGB').save(f'web_assets/{cid8}.jpg',quality=90)
    print(f'{cid8}.jpg {os.path.getsize(f"web_assets/{cid8}.jpg")//1024}KB  ADE6 fp16={pred_fp16[cid8]["ade6"]:.2f} int4={pred_int4[cid8]["ade6"]:.2f}')
print('WEB_ASSETS_INT4_V2_DONE')
