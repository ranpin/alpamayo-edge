import subprocess, struct, io, zlib, os, glob, json
import pandas as pd, numpy as np

TOK=os.environ.get("HF_TOKEN","")  # 从环境变量读，勿硬编码
URL="https://hf-mirror.com/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip"
SIZE=2050553236
full={os.path.basename(f).split('.')[0][:8]:os.path.basename(f).split('.')[0] for f in glob.glob('ego_all/*.parquet')}
CAND_T0={'46003675':7.0,'ecafce84':9.0,'058bf960':7.0,'76f4a312':8.0,'fa83bcb8':4.0}  # 1f7ef408 已有
NEW=list(CAND_T0.keys())

_n=[0]
def fetch(start,end):
    _n[0]+=1; out=f"/tmp/zrm_{_n[0]}.bin"
    for a in range(3):
        r=subprocess.run(["curl","-sL","--max-time","90","-r",f"{start}-{end}","-H",f"Authorization: Bearer {TOK}",URL,"-o",out,"-w","%{http_code}"],capture_output=True,text=True)
        if os.path.exists(out) and os.path.getsize(out)>0 and r.stdout.strip()in('206','200'): 
            d=open(out,"rb").read(); os.remove(out); return d
    d=open(out,"rb").read() if os.path.exists(out) else b''; os.path.exists(out) and os.remove(out); return d

# 1) tail->EOCD  2) 中央目录
tail=fetch(SIZE-256*1024,SIZE-1); i=tail.rfind(b'\x50\x4b\x05\x06')
cd_size=struct.unpack('<I',tail[i+12:i+16])[0]; cd_off=struct.unpack('<I',tail[i+16:i+20])[0]
cd=fetch(cd_off,cd_off+cd_size-1)
pos=0; entries={}
while pos+46<=len(cd) and cd[pos:pos+4]==b'\x50\x4b\x01\x02':
    method=struct.unpack('<H',cd[pos+10:pos+12])[0]; csize=struct.unpack('<I',cd[pos+20:pos+24])[0]
    fnl=struct.unpack('<H',cd[pos+28:pos+30])[0]; exl=struct.unpack('<H',cd[pos+30:pos+32])[0]; cml=struct.unpack('<H',cd[pos+32:pos+34])[0]
    loff=struct.unpack('<I',cd[pos+42:pos+46])[0]; name=cd[pos+46:pos+46+fnl].decode('utf-8','replace')
    entries[name]=(method,csize,loff); pos+=46+fnl+exl+cml
print('CD entries:',len(entries))
os.makedirs('ts',exist_ok=True)
def get_ts(cid8):
    cid=full[cid8]; tgt=f'{cid}.camera_front_wide_120fov.timestamps.parquet'
    if tgt not in entries: print(cid8,'NOT FOUND'); return None
    method,csize,loff=entries[tgt]; lh=fetch(loff,loff+30+300)
    lfnl=struct.unpack('<H',lh[26:28])[0]; lexl=struct.unpack('<H',lh[28:30])[0]
    data=fetch(loff+30+lfnl+lexl, loff+30+lfnl+lexl+csize-1)
    raw=zlib.decompress(data,-15) if method==8 else data
    return pd.read_parquet(io.BytesIO(raw))['timestamp'].values.astype(np.int64)

# 取 5 新 clip 时间戳
for cid8 in NEW:
    vt=get_ts(cid8)
    if vt is not None: np.save(f'ts/{cid8}_fw_ts.npy',vt); print(f'{cid8}: {len(vt)} 帧 ts[0]={vt[0]}')

# 算每个 clip 的精确帧映射
def mapping(cid8, t0cand, vt):
    cid=full[cid8]
    df=pd.read_parquet(f'ego_all/{cid}.egomotion.parquet').sort_values('timestamp')
    et=df['timestamp'].values.astype(np.int64)
    vt_rel=(vt-et[0])/1e6
    i0=int(np.argmin(np.abs(vt_rel-t0cand))); t0=vt_rel[i0]
    idxs=[int(np.argmin(np.abs(vt_rel-(t0-dt)))) for dt in (0.3,0.2,0.1,0.0)]
    return {'cid8':cid8,'cid':cid,'t0':round(float(t0),4),'frame_t0':i0,
            'frame_idxs':idxs,'frame_times':[round(float(vt_rel[j]),4) for j in idxs],
            'sync_off':round(float(vt_rel[0]),4)}

allmap={}
allmap['1f7ef408']=mapping('1f7ef408',12.5,np.load('video_ts_375.npy').astype(np.int64))
for cid8 in NEW:
    f=f'ts/{cid8}_fw_ts.npy'
    if os.path.exists(f): allmap[cid8]=mapping(cid8,CAND_T0[cid8],np.load(f))
json.dump(allmap,open('clip_frame_map.json','w'),indent=1)
print('\n=== 6 clip 精确帧映射 ===')
for k,m in allmap.items():
    print(f"  {k}: t0={m['t0']}s frame_t0={m['frame_t0']} 4帧={m['frame_idxs']} sync_off={m['sync_off']:+.4f}s")
print('MAP_DONE')
