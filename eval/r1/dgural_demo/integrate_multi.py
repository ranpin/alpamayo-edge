import sys, json, numpy as np, torch
sys.path.insert(0,'/home/nvidia/chenrunbin/alpamayo-edge/trtedge-0.10.1/examples/accuracy/scripts')
import compute_minade as CM

GD='/home/nvidia/chenrunbin/alpamayo-edge/eval_data/dgural_multi'
out=json.load(open(f'{GD}/r1_multi_out.json'))
mcd=json.load(open(f'{GD}/multi_clip_data.json'))
inp=json.load(open(f'{GD}/r1_multi_in.json'))
req_ids=[r['id'] for r in inp['requests']]
resps=out['responses']
print('TOP keys:',list(out.keys()),'| n_resp:',len(resps),'| resp[0] keys:',list(resps[0].keys()))
has_id='id' in resps[0]
print('responses 带 id:',has_id)

def integrate(ot,hxyz,hrot):
    sa=torch.from_numpy(np.array(ot,dtype=np.float32)).reshape(1,64,2)
    v0=CM.estimate_v0(torch.tensor(hxyz,dtype=torch.float32).unsqueeze(0),
                      torch.tensor(hrot,dtype=torch.float32).unsqueeze(0))
    accel=sa[...,0]*CM.ACCEL_STD+CM.ACCEL_MEAN; kappa=sa[...,1]*CM.CURVATURE_STD+CM.CURVATURE_MEAN
    dt=CM.DT; dt2=0.5*dt**2; hdt=0.5*dt
    vel=torch.cat([v0.unsqueeze(-1),v0.unsqueeze(-1)+torch.cumsum(accel*dt,-1)],-1)
    iy=torch.zeros_like(v0)
    th=torch.cat([iy.unsqueeze(-1),iy.unsqueeze(-1)+torch.cumsum(kappa*vel[...,:-1]*dt,-1)+torch.cumsum(kappa*accel*dt2,-1)],-1)
    x=torch.cumsum(vel[...,:-1]*torch.cos(th[...,:-1])*hdt,-1)+torch.cumsum(vel[...,1:]*torch.cos(th[...,1:])*hdt,-1)
    y=torch.cumsum(vel[...,:-1]*torch.sin(th[...,:-1])*hdt,-1)+torch.cumsum(vel[...,1:]*torch.sin(th[...,1:])*hdt,-1)
    return torch.stack([x,y],-1).numpy()[0], float(v0[0])

result={}
for i in range(len(resps)):
    resp=resps[i]; rid=resp['id'] if has_id else req_ids[i]
    d=mcd[rid]; pred,v0=integrate(resp['output_trajectory'],d['ego_history_xyz'],d['ego_history_rot'])
    gt=np.array(d['gt_xyz'])[:,:2]; dd=np.linalg.norm(pred-gt,axis=1)
    result[rid]={'pred_xy':pred.tolist(),'v0':v0,'cot':resp.get('output_text',''),
                 'ade3':float(dd[:30].mean()),'ade6':float(dd[:64].mean()),'fde':float(dd[-1])}
    print(f'{rid}: v0={v0:5.2f} ADE3={dd[:30].mean():5.2f} ADE6={dd[:64].mean():5.2f} FDE={dd[-1]:5.2f} | pred末={np.round(pred[-1],1)} gt末={np.round(gt[-1],1)}')
json.dump(result,open(f'{GD}/r1_multi_pred.json','w'))
ades=[r['ade6'] for r in result.values()]
print(f'\n6clip 平均 ADE@6.4s={np.mean(ades):.2f}m  平均FDE={np.mean([r["fde"] for r in result.values()]):.2f}m')
print('SAVED r1_multi_pred.json')
