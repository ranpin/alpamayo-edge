#!/usr/bin/env python3
"""Generate a local (NON-PUBLISHED) BEV trajectory visualization page.

Reads eval/r1/viz/traj_viz.json and emits eval/r1/viz/index.html with the data
embedded inline (const DATA). Each clip renders an SVG bird's-eye-view plot of
the GT trajectory vs the 6 predicted seeds (best-seed highlighted).

⚠️ LICENSE: the GT/predicted trajectories derive from the gated
nvidia/PhysicalAI-Autonomous-Vehicles dataset (license:other, §3 Confidentiality).
This page is LOCAL ONLY — do NOT publish to the public demo site. See
docs/PROBLEMS.md §M6 / the license discussion.

Usage:  .venv/bin/python eval/r1/viz/gen_viz_page.py
Output: eval/r1/viz/index.html  (open locally in a browser)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "traj_viz.json")))

# Order clips: by class (turn, gentle, straight) then by minADE desc, so the most
# interesting (large lateral / high error) come first.
CLS_ORDER = {"turn": 0, "gentle": 1, "straight": 2}
clips = sorted(DATA.items(),
               key=lambda kv: (CLS_ORDER.get(kv[1]["cls"], 9), -kv[1]["minade"]))

data_line = "const DATA=" + json.dumps(
    [{"id": cid, **v} for cid, v in clips],
    ensure_ascii=False, separators=(",", ":")) + ";"

HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpamayo-R1 轨迹可视化（本地，勿公开）</title>
<link rel="stylesheet" href="https://ranpin.github.io/demo-shell.css">
<style>
.tv-wrap{max-width:1040px;margin:0 auto;padding:24px 16px}
.tv-note{background:#fef3c7;border-left:4px solid #f59e0b;padding:10px 14px;
  border-radius:6px;font-size:13px;margin:12px 0;color:#78350f}
.tv-stats{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.tv-stat{background:var(--ds-card,#fff);border:1px solid var(--ds-border,#e2e8f0);
  border-radius:8px;padding:12px 16px;min-width:120px;text-align:center}
.tv-stat b{display:block;font-size:22px;color:var(--ds-accent,#1a67eb)}
.tv-stat span{font-size:12px;color:var(--ds-muted,#64748b)}
.tv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.tv-card{background:var(--ds-card,#fff);border:1px solid var(--ds-border,#e2e8f0);
  border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:8px}
.tv-card h3{font-size:14px;margin:0;display:flex;justify-content:space-between;align-items:center}
.tv-tag{font-size:11px;padding:2px 8px;border-radius:10px;background:var(--ds-badge,#eff6ff);
  color:var(--ds-accent,#1a67eb)}
.tv-meta{font-size:12px;color:var(--ds-muted,#64748b);display:flex;gap:10px;flex-wrap:wrap}
.tv-meta b{color:var(--ds-fg,#0f172a)}
.tv-ade{font-weight:600}
.tv-ade.good{color:#16a34a}.tv-ade.mid{color:#d97706}.tv-ade.bad{color:#dc2626}
.tv-fig{width:100%;background:var(--ds-bg,#f8fafc);border-radius:6px}
.tv-legend{font-size:11px;color:var(--ds-muted,#64748b);display:flex;gap:12px;align-items:center}
.tv-legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:4px;vertical-align:middle}
.tv-controls{display:flex;gap:10px;align-items:center;margin:12px 0;font-size:13px}
.tv-controls label{display:flex;gap:6px;align-items:center;cursor:pointer}
</style></head><body>
<div class="tv-wrap">
<h1>Alpamayo-R1 轨迹预测可视化</h1>
<div class="tv-note">⚠️ <b>本地页面，请勿公开</b>：GT 与预测轨迹派生自 gated 数据集
nvidia/PhysicalAI-Autonomous-Vehicles（license:other，§3 将基准/性能数据列为保密信息）。
本页仅供本地查看。坐标为 ego 系（x 前 / y 左），BEV 俯视图，自车在原点朝上。</div>
<div class="tv-stats" id="stats"></div>
<div class="tv-controls">
  <label><input type="checkbox" id="showAll" checked> 显示全部 6 个 seed（否则只显示最优 seed）</label>
  <label><input type="checkbox" id="equalScale"> 统一坐标比例（否则每图自适应）</label>
</div>
<div class="tv-grid" id="grid"></div>
</div>
<script>
""" + data_line + """
const CLS_ZH={turn:"转弯",gentle:"缓弯",straight:"直行"};
const SPD_ZH={fast:"快",mid:"中",slow:"慢"};
function adeCls(a){return a<0.5?"good":(a<1.2?"mid":"bad");}
function fmt(x){return (+x).toFixed(2);}
function render(){
  const showAll=document.getElementById('showAll').checked;
  const eq=document.getElementById('equalScale').checked;
  const grid=document.getElementById('grid');grid.innerHTML='';
  // global range for equal-scale mode
  let gminx=1e9,gmaxx=-1e9,gminy=1e9,gmaxy=-1e9;
  if(eq){for(const c of DATA){for(const p of c.gt){gminx=Math.min(gminx,p[0]);gmaxx=Math.max(gmaxx,p[0]);gminy=Math.min(gminy,p[1]);gmaxy=Math.max(gmaxy,p[1]);}
    for(const pr of c.preds)for(const p of pr){gminx=Math.min(gminx,p[0]);gmaxx=Math.max(gmaxx,p[0]);gminy=Math.min(gminy,p[1]);gmaxy=Math.max(gmaxy,p[1]);}}}
  // stats
  const ades=DATA.map(c=>c.minade);
  const mean=ades.reduce((a,b)=>a+b,0)/ades.length;
  const sorted=[...ades].sort((a,b)=>a-b);
  document.getElementById('stats').innerHTML=
    `<div class="tv-stat"><b>${DATA.length}</b><span>clips</span></div>`+
    `<div class="tv-stat"><b>${mean.toFixed(3)}</b><span>minADE mean (m)</span></div>`+
    `<div class="tv-stat"><b>${sorted[Math.floor(sorted.length/2)].toFixed(3)}</b><span>median (m)</span></div>`+
    `<div class="tv-stat"><b>${sorted[0].toFixed(3)}</b><span>best (m)</span></div>`+
    `<div class="tv-stat"><b>${sorted[sorted.length-1].toFixed(3)}</b><span>worst (m)</span></div>`;
  for(const c of DATA){
    const card=document.createElement('div');card.className='tv-card';
    // compute per-clip range
    let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
    const all=eq?null:c.gt.concat(...c.preds);
    if(!eq){for(const p of all){minx=Math.min(minx,p[0]);maxx=Math.max(maxx,p[0]);miny=Math.min(miny,p[1]);maxy=Math.max(maxy,p[1]);}}
    else{minx=gminx;maxx=gmaxx;miny=gminy;maxy=gmaxy;}
    // pad
    const padX=(maxx-minx)*0.08+1,padY=(maxy-miny)*0.08+1;
    minx-=padX;maxx+=padX;miny-=padY;maxy+=padY;
    const W=300,H=260;
    const sx=p=>((p[0]-minx)/(maxx-minx))*W;
    // BEV: forward (x) up, left (y) left => svg_y = H - scaled_x, svg_x = W - scaled_y? 
    // We plot x (forward) as vertical (up), y (left) as horizontal (left).
    const sy=p=>H-((p[0]-minx)/(maxx-minx))*H;      // forward -> up
    const sx2=p=>W-((p[1]-miny)/(maxy-miny))*W;     // left -> left (so +y -> left)
    const path=pts=>pts.map((p,i)=>(i?'L':'M')+sx2(p).toFixed(1)+' '+sy(p).toFixed(1)).join(' ');
    let svg=`<svg class="tv-fig" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    // grid / axes
    svg+=`<line x1="0" y1="${sy([0,0])}" x2="${W}" y2="${sy([0,0])}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3 3"/>`;
    svg+=`<line x1="${sx2([0,0])}" y1="0" x2="${sx2([0,0])}" y2="${H}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3 3"/>`;
    // predicted seeds (thin), best highlighted
    const seeds=showAll?c.preds.map((_,i)=>i):[c.best];
    for(const i of seeds){
      const isBest=i===c.best;
      svg+=`<path d="${path(c.preds[i])}" fill="none" stroke="${isBest?'#1a67eb':'#94a3b8'}" stroke-width="${isBest?2.4:1.1}" opacity="${isBest?1:0.55}"/>`;
    }
    // GT
    svg+=`<path d="${path(c.gt)}" fill="none" stroke="#16a34a" stroke-width="2.2"/>`;
    // ego marker
    svg+=`<circle cx="${sx2([0,0])}" cy="${sy([0,0])}" r="4" fill="#0f172a"/>`;
    svg+=`</svg>`;
    card.innerHTML=`<h3><span>${c.id.slice(0,8)}</span><span class="tv-tag">${CLS_ZH[c.cls]||c.cls} · ${SPD_ZH[c.spd]||c.spd}</span></h3>`+
      `<div class="tv-meta"><span>v0 <b>${fmt(c.v0)}</b> m/s</span><span>前行 <b>${fmt(c.fwd)}</b> m</span>`+
      `<span>max横向 <b>${fmt(c.maxlat)}</b> m</span>`+
      `<span class="tv-ade ${adeCls(c.minade)}">minADE <b>${fmt(c.minade)}</b> m</span></div>`+
      svg+
      `<div class="tv-legend"><span><i style="background:#16a34a"></i>GT</span>`+
      `<span><i style="background:#1a67eb"></i>最优 seed(${c.best})</span>`+
      (showAll?`<span><i style="background:#94a3b8"></i>其余 seed</span>`:'')+
      `<span style="margin-left:auto">ADE: ${c.ades.map(a=>fmt(a)).join(' ')}</span></div>`;
    grid.appendChild(card);
  }
}
document.getElementById('showAll').addEventListener('change',render);
document.getElementById('equalScale').addEventListener('change',render);
render();
</script></body></html>"""

out = os.path.join(HERE, "index.html")
open(out, "w").write(HTML)
print("wrote", out, os.path.getsize(out), "bytes")
print("clips:", len(DATA))
