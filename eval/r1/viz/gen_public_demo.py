#!/usr/bin/env python3
"""Generate the PUBLIC (desensitized) BEV trajectory demo page.

Reads the LOCAL eval/r1/viz/traj_viz.json (gated-derived, gitignored) and emits
a DESENSITIZED public page at docs/site/r1-trajectory/index.html.

中度脱敏 (data-layer, so view-source cannot recover the removed fields):
  - clip_id (UUID)  -> anonymous index "clip-01".."clip-18"
  - REMOVE ades[], minade, best  (these are §3 benchmark/performance data)
  - REMOVE the aggregate stats cards (mean/median/best/worst minADE)
  - KEEP gt[], preds[] curves, v0, fwd, maxlat, cls, spd  (scene descriptors)
  - 6 diffusion samples shown uniformly (no "best" highlight = no perf ranking)

⚠️ RESIDUAL RISK (user-accepted 2026-10-27): GT curves are derived from the
gated nvidia/PhysicalAI-AV egomotion labels, so publishing them still touches
§4.6 ("distributing the Dataset in part"). Desensitization removes clip ids and
all benchmark numbers but does NOT eliminate this. User chose "脱敏后公开，保留GT".

Usage:  .venv/bin/python eval/r1/viz/gen_public_demo.py
Output: docs/site/r1-trajectory/index.html  (committed + deployed to gh-pages)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))          # eval/r1/viz
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # repo root
SRC = json.load(open(os.path.join(HERE, "traj_viz.json")))

CLS_ORDER = {"turn": 0, "gentle": 1, "straight": 2}
items = sorted(SRC.items(),
               key=lambda kv: (CLS_ORDER.get(kv[1]["cls"], 9), -kv[1]["maxlat"]))

def r2(pts):  # round coords to cm, shrink payload
    return [[round(p[0], 2), round(p[1], 2)] for p in pts]

# ---- DESENSITIZE: build public records with NO id/ades/minade/best ----
pub = []
for i, (cid, v) in enumerate(items, 1):
    pub.append({
        "id": f"clip-{i:02d}",          # anonymous index, NOT the UUID
        "cls": v["cls"], "spd": v["spd"],
        "v0": round(v["v0"], 2), "fwd": round(v["fwd"], 2),
        "maxlat": round(v["maxlat"], 2),
        "gt": r2(v["gt"]),
        "preds": [r2(p) for p in v["preds"]],
    })
    # sanity: ensure no sensitive field sneaks in
    assert "ades" not in pub[-1] and "minade" not in pub[-1] and "best" not in pub[-1]
    assert cid[:8] not in json.dumps(pub[-1]), f"UUID leak in {cid}"

data_line = "const DATA=" + json.dumps(pub, ensure_ascii=False,
                                      separators=(",", ":")) + ";"

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alpamayo-R1 轨迹预测 · Jetson Orin 边缘部署 Demo</title>
<meta name="description" content="Alpamayo-R1（10B 自动驾驶 VLA）部署在 Jetson AGX Orin（TensorRT-Edge-LLM 三引擎），对真实驾驶片段输出 6.4s 轨迹；扩散动作头采样 6 个种子，BEV 俯视图对比真实轨迹。真实 Orin 推理输出。">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>🛣️</text></svg>">
<link rel="stylesheet" href="https://ranpin.github.io/demo-shell.css">
<script src="https://ranpin.github.io/theme.js"></script>
<style>
.tv-controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:10px 0 18px;font-size:14px}
.tv-controls label{display:flex;gap:7px;align-items:center;cursor:pointer;color:var(--ds-text)}
.tv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px}
.tv-card{background:var(--ds-panel);border:1px solid var(--ds-border);border-radius:10px;
  padding:14px;display:flex;flex-direction:column;gap:9px}
.tv-card h3{font-size:14px;margin:0;display:flex;justify-content:space-between;align-items:center;gap:8px}
.tv-tag{font-size:11px;padding:2px 9px;border-radius:10px;background:var(--ds-accent-soft);
  color:var(--ds-accent-strong);border:1px solid var(--ds-accent-ring);white-space:nowrap}
.tv-meta{font-size:12.5px;color:var(--ds-muted);display:flex;gap:12px;flex-wrap:wrap}
.tv-meta b{color:var(--ds-text);font-weight:600}
.tv-fig{width:100%;height:auto;background:var(--ds-bg);border:1px solid var(--ds-border);border-radius:6px}
.tv-fig .grid{stroke:var(--ds-border);stroke-width:1;stroke-dasharray:3 3}
.tv-fig .pred{fill:none;stroke:var(--ds-accent);stroke-width:1.6;opacity:.6}
.tv-fig .gt{fill:none;stroke:#16a34a;stroke-width:2.4}
.tv-fig .ego{fill:var(--ds-text)}
.tv-legend{font-size:11.5px;color:var(--ds-muted);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.tv-legend i{display:inline-block;width:15px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.tv-note{font-size:14px;line-height:1.75;color:var(--ds-text)}
.tv-method{font-size:14px;line-height:1.85;color:var(--ds-text)}
.tv-method code{background:var(--ds-bg);border:1px solid var(--ds-border);border-radius:4px;padding:1px 5px;font-size:13px}
.tv-method li{margin:5px 0}
</style>
</head>
<body>
<header class="ds-header">
  <div class="ds-header__inner">
    <div class="ds-header__row">
      <a class="ds-home" href="https://ranpin.github.io/">← 返回主页</a>
      <h1>Alpamayo-R1 轨迹预测 Demo</h1>
      <button class="ds-theme-btn" type="button" onclick="toggleTheme()" aria-label="切换明暗主题"><span class="ds-theme-icon">🌙</span></button>
    </div>
    <p class="ds-header__sub">NVIDIA Alpamayo-R1（10B 自动驾驶 VLA）部署在 Jetson AGX Orin，对真实驾驶片段输出 6.4 秒未来轨迹。BEV 俯视图对比真实行驶轨迹与模型的扩散采样预测。真实 Orin 推理输出。</p>
  </div>
</header>
<div class="ds-wrap">

  <section class="ds-section">
    <h2 class="ds-h2">演示</h2>
    <p class="tv-note">每条片段绘制 <b style="color:#16a34a">真实行驶轨迹（GT，绿）</b> 与模型的
      <b style="color:var(--ds-accent)">6 个扩散采样预测（蓝）</b>。坐标系为 ego 系俯视图（BEV）：自车位于原点、朝上，
      x 轴向前、y 轴向左。GT 取自自动驾驶数据集的测试子集，预测为 Orin 实机推理输出；片段已匿名化、
      页面不含量化基准指标，仅展示轨迹形态与多模态分布。</p>
    <div class="tv-controls">
      <label><input type="checkbox" id="showPreds" checked> 显示 6 个预测采样</label>
      <label><input type="checkbox" id="equalScale"> 统一坐标比例（否则每图自适应）</label>
    </div>
    <div class="tv-grid" id="grid"></div>
  </section>

  <section class="ds-section">
    <h2 class="ds-h2">核心方法</h2>
    <div class="tv-method">
    <ul>
      <li><b>模型</b>：Alpamayo-R1 = VLM backbone（Qwen3-VL 8B）+ 扩散动作专家（2.3B）。单次推理同时产出思维链文本与 <code>output_trajectory [64,2]</code>。</li>
      <li><b>轨迹积分</b>：动作头输出 <code>(accel, κ)</code> 序列，经 unicycle 运动学模型积分得 BEV 轨迹（6.4 s / 64 waypoint，10 Hz）。</li>
      <li><b>多模态采样</b>：扩散头对每条片段采样 6 个噪声种子，展示其轨迹分布（图中 6 条蓝线）——这是扩散式轨迹预测器的设计属性。</li>
      <li><b>边缘部署</b>：TensorRT-Edge-LLM 三引擎（llm / visual / action）建于 Jetson AGX Orin（sm_87 / JetPack 6.2）。action 专家的 <code>Attention</code>/<code>RotaryEmbedding</code>/<code>TensorScatter</code> 算子需 TRT≥11，用<b>分解注意力补丁</b>打穿该墙（Orin 的 TRT 10.7 不支持）。</li>
    </ul>
    </div>
  </section>

  <footer class="ds-footer">
    <div>Alpamayo-R1 边缘部署实证 · 真实 Jetson Orin 推理输出</div>
    <div>代码与部署细节见 <a href="https://github.com/ranpin/alpamayo-edge">github.com/ranpin/alpamayo-edge</a></div>
  </footer>
</div>
<script>
""" + data_line + """
const CLS_ZH={turn:"转弯",gentle:"缓弯",straight:"直行"};
const SPD_ZH={fast:"快",mid:"中",slow:"慢"};
function fmt(x){return (+x).toFixed(1);}
function render(){
  const showPreds=document.getElementById('showPreds').checked;
  const eq=document.getElementById('equalScale').checked;
  const grid=document.getElementById('grid');grid.innerHTML='';
  let gminx=1e9,gmaxx=-1e9,gminy=1e9,gmaxy=-1e9;
  if(eq){for(const c of DATA){for(const p of c.gt){gminx=Math.min(gminx,p[0]);gmaxx=Math.max(gmaxx,p[0]);gminy=Math.min(gminy,p[1]);gmaxy=Math.max(gmaxy,p[1]);}
    for(const pr of c.preds)for(const p of pr){gminx=Math.min(gminx,p[0]);gmaxx=Math.max(gmaxx,p[0]);gminy=Math.min(gminy,p[1]);gmaxy=Math.max(gmaxy,p[1]);}}}
  for(const c of DATA){
    const card=document.createElement('div');card.className='tv-card';
    let minx,maxx,miny,maxy;
    if(eq){minx=gminx;maxx=gmaxx;miny=gminy;maxy=gmaxy;}
    else{minx=1e9;maxx=-1e9;miny=1e9;maxy=-1e9;
      for(const p of c.gt){minx=Math.min(minx,p[0]);maxx=Math.max(maxx,p[0]);miny=Math.min(miny,p[1]);maxy=Math.max(maxy,p[1]);}
      if(showPreds)for(const pr of c.preds)for(const p of pr){minx=Math.min(minx,p[0]);maxx=Math.max(maxx,p[0]);miny=Math.min(miny,p[1]);maxy=Math.max(maxy,p[1]);}}
    const padX=(maxx-minx)*0.08+1,padY=(maxy-miny)*0.08+1;
    minx-=padX;maxx+=padX;miny-=padY;maxy+=padY;
    const W=300,H=250;
    const sy=p=>H-((p[0]-minx)/(maxx-minx))*H;
    const sx=p=>W-((p[1]-miny)/(maxy-miny))*W;
    const path=pts=>pts.map((p,i)=>(i?'L':'M')+sx(p).toFixed(1)+' '+sy(p).toFixed(1)).join(' ');
    let svg=`<svg class="tv-fig" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${c.id} 轨迹图">`;
    svg+=`<line class="grid" x1="0" y1="${sy([0,0]).toFixed(1)}" x2="${W}" y2="${sy([0,0]).toFixed(1)}"/>`;
    svg+=`<line class="grid" x1="${sx([0,0]).toFixed(1)}" y1="0" x2="${sx([0,0]).toFixed(1)}" y2="${H}"/>`;
    if(showPreds)for(const pr of c.preds)
      svg+=`<path class="pred" d="${path(pr)}"/>`;
    svg+=`<path class="gt" d="${path(c.gt)}"/>`;
    svg+=`<circle class="ego" cx="${sx([0,0]).toFixed(1)}" cy="${sy([0,0]).toFixed(1)}" r="4"/>`;
    svg+=`</svg>`;
    card.innerHTML=`<h3><span>${c.id}</span><span class="tv-tag">${CLS_ZH[c.cls]||c.cls} · ${SPD_ZH[c.spd]||c.spd}</span></h3>`+
      `<div class="tv-meta"><span>v0 <b>${fmt(c.v0)}</b> m/s</span><span>前行 <b>${fmt(c.fwd)}</b> m</span><span>max横向 <b>${fmt(c.maxlat)}</b> m</span></div>`+
      svg+
      `<div class="tv-legend"><span><i style="background:#16a34a"></i>真实轨迹 GT</span>`+
      (showPreds?`<span><i style="background:var(--ds-accent)"></i>预测采样 ×6</span>`:`<span style="opacity:.6">预测已隐藏</span>`)+`</div>`;
    grid.appendChild(card);
  }
}
document.getElementById('showPreds').addEventListener('change',render);
document.getElementById('equalScale').addEventListener('change',render);
render();
</script>
</body></html>"""

outdir = os.path.join(REPO, "docs", "site", "r1-trajectory")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "index.html")
open(out, "w").write(HTML)
print("wrote", out, os.path.getsize(out), "bytes")
print("clips:", len(pub), "| anon ids:", pub[0]["id"], "..", pub[-1]["id"])
# leak self-check on the written file
blob = open(out).read()
for bad in ["minade", "ades", '"best"', *list(SRC.keys())]:
    if bad in blob:
        print("!!! LEAK:", bad, "found in output")
print("leak self-check done (no output above = clean)")
