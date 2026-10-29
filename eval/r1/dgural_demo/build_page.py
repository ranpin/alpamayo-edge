import json, os, shutil
import numpy as np

SITE='/Users/ranpin/projects/alpamayo-edge/docs/site/r1-trajectory'
pred=json.load(open('/tmp/cdd_poc/r1_multi_pred.json'))
mcd=json.load(open('/tmp/cdd_poc/multi_clip_data.json'))

NOTES={
 'ecafce84':'CoT 正确识别「绿灯左转」→ 轨迹几乎完美贴合（本组最佳）',
 '76f4a312':'CoT 识别「并入右转道并减速」→ 贴合良好',
 '46003675':'CoT 说「保持车道」，轨迹却过度左转 —— 思维链与动作脱节',
 '1f7ef408':'CoT 只盯「跟前车」→ 对右转欠转、冲得偏远',
 'fa83bcb8':'CoT 关注「让行标志减速」→ 高速右转欠转',
 '058bf960':'CoT 只盯「跟前车」→ 漏判即将到来的左转（蓝线直冲，本组最差）',
}
DATA=[]
for cid8,p in pred.items():
    ge=np.array(mcd[cid8]['gt_xyz']); lat=ge[-1,1]; fwd=ge[-1,0]
    DATA.append({'id':cid8,'turn':('左转' if lat>0 else '右转'),'lat':round(abs(lat)),'fwd':round(fwd),
                 'v0':round(p['v0'],1),'ade3':round(p['ade3'],2),'ade6':round(p['ade6'],2),
                 'fde':round(p['fde'],2),'cot':p['cot'],'note':NOTES[cid8],'img':f'assets/{cid8}.jpg'})
DATA.sort(key=lambda d:d['ade6'])   # ADE6 升序：好→差
for i,d in enumerate(DATA): d['seq']=i+1

os.makedirs(f'{SITE}/assets',exist_ok=True)
for d in DATA:
    shutil.copy(f'/tmp/cdd_poc/web_assets/{d["id"]}.jpg', f'{SITE}/assets/{d["id"]}.jpg')

HTML=r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alpamayo-R1 轨迹预测 · 真实帧叠加 Demo · Jetson Orin</title>
<meta name="description" content="Alpamayo-R1（10B 自动驾驶 VLA）部署在 Jetson AGX Orin（TensorRT-Edge-LLM 三引擎），对真实驾驶片段输出 6.4s 未来轨迹，与真值 GT 一起用鱼眼标定投影叠回原始前视相机画面。全部为 Orin 实机推理输出。">
<meta property="og:title" content="Alpamayo-R1 轨迹预测 · 真实帧叠加 Demo">
<meta property="og:description" content="10B 自动驾驶 VLA 在 Jetson Orin 实机推理，预测轨迹与真值一起叠回真实前视相机画面。">
<meta property="og:image" content="https://ranpin.github.io/alpamayo-edge/r1-trajectory/assets/ecafce84.jpg">
<meta property="og:url" content="https://ranpin.github.io/alpamayo-edge/r1-trajectory/">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>🛣️</text></svg>">
<link rel="stylesheet" href="https://ranpin.github.io/demo-shell.css">
<script src="https://ranpin.github.io/theme.js"></script>
<style>
.r1t-legend{display:flex;gap:20px;align-items:center;flex-wrap:wrap;font-size:13.5px;color:var(--ds-text);margin:10px 0 16px;justify-content:center}
.r1t-legend i{display:inline-block;width:22px;height:4px;border-radius:2px;margin-right:7px;vertical-align:middle}
.r1t-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:18px}
.r1t-card{background:var(--ds-panel);border:1px solid var(--ds-border);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.r1t-figwrap{position:relative}
.r1t-fig{width:100%;height:auto;display:block;background:var(--ds-bg)}
.r1t-seq{position:absolute;top:10px;right:10px;min-width:31px;height:31px;padding:0 5px;border-radius:16px;background:var(--ds-accent);color:#fff;font-weight:700;font-size:15px;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 6px rgba(0,0,0,.55)}
.r1t-seqsm{display:inline-flex;min-width:19px;height:19px;padding:0 3px;border-radius:10px;background:var(--ds-accent);color:#fff;font-weight:700;font-size:11.5px;align-items:center;justify-content:center;vertical-align:middle;margin-right:6px}
.r1t-cap{padding:12px 15px 14px;display:flex;flex-direction:column;gap:9px}
.r1t-captop{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.r1t-tag{font-size:12px;padding:2px 11px;border-radius:11px;background:var(--ds-accent-soft);color:var(--ds-accent-strong);border:1px solid var(--ds-accent-ring);white-space:nowrap;font-weight:600}
.r1t-metrics{font-size:12.5px;color:var(--ds-muted);display:flex;gap:11px;flex-wrap:wrap}
.r1t-metrics b{color:var(--ds-text);font-weight:700}
.r1t-cot{font-size:13px;line-height:1.6;color:var(--ds-text);font-style:italic;border-left:3px solid var(--ds-accent);padding-left:11px}
.r1t-note{font-size:12.5px;color:var(--ds-muted);line-height:1.6}
.r1t-tablewrap{overflow-x:auto;margin-top:6px}
.r1t-table{width:100%;border-collapse:collapse;font-size:13.5px;min-width:560px}
.r1t-table th,.r1t-table td{padding:9px 11px;text-align:center;border-bottom:1px solid var(--ds-border)}
.r1t-table th{color:var(--ds-muted);font-weight:600;font-size:12.5px}
.r1t-table td.l,.r1t-table th.l{text-align:left}
.r1t-table tr:hover td{background:var(--ds-bg)}
.r1t-best{color:#16a34a;font-weight:700}.r1t-worst{color:#dc2626;font-weight:700}
.r1t-cal{font-size:12.5px;color:var(--ds-muted);line-height:1.75;margin-top:14px;padding:12px 14px;background:var(--ds-bg);border:1px solid var(--ds-border);border-radius:8px}
.r1t-method{font-size:14px;line-height:1.85;color:var(--ds-text)}
.r1t-method code{background:var(--ds-bg);border:1px solid var(--ds-border);border-radius:4px;padding:1px 5px;font-size:13px}
.r1t-method li{margin:6px 0}
@media(max-width:680px){.r1t-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="ds-header">
  <div class="ds-header__inner">
    <div class="ds-header__row">
      <a class="ds-home" href="https://ranpin.github.io/">← 返回主页</a>
      <h1>Alpamayo-R1 轨迹预测 · 真实帧叠加</h1>
      <button class="ds-theme-btn" type="button" onclick="toggleTheme()" aria-label="切换明暗主题"><span class="ds-theme-icon">🌙</span></button>
    </div>
    <p class="ds-header__sub">NVIDIA Alpamayo-R1（10B 自动驾驶 VLA）部署在 Jetson AGX Orin（TensorRT-Edge-LLM 三引擎），对真实驾驶片段输出 6.4 秒未来轨迹，与真值（GT）一起用该片段的鱼眼相机标定投影叠回<b>原始前视相机画面</b>。全部为 Orin 实机推理输出。</p>
  </div>
</header>
<div class="ds-wrap">

  <section class="ds-section">
    <h2 class="ds-h2">演示</h2>
    <p>每张图为一个真实驾驶片段的<b>前视相机原始帧</b>，叠加两条 6.4 秒未来轨迹：<b style="color:#16a34a">GT 真值（绿）</b>取自数据集 egomotion 标签、<b style="color:var(--ds-accent)">Alpamayo-R1 预测（蓝）</b>为 Orin 实机推理输出。两者都用该片段实测的鱼眼标定投影回图像平面，因此贴合实际路面。</p>
    <p>卡片标签 <b>「6.4s 终点：前行 X m · 左/右偏 Y m」</b> 是该片段 <b>GT 轨迹在 6.4 秒预测窗末端</b>、相对自车 t₀ 位置的<b>位移分解</b>——前行 X m 为纵向分量、左/右偏 Y m 为横向分量，合起来描述<b>终点落在哪</b>（一个位移矢量的两个分量），<b>不是</b>「先直行 X m 再转弯」这类分段机动指令。卡片另标初速 v₀ 与 ADE@6.4s（误差指标的公式与口径见下方「评测结果」），以及模型同时产出的思维链（CoT）。每张图右上角的<b>序号 1–6</b> 按 ADE@6.4s 从好到差编排，与下方「评测结果」表逐行对应。</p>
    <div class="r1t-legend">
      <span><i style="background:#16a34a"></i>GT 真值轨迹</span>
      <span><i style="background:var(--ds-accent)"></i>Alpamayo-R1 预测</span>
      <span style="color:var(--ds-muted)">ADE / FDE 误差公式与口径见下方「评测结果」</span>
    </div>
    <div class="r1t-grid" id="grid"></div>
  </section>

  <section class="ds-section">
    <h2 class="ds-h2">评测结果</h2>
    <p>下表为 6 个片段上 R1 预测相对 GT 的逐点欧氏位移误差（ego 系，按 ADE@6.4s 升序）。<b>核心观察：R1 的思维链与其轨迹精度强相关</b>——当 CoT 正确识别出机动（如「绿灯左转」「并入右转道」）时轨迹几乎完美贴合（ADE 低至 0.84m）；当 CoT 退化为泛化的「跟前车」时，模型会漏判或欠判即将到来的转弯（最差例蓝线直冲、绿线左拐，ADE 5.90m）。</p>
    <div class="r1t-tablewrap"><table class="r1t-table" id="tbl"></table></div>
    <div class="r1t-cal"><b>误差公式</b>：ADE@T（平均位移误差）= 预测与 GT 在前 T 秒每个 waypoint 上欧氏距离的均值 = (1/N)·Σᵢ‖predᵢ − gtᵢ‖；FDE@T（终点位移误差）= 第 T 秒末 waypoint 的欧氏距离 = ‖pred_N − gt_N‖。@3s 取前 30 点、@6.4s 取全部 64 点（10 Hz）；均在 t₀ 时刻 ego 系（自车为原点、x 前 y 左）计算，单位米。<br><b>口径与诚实边界</b>：① 单噪声种子（seed=5、temperature=0）确定性推理，<b>非</b>官方 minADE6（6 采样取最优）；② 6 个片段均为「6.4s 预测窗内明显转弯」的困难样本、非随机抽样，误差天然高于官方全集；③ 数据源 dgural 公开样本的 README 明确声明「不适合 benchmarking / 统计分析」——本页数字仅为<b>逐例行为展示</b>，不构成代表性精度结论；④ 官方 Alpamayo-R1 在 644 clip 上报告 minADE6@6.4s≈0.79m，口径不同，不可与本表直接比较。</div>
  </section>

  <section class="ds-section">
    <h2 class="ds-h2">核心方法</h2>
    <div class="r1t-method"><ul>
      <li><b>模型</b>：Alpamayo-R1 = VLM backbone（Qwen3-VL 8B）+ 扩散动作专家（2.3B）。单次推理同时产出思维链文本与 <code>output_trajectory [64,2]</code>（归一化 <code>accel, κ</code>）。</li>
      <li><b>输入</b>：4 路相机（cross_left / front_wide / cross_right / front_tele）× 4 个历史时刻（t₀−0.3…t₀）共 16 帧 + 16 点 ego 历史轨迹。</li>
      <li><b>边缘部署</b>：TensorRT-Edge-LLM 三引擎（llm / visual / action）建于 Jetson AGX Orin（sm_87 / JetPack 6.2）。action 专家的 <code>Attention</code>/<code>RotaryEmbedding</code>/<code>TensorScatter</code> 算子需 TRT≥11，用<b>分解注意力补丁</b>打穿该墙（Orin 的 TRT 10.7 不原生支持）。</li>
      <li><b>轨迹积分</b>：动作头输出经运动学自行车模型积分得 BEV 轨迹（6.4s / 64 点 / 10Hz）；初速 v₀ 由 ego 历史经官方 <code>compute_minade</code> 的最小二乘（Cholesky）反解，与 minADE 评测同一套后处理口径。</li>
      <li><b>投影叠加</b>：GT 与 R1 轨迹均用该片段实测的<b>鱼眼相机标定</b>（多项式畸变 fw_poly + 四元数外参）投影回前视图像；GT 带真实路面高程 z，R1 的 2D 输出按 GT 路面高程插值赋 z，使两者都贴合路面。</li>
      <li><b>数据来源</b>：图像帧取自 dgural 公开样本（NVIDIA PhysicalAI-AV 的公开 100-clip 子集）；GT 轨迹与相机标定取自 gated NVIDIA PhysicalAI-AV；帧↔egomotion 时间对齐用数据集自带的逐帧时间戳权威校准（非按 fps 推算）。</li>
    </ul></div>
  </section>

  <footer class="ds-footer">
    <div>Alpamayo-R1 边缘部署实证 · 真实 Jetson Orin 推理输出 · 真实相机帧叠加</div>
    <div>代码与部署细节见 <a href="https://github.com/ranpin/alpamayo-edge">github.com/ranpin/alpamayo-edge</a></div>
  </footer>
</div>
<script>
const DATA=__DATA__;
const grid=document.getElementById('grid');
DATA.forEach(d=>{
  const c=document.createElement('div');c.className='r1t-card';
  c.innerHTML=`<div class="r1t-figwrap"><img class="r1t-fig" src="${d.img}" alt="示例${d.seq} ${d.id} 真实帧轨迹叠加：GT 与 R1 预测" loading="lazy"><span class="r1t-seq">${d.seq}</span></div>
   <div class="r1t-cap">
    <div class="r1t-captop"><span class="r1t-tag">6.4s 终点：前行 ${d.fwd}m · ${d.turn==='右转'?'右偏':'左偏'} ${d.lat}m</span>
      <span class="r1t-metrics">v₀=<b>${d.v0}</b> m/s · ADE@6.4s=<b>${d.ade6}</b> m</span></div>
    <div class="r1t-cot">“${d.cot}”</div>
    <div class="r1t-note">${d.note}</div>
   </div>`;
  grid.appendChild(c);
});
const tbl=document.getElementById('tbl');
const best=Math.min(...DATA.map(d=>d.ade6)),worst=Math.max(...DATA.map(d=>d.ade6));
let h='<thead><tr><th class="l">示例</th><th>转向</th><th>v₀ (m/s)</th><th>ADE@3s (m)</th><th>ADE@6.4s (m)</th><th>FDE@6.4s (m)</th></tr></thead><tbody>';
DATA.forEach(d=>{
  const cls=d.ade6===best?'r1t-best':(d.ade6===worst?'r1t-worst':'');
  h+=`<tr><td class="l"><span class="r1t-seqsm">${d.seq}</span><code>${d.id}</code></td><td>${d.turn}</td><td>${d.v0}</td><td>${d.ade3}</td><td class="${cls}">${d.ade6}</td><td>${d.fde}</td></tr>`;
});
h+='</tbody>';tbl.innerHTML=h;
</script>
</body>
</html>'''

html=HTML.replace('__DATA__', json.dumps(DATA,ensure_ascii=False,separators=(',',':')))
open(f'{SITE}/index.html','w').write(html)
print('index.html bytes:',len(html))
print('DATA clips:',[d['id'] for d in DATA])
print('assets:',sorted(os.listdir(f'{SITE}/assets')))
