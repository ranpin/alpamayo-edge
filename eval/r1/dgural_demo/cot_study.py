#!/usr/bin/env python3
"""CoT–轨迹关系研究（18 锚点）：数字全部从两处已发布权威源提取——
静态 6 条 = docs/site/r1-trajectory/index.html 的 const DATA（cot/ade6/fde/turn）；
滚动 12 条 = eval/r1/dgural_demo/README.md 表 2（ADE/FDE）+ index.html 视频卡 CoT 原文。
分类为人工判定（判据见 README「CoT–轨迹关系研究」节），分类映射在下方 CLS，本脚本只做提取+连接+统计。"""
import re, json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
html = open(f'{REPO}/docs/site/r1-trajectory/index.html', encoding='utf-8').read()

# ---- 静态 6：DATA ----
m = re.search(r'const DATA=(\[.*?\]);', html, re.S)
static = json.loads(m.group(1))
assert len(static) == 6, f'DATA 解析到 {len(static)} 条，应 6'

# ---- 滚动 12：README 表 2 ----
readme = open(f'{REPO}/eval/r1/dgural_demo/README.md', encoding='utf-8').read()
roll = {}
for line in readme.splitlines():
    mm = re.match(r'\|\s*(0dd3eb7f|ecafce84|1fcf2fef|1f7ef408)\s*\|([^|]+)\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*\|\s*([^|]+)\|', line)
    if mm:
        cid8, scene, t0, v0, ade6, fde, ef, el, form = mm.groups()
        roll[f'{cid8}_t{t0}'] = dict(scene=scene.strip(), t0=float(t0), ade6=float(ade6), fde=float(fde),
                                     ef=float(ef), el=float(el), form=form.strip())
assert len(roll) == 12, f'表2 解析到 {len(roll)} 行，应 12'

# ---- 滚动 12 CoT 原文：index.html 视频卡（按卡切分保证归属）----
cards = re.findall(r'图 (\d+) · 片段 (\w+)（[^）]+）滚动预测.*?<div class="r1t-cotlist">(.*?)<div class="r1t-note">', html, re.S)
cot_by_tag = {}
for fig, cid8, blk in cards:
    for t0v, txt in re.findall(r'<b>t₀=([\d.]+)s</b>\s*[“"](.+?)[”"]', blk):
        cot_by_tag[f'{cid8}_t{t0v}'] = txt
assert len(cot_by_tag) == 12, f'CoT 解析到 {len(cot_by_tag)} 条，应 12'

# ---- 人工分类 ----
# A=点对机动  A2=场景对/动作时点错  B=退化泛化跟车(实际有机动)  C=文本-轨迹解耦(文本与轨迹或实际矛盾)
CLS = {
  'ecafce84_s':      ('A',  '左转', '点出左转（绿灯）'),
  '76f4a312_s':      ('A',  '右转', '点出进入右转道'),
  '46003675_s':      ('C',  '左转', '说 keep lane/随曲率，轨迹过度左转'),
  '1f7ef408_s':      ('B',  '右转', '泛化跟车，未点右转→欠转'),
  'fa83bcb8_s':      ('B',  '右转', '泛化跟车，未点右转→欠转'),
  '058bf960_s':      ('B',  '左转', '泛化跟车，漏判左转'),
  '0dd3eb7f_t1.5':   ('A',  '直行跟车', 'keep speed 对场景（误差=纵向量级）'),
  '0dd3eb7f_t7.5':   ('A',  '直行跟车', 'keep distance 对场景'),
  '0dd3eb7f_t13.5':  ('A',  '直行跟车', 'keep distance 对场景'),
  'ecafce84_t1.5':   ('A2', '红灯→蠕行', '识别红灯对；CoT=轨迹=停住，GT 蠕行 5.5m（时点错）'),
  'ecafce84_t7.5':   ('C',  '左转', '说直行过口，轨迹左转（轨迹对）'),
  'ecafce84_t13.5':  ('C',  '左转', '说右转，轨迹与 GT 均左转（轨迹大体对）'),
  '1fcf2fef_t1.5':   ('A',  '右转', '点出右转（绿灯）→最佳 ADE 0.60'),
  '1fcf2fef_t7.5':   ('B',  '右转', '调速跟车，未点右转→欠转 6.9m'),
  '1fcf2fef_t13.5':  ('C',  '转后直行', '说左nudge，实际近直行（轻微）'),
  '1f7ef408_t1.5':   ('B',  '变道', '泛化跟车，未点变道'),
  '1f7ef408_t7.5':   ('C',  '近直行', '说右变道，轨迹与 GT 近直行略偏左（轨迹对）'),
  '1f7ef408_t13.5':  ('B',  '右转', '泛化跟车，未点右转→欠转 6.8m'),
}

rows = []
for d in static:
    k = d['id'] + '_s'
    cat, scene, claim = CLS[k]
    rows.append(dict(key=k, src='静态', scene=scene, claim=claim, cat=cat,
                     ade6=d['ade6'], fde=d['fde'], cot=d['cot']))
for tag, r in roll.items():
    cat, scene, claim = CLS[tag]
    rows.append(dict(key=tag, src='滚动', scene=scene, claim=claim, cat=cat,
                     ade6=r['ade6'], fde=r['fde'], cot=cot_by_tag[tag], form=r['form'], el=r['el']))

print(f"{'锚点':18} {'源':4} {'类别':3} {'ADE6':>5} {'FDE':>5}  场景 | CoT 判定")
for r in sorted(rows, key=lambda x: (x['cat'], -x['ade6'])):
    print(f"{r['key']:18} {r['src']:4} {r['cat']:3} {r['ade6']:5.2f} {r['fde']:5.2f}  {r['scene']} | {r['claim']}")

print('\n=== 分类统计 ===')
NAMES = {'A': 'A 点对机动', 'A2': "A' 场景对/时点错", 'B': 'B 退化跟车', 'C': 'C 文本-轨迹解耦'}
for cat in ['A', 'A2', 'B', 'C']:
    sub = [r for r in rows if r['cat'] == cat]
    ades = [r['ade6'] for r in sub]
    print(f"{NAMES[cat]:16} n={len(sub)}  ADE6 mean={np.mean(ades):.2f}  range=[{min(ades):.2f},{max(ades):.2f}]")
b = [r['ade6'] for r in rows if r['cat'] == 'B']; a = [r['ade6'] for r in rows if r['cat'] == 'A']
c = [r['ade6'] for r in rows if r['cat'] == 'C']
print(f"\nB/A 均值比 = {np.mean(b)/np.mean(a):.2f}×")
print(f"最差 3 条 ADE: {sorted([(r['ade6'], r['key'], r['cat']) for r in rows], reverse=True)[:3]}")
print(f"C 类中轨迹大体对的（ADE<2.2）: {sum(1 for x in c if x < 2.2)}/{len(c)}（阈值取 2.5 结论不变）")
