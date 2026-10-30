#!/usr/bin/env python3
"""从 GT 未来轨迹反推 oracle nav，建 nav 版请求 io/r1_eval_in18_nav.json（在 goat 上跑）。

为何要反推（2026-10-30）：18 个 eval clip 无导航标注（gt 只有 gt_xy/ego_history），官方 nav_demo
是另一批 clip。要测「1.5 是否需要导航引导」只能从 GT 反推 oracle nav——这是上界测试，会泄漏 GT
转向意图，结论须按此口径披露（见 eval/r1/crossgen_a15.md §四/§六）。

nav 格式对齐官方 helper.create_message：user 文本 = traj_history 占位 + <|route_start|>{nav}<|route_end|>
+ prompt。TRTEdge 请求里 trajectory 是独立 content 项（runtime 自转 48 个 traj_history token），
故只需把 text 项前缀加上 route 段，token 顺序即与官方逐位一致。

maneuver 判据：按 GT 未来轨迹的航向变化量（末段航向 − 首段航向，非起终点连线角）分
RIGHT(<-12°)/LEFT(>12°)/STRAIGHT；距离取航向首次偏过 10° 处的弧长（round 到 5m）。
本批 18 clip 实测 6 RIGHT + 12 STRAIGHT + 0 LEFT。

用法（goat）：python3 derive_a15_oracle_nav.py
读 eval_data/gt_chunk3116.json + io/r1_eval_in18.json，写 io/r1_eval_in18_nav.json。
"""
import copy
import json
import math
from collections import Counter

A = "/home/nvidia/chenrunbin/alpamayo-edge"


def norm(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def derive(xy):
    s = [0.0]
    for i in range(1, len(xy)):
        s.append(s[-1] + math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]))
    hd = [math.atan2(xy[i + 1][1] - xy[i][1], xy[i + 1][0] - xy[i][0]) for i in range(len(xy) - 1)]
    h0 = hd[0]
    total = math.degrees(norm(hd[-1] - h0))
    ts = s[-1]
    for i, h in enumerate(hd):
        if abs(math.degrees(norm(h - h0))) > 10:
            ts = s[i]
            break
    dm = max(5, int(round(ts / 5) * 5))
    if total < -12:
        return f"Turn right in {dm}m", "RIGHT"
    if total > 12:
        return f"Turn left in {dm}m", "LEFT"
    return "Continue straight", "STRAIGHT"


gt = json.load(open(f"{A}/eval_data/gt_chunk3116.json"))
navs = {cid: derive(g["gt_xy"]) for cid, g in gt.items()}

d = json.load(open(f"{A}/io/r1_eval_in18.json"))
dn = copy.deepcopy(d)
for r in dn["requests"]:
    nav = navs.get(r["id"], ("Continue straight", "STRAIGHT"))[0]
    for m in r["messages"]:
        if m["role"] == "user":
            for it in m["content"]:
                if it.get("type") == "text":
                    it["text"] = f"<|route_start|>{nav}<|route_end|>{it['text']}"
json.dump(dn, open(f"{A}/io/r1_eval_in18_nav.json", "w"))

print("built r1_eval_in18_nav.json |", dict(Counter(c for _, c in navs.values())))
for cid, (nav, cls) in sorted(navs.items(), key=lambda x: (x[1][1], x[0])):
    print(f"  {cid[:8]} [{cls:8}] {nav}")
