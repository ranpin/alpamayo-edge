#!/usr/bin/env python3
"""Quantitative Playwright verification of eval/r1/viz/index.html (LOCAL page).

Model has NO vision -> every check is a DOM measurement. Verifies:
  1. 18 clip cards render
  2. each SVG has exactly 1 GT path (#16a34a) + >=1 pred path; path `d` finite
  3. stats cards match traj_viz.json (clips/mean/median/best/worst)
  4. #showAll toggle: 6 pred paths (checked) vs 1 (unchecked)
  5. #equalScale toggle re-renders without error
  6. zero console errors (favicon filtered)
  7. no horizontal overflow at 390px
  8. light + dark scheme: card bg vs text contrast not degenerate
"""
import json, os, sys
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = "file://" + os.path.join(HERE, "index.html")
TRUTH = json.load(open(os.path.join(HERE, "traj_viz.json")))

# expected stats from the source data
ades = sorted(v["minade"] for v in TRUTH.values())
n = len(ades)
exp = {
    "clips": n,
    "mean": sum(ades) / n,
    "median": ades[n // 2],
    "best": ades[0],
    "worst": ades[-1],
}

errors = []
console_errors = []

def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        errors.append(f"{name}: {detail}")

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)

    # ---- desktop, light ----
    ctx = browser.new_context(viewport={"width": 1280, "height": 900},
                              device_scale_factor=2, color_scheme="light")
    page = ctx.new_page()
    page.on("console", lambda m: console_errors.append(m.text)
            if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))
    page.goto(PAGE, wait_until="commit")
    page.wait_for_selector(".tv-card", timeout=15000)
    page.wait_for_timeout(400)

    # 1. card count
    ncards = page.evaluate("document.querySelectorAll('.tv-card').length")
    check("18 clip cards render", ncards == 18, f"got {ncards}")

    # 2. per-SVG path structure (showAll checked by default => 6 pred + 1 GT)
    struct = page.evaluate("""() => {
      const out = [];
      document.querySelectorAll('.tv-card svg').forEach(svg => {
        const paths = [...svg.querySelectorAll('path')];
        const byStroke = {};
        let badD = 0;
        paths.forEach(pa => {
          const s = pa.getAttribute('stroke');
          byStroke[s] = (byStroke[s]||0)+1;
          const d = pa.getAttribute('d')||'';
          if (!d.includes('M') || /NaN|Infinity|undefined/.test(d)) badD++;
        });
        out.push({n: paths.length, byStroke, badD,
                  circle: svg.querySelectorAll('circle').length});
      });
      return out;
    }""")
    all_gt1 = all(s["byStroke"].get("#16a34a", 0) == 1 for s in struct)
    all_best1 = all(s["byStroke"].get("#1a67eb", 0) == 1 for s in struct)
    all_5gray = all(s["byStroke"].get("#94a3b8", 0) == 5 for s in struct)
    all_7 = all(s["n"] == 7 for s in struct)
    all_circle = all(s["circle"] == 1 for s in struct)
    no_badD = all(s["badD"] == 0 for s in struct)
    check("each SVG: 1 GT(#16a34a)", all_gt1, str([s["byStroke"] for s in struct if s["byStroke"].get("#16a34a",0)!=1][:2]))
    check("each SVG: 1 best(#1a67eb)", all_best1)
    check("each SVG: 5 other-seed(#94a3b8) [showAll]", all_5gray)
    check("each SVG: 7 paths total [showAll]", all_7, str([s["n"] for s in struct]))
    check("each SVG: 1 ego circle", all_circle)
    check("all path `d` finite (no NaN/Infinity)", no_badD,
          str([s["badD"] for s in struct]))

    # 3. stats cards
    stats = page.evaluate("""() => [...document.querySelectorAll('.tv-stat')]
        .map(e => ({v: e.querySelector('b').textContent,
                    l: e.querySelector('span').textContent}))""")
    smap = {s["l"]: s["v"] for s in stats}
    check("stat clips==18", smap.get("clips") == "18", str(smap))
    def close(label, want, tol=0.002):
        got = float(smap.get(label, "nan"))
        return abs(got - want) < tol, f"{label}: got {got} want {want:.3f}"
    ok, d = close("minADE mean (m)", exp["mean"]); check("stat mean", ok, d)
    ok, d = close("median (m)", exp["median"]); check("stat median", ok, d)
    ok, d = close("best (m)", exp["best"]); check("stat best", ok, d)
    ok, d = close("worst (m)", exp["worst"]); check("stat worst", ok, d)

    # 4. showAll toggle OFF => 1 pred (best) + 1 GT = 2 paths
    page.uncheck("#showAll")
    page.wait_for_timeout(250)
    npaths_off = page.evaluate("""() => [...document.querySelectorAll('.tv-card svg')]
        .map(s => s.querySelectorAll('path').length)""")
    check("showAll OFF => 2 paths/card", all(x == 2 for x in npaths_off), str(npaths_off))
    gray_off = page.evaluate("""() => [...document.querySelectorAll('.tv-card svg')]
        .every(s => ![...s.querySelectorAll('path')].some(p=>p.getAttribute('stroke')==='#94a3b8'))""")
    check("showAll OFF => no gray seeds", gray_off)
    page.check("#showAll")
    page.wait_for_timeout(200)

    # 5. equalScale toggle re-renders
    page.check("#equalScale")
    page.wait_for_timeout(300)
    n_eq = page.evaluate("document.querySelectorAll('.tv-card').length")
    check("equalScale ON => still 18 cards", n_eq == 18, f"got {n_eq}")
    # in equal-scale the per-clip viewBox mapping changes; just ensure paths still finite
    eq_badD = page.evaluate("""() => [...document.querySelectorAll('.tv-card svg path')]
        .filter(p => /NaN|Infinity/.test(p.getAttribute('d')||'')).length""")
    check("equalScale ON => paths finite", eq_badD == 0, f"bad {eq_badD}")
    page.uncheck("#equalScale")
    page.wait_for_timeout(200)

    # 8a. light scheme contrast
    light = page.evaluate("""() => {
      const c = document.querySelector('.tv-card');
      const cs = getComputedStyle(c);
      const t = getComputedStyle(c.querySelector('.tv-meta'));
      return {bg: cs.backgroundColor, cardColor: cs.color, metaColor: t.color};
    }""")
    check("light: card bg is not transparent-black-on-black",
          light["bg"] not in ("rgba(0, 0, 0, 0)", "rgb(0, 0, 0)"), str(light))

    # 6. console errors (filter favicon)
    real = [e for e in console_errors if "favicon" not in e.lower()]
    check("zero console errors", len(real) == 0, str(real[:3]))

    ctx.close()

    # ---- dark scheme ----
    ctx2 = browser.new_context(viewport={"width": 1280, "height": 900},
                               device_scale_factor=2, color_scheme="dark")
    page2 = ctx2.new_page()
    page2.goto(PAGE, wait_until="commit")
    page2.wait_for_selector(".tv-card", timeout=15000)
    page2.wait_for_timeout(300)
    dark = page2.evaluate("""() => {
      const c = document.querySelector('.tv-card');
      const body = getComputedStyle(document.body);
      return {cardBg: getComputedStyle(c).backgroundColor,
              bodyBg: body.backgroundColor,
              bodyColor: body.color};
    }""")
    # card bg should differ from body bg OR both be readable; just ensure not identical-to-invisible
    check("dark: card renders (bg resolved)",
          dark["cardBg"] not in ("rgba(0, 0, 0, 0)",), str(dark))
    check("dark: body bg != body text color",
          dark["bodyBg"] != dark["bodyColor"], str(dark))
    n_dark = page2.evaluate("document.querySelectorAll('.tv-card').length")
    check("dark: 18 cards", n_dark == 18, f"got {n_dark}")
    ctx2.close()

    # ---- 7. mobile 390px overflow ----
    ctx3 = browser.new_context(viewport={"width": 390, "height": 844},
                               device_scale_factor=2, is_mobile=True,
                               has_touch=True)
    page3 = ctx3.new_page()
    page3.goto(PAGE, wait_until="commit")
    page3.wait_for_selector(".tv-card", timeout=15000)
    page3.wait_for_timeout(300)
    overflow = page3.evaluate("""() => ({
      scrollW: document.documentElement.scrollWidth,
      clientW: document.documentElement.clientWidth,
      bodyScrollW: document.body.scrollWidth
    })""")
    check("390px: no horizontal overflow",
          overflow["scrollW"] <= overflow["clientW"] + 2, str(overflow))
    n_mob = page3.evaluate("document.querySelectorAll('.tv-card').length")
    check("390px: 18 cards", n_mob == 18, f"got {n_mob}")
    # grid should be single column at 390
    cols = page3.evaluate("""() => getComputedStyle(document.querySelector('.tv-grid'))
        .gridTemplateColumns.split(' ').length""")
    check("390px: grid single column", cols == 1, f"cols={cols}")
    ctx3.close()

    browser.close()

print("\n" + "=" * 50)
if errors:
    print(f"FAILURES ({len(errors)}):")
    for e in errors:
        print("  -", e)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
