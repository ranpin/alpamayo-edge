#!/usr/bin/env python3
"""Playwright verification of the PUBLIC desensitized demo
docs/site/r1-trajectory/index.html. Model has NO vision -> all DOM measurements.

Critical: LEAK checks (no real clip UUID, no ades/minade/best in embedded DATA,
no minADE/ADE text) on top of structure/interaction/theme/mobile/console.
"""
import json, os, sys
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))            # eval/r1/viz
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PAGE = "file://" + os.path.join(REPO, "docs/site/r1-trajectory/index.html")
SRC = json.load(open(os.path.join(HERE, "traj_viz.json")))
UUIDS = list(SRC.keys())
PREFIXES = [u[:8] for u in UUIDS]

errors, console_errors = [], []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond: errors.append(f"{name}: {detail}")

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(viewport={"width":1280,"height":900},
                              device_scale_factor=2, color_scheme="light")
    page = ctx.new_page()
    page.on("console", lambda m: console_errors.append(m.text) if m.type=="error" else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))
    page.goto(PAGE, wait_until="commit")
    page.wait_for_selector(".tv-card", timeout=15000)
    page.wait_for_timeout(500)

    # ---- LEAK CHECKS (most important) ----
    html = page.content()
    uuid_leaks = [u for u in UUIDS if u in html] + [pfx for pfx in PREFIXES if pfx in html]
    check("LEAK: no real clip UUID in page", len(uuid_leaks)==0, str(uuid_leaks[:3]))
    data_keys = page.evaluate("()=>Object.keys(DATA[0]).sort().join(',')")
    check("LEAK: DATA[0] keys are desensitized",
          data_keys=="cls,fwd,gt,id,maxlat,preds,spd,v0", data_keys)
    bad_keys = page.evaluate("()=>DATA.some(c=>('ades' in c)||('minade' in c)||('best' in c))")
    check("LEAK: no ades/minade/best in any DATA record", bad_keys==False)
    body_txt = page.evaluate("()=>document.body.innerText")
    leak_words = [w for w in ["minADE","minade","ADE:","0.804","0.602","2.504"] if w in body_txt]
    check("LEAK: no minADE/ADE text rendered", len(leak_words)==0, str(leak_words))
    anon_ok = page.evaluate("()=>DATA.every((c,i)=>c.id===`clip-${String(i+1).padStart(2,'0')}`)")
    check("anon ids are clip-01..clip-18 in order", anon_ok)

    # ---- structure ----
    ncards = page.evaluate("document.querySelectorAll('.tv-card').length")
    check("18 clip cards render", ncards==18, f"got {ncards}")
    struct = page.evaluate("""()=>[...document.querySelectorAll('.tv-card svg')].map(svg=>{
        const ps=[...svg.querySelectorAll('path')];let badD=0;
        ps.forEach(pa=>{const d=pa.getAttribute('d')||'';if(!d.includes('M')||/NaN|Infinity|undefined/.test(d))badD++;});
        return {n:ps.length, gt:svg.querySelectorAll('path.gt').length,
                pred:svg.querySelectorAll('path.pred').length, badD,
                ego:svg.querySelectorAll('circle.ego').length};})""")
    check("each SVG: 1 GT(.gt)", all(s["gt"]==1 for s in struct), str([s["gt"] for s in struct]))
    check("each SVG: 6 preds(.pred) [showPreds on]", all(s["pred"]==6 for s in struct),
          str([s["pred"] for s in struct]))
    check("each SVG: 7 paths total", all(s["n"]==7 for s in struct), str([s["n"] for s in struct]))
    check("each SVG: 1 ego circle(.ego)", all(s["ego"]==1 for s in struct))
    check("all path d finite", all(s["badD"]==0 for s in struct))
    # CSS classes must actually resolve colors (var() works in class, not presentation attr)
    colors = page.evaluate("""()=>{const g=document.querySelector('.tv-card svg path.gt');
        const pr=document.querySelector('.tv-card svg path.pred');
        const eg=document.querySelector('.tv-card svg circle.ego');
        const cd=document.querySelector('.tv-card');
        return {gt:getComputedStyle(g).stroke, pred:getComputedStyle(pr).stroke,
                ego:getComputedStyle(eg).fill, cardBg:getComputedStyle(cd).backgroundColor};}""")
    check("GT stroke resolves green #16a34a", colors["gt"]=="rgb(22, 163, 74)", colors["gt"])
    check("pred stroke resolves accent #1a67eb (light)", colors["pred"]=="rgb(26, 103, 235)", colors["pred"])
    check("ego fill resolves (var --ds-text)", colors["ego"].startswith("rgb"), colors["ego"])
    check("card bg resolves panel white (light)", colors["cardBg"]=="rgb(255, 255, 255)", colors["cardBg"])
    # v0/fwd/maxlat present
    meta_ok = page.evaluate("""()=>[...document.querySelectorAll('.tv-meta')].every(m=>
        /v0/.test(m.innerText)&&/前行/.test(m.innerText)&&/max横向/.test(m.innerText))""")
    check("each card shows v0/前行/max横向", meta_ok)

    # ---- ds shell + theme ----
    shell = page.evaluate("""()=>({header:!!document.querySelector('.ds-header'),
        sections:document.querySelectorAll('.ds-section').length,
        footer:!!document.querySelector('.ds-footer'),
        dsCard:getComputedStyle(document.documentElement).getPropertyValue('--ds-panel').trim(),
        toggleDefined:typeof window.toggleTheme==='function'})""")
    check("ds-header present", shell["header"])
    check("ds-section count==2 (演示+核心方法)", shell["sections"]==2, f"got {shell['sections']}")
    check("ds-footer present", shell["footer"])
    check("demo-shell.css loaded (--ds-panel resolved)", shell["dsCard"]!="", repr(shell["dsCard"]))
    check("theme.js loaded (toggleTheme defined)", shell["toggleDefined"])
    # toggle dark
    before = page.evaluate("()=>document.documentElement.getAttribute('data-theme')||document.documentElement.className")
    page.evaluate("()=>toggleTheme()")
    page.wait_for_timeout(200)
    after = page.evaluate("()=>document.documentElement.getAttribute('data-theme')||document.documentElement.className")
    check("toggleTheme changes theme", before!=after, f"{before!r} -> {after!r}")
    page.evaluate("()=>toggleTheme()")  # restore
    page.wait_for_timeout(150)

    # ---- interactions ----
    page.uncheck("#showPreds"); page.wait_for_timeout(250)
    npaths_off = page.evaluate("()=>[...document.querySelectorAll('.tv-card svg')].map(s=>s.querySelectorAll('path').length)")
    check("showPreds OFF => 1 path/card (GT only)", all(x==1 for x in npaths_off), str(npaths_off))
    pred_off = page.evaluate("()=>[...document.querySelectorAll('.tv-card svg')].some(s=>s.querySelectorAll('path.pred').length>0)")
    check("showPreds OFF => no .pred paths", pred_off==False)
    page.check("#showPreds"); page.wait_for_timeout(200)
    page.check("#equalScale"); page.wait_for_timeout(300)
    check("equalScale ON => still 18 cards", page.evaluate("document.querySelectorAll('.tv-card').length")==18)
    check("equalScale ON => paths finite",
          page.evaluate("()=>[...document.querySelectorAll('.tv-card svg path')].filter(p=>/NaN|Infinity/.test(p.getAttribute('d')||'')).length")==0)
    page.uncheck("#equalScale"); page.wait_for_timeout(150)

    # ---- console ----
    real=[e for e in console_errors if "favicon" not in e.lower()]
    check("zero console errors", len(real)==0, str(real[:3]))
    ctx.close()

    # ---- mobile 390 ----
    ctx3 = browser.new_context(viewport={"width":390,"height":844}, device_scale_factor=2,
                               is_mobile=True, has_touch=True)
    pg3 = ctx3.new_page(); pg3.goto(PAGE, wait_until="commit")
    pg3.wait_for_selector(".tv-card", timeout=15000); pg3.wait_for_timeout(300)
    ov = pg3.evaluate("()=>({sw:document.documentElement.scrollWidth,cw:document.documentElement.clientWidth})")
    check("390px: no horizontal overflow", ov["sw"]<=ov["cw"]+2, str(ov))
    check("390px: 18 cards", pg3.evaluate("document.querySelectorAll('.tv-card').length")==18)
    cols = pg3.evaluate("()=>getComputedStyle(document.querySelector('.tv-grid')).gridTemplateColumns.split(' ').length")
    check("390px: grid single column", cols==1, f"cols={cols}")
    ctx3.close()
    browser.close()

print("\n"+"="*50)
if errors:
    print(f"FAILURES ({len(errors)}):"); [print("  -",e) for e in errors]; sys.exit(1)
print("ALL CHECKS PASSED")
