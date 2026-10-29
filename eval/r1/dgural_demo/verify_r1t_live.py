from playwright.sync_api import sync_playwright
URL='https://ranpin.github.io/alpamayo-edge/r1-trajectory/'
def run(theme):
    with sync_playwright() as pw:
        b=pw.chromium.launch(channel='chrome')
        ctx=b.new_context(viewport={'width':1280,'height':1000},device_scale_factor=2)
        if theme=='dark': ctx.add_init_script("localStorage.setItem('ranpin-theme','dark')")
        pg=ctx.new_page(); errs=[]
        pg.on('console',lambda m: errs.append(m.text) if m.type=='error' else None)
        pg.on('pageerror',lambda e: errs.append(str(e)))
        pg.goto(URL,wait_until='commit')
        pg.wait_for_selector('.r1t-card',timeout=20000)
        pg.wait_for_function("()=>{const i=[...document.querySelectorAll('.r1t-fig')];return i.length===6&&i.every(x=>x.complete&&x.naturalWidth>0);}",timeout=20000)
        r=pg.evaluate("()=>({cards:document.querySelectorAll('.r1t-card').length,rows:document.querySelectorAll('#tbl tbody tr').length,theme:document.documentElement.getAttribute('data-theme'),overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,cot:document.body.textContent.includes('Turn left due to green traffic light')})")
        ce=[e for e in errs if 'favicon' not in e.lower()]
        print(f'[{theme}] cards={r["cards"]} rows={r["rows"]} theme={r["theme"]} overflow={r["overflow"]} cot={r["cot"]} console_err={len(ce)} {ce[:3]}')
        b.close(); return r
l=run('light'); d=run('dark')
ok=(l['cards']==6 and l['rows']==6 and d['cards']==6 and d['rows']==6 and l['overflow']<=1 and d['overflow']<=1 and l['cot'] and d['theme']=='dark')
print('LIVE VERIFY','PASS' if ok else 'FAIL')
