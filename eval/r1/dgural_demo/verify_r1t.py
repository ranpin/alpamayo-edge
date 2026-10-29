from playwright.sync_api import sync_playwright
URL='http://localhost:8799/r1-trajectory/'
def run(theme):
    with sync_playwright() as pw:
        b=pw.chromium.launch(channel='chrome')
        ctx=b.new_context(viewport={'width':1280,'height':1000},device_scale_factor=2)
        if theme=='dark': ctx.add_init_script("localStorage.setItem('ranpin-theme','dark')")
        pg=ctx.new_page(); errs=[]
        pg.on('console',lambda m: errs.append(m.text) if m.type=='error' else None)
        pg.on('pageerror',lambda e: errs.append(str(e)))
        pg.goto(URL,wait_until='commit')
        pg.wait_for_selector('.r1t-card',timeout=15000)
        pg.wait_for_function("()=>{const i=[...document.querySelectorAll('.r1t-fig')];return i.length===6&&i.every(x=>x.complete&&x.naturalWidth>0);}",timeout=15000)
        r=pg.evaluate("""()=>({
          cards:document.querySelectorAll('.r1t-card').length,
          rows:document.querySelectorAll('#tbl tbody tr').length,
          overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
          theme:document.documentElement.getAttribute('data-theme'),
          bg:getComputedStyle(document.body).backgroundColor,
          seqBadges:document.querySelectorAll('.r1t-seq').length,
          seqTexts:[...document.querySelectorAll('.r1t-seq')].map(x=>x.textContent).join(','),
          tblSeq:document.querySelectorAll('.r1t-seqsm').length,
          tblHeader:document.querySelector('#tbl thead th').textContent,
          seqIntro:document.body.textContent.includes('序号 1–6'),
          formula:document.body.textContent.includes('平均位移误差')&&document.body.textContent.includes('终点位移误差'),
          intro:document.body.textContent.includes('位移分解'),
          tag:document.querySelector('.r1t-tag').textContent,
          tagOF:[...document.querySelectorAll('.r1t-tag')].filter(t=>t.scrollWidth>t.clientWidth+1).length
        })""")
        print(f'[{theme}] cards={r["cards"]} rows={r["rows"]} overflow={r["overflow"]} theme={r["theme"]}')
        print(f'   图序号徽章={r["seqBadges"]}个 文本=[{r["seqTexts"]}] 表序号={r["tblSeq"]}个 表头="{r["tblHeader"]}" 序号intro={r["seqIntro"]}')
        print(f'   formula={r["formula"]} intro位移分解={r["intro"]} tag="{r["tag"]}" tag溢出={r["tagOF"]}')
        ce=[e for e in errs if 'favicon' not in e.lower()]
        print(f'   console_err={len(ce)} {ce[:3]}')
        b.close(); return r
l=run('light'); d=run('dark')
ok=(l['cards']==6 and l['rows']==6 and l['overflow']<=1 and d['overflow']<=1
    and l['seqBadges']==6 and l['seqTexts']=='1,2,3,4,5,6' and l['tblSeq']==6 and l['tblHeader']=='示例'
    and l['seqIntro'] and l['formula'] and l['intro'] and l['tagOF']==0 and d['theme']=='dark')
print('\nVERIFY','PASS' if ok else 'FAIL')
