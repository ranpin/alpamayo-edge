import sys
from playwright.sync_api import sync_playwright

LOCAL = 'http://localhost:8799/r1-trajectory/'

def main(url):
    def run(theme):
        with sync_playwright() as pw:
            b = pw.chromium.launch(channel='chrome')
            ctx = b.new_context(viewport={'width': 1280, 'height': 1000}, device_scale_factor=2)
            if theme == 'dark':
                ctx.add_init_script("localStorage.setItem('ranpin-theme','dark')")
            pg = ctx.new_page(); errs = []
            pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(url, wait_until='commit')
            pg.wait_for_selector('.r1t-card', timeout=20000)
            pg.wait_for_function("()=>{const i=[...document.querySelectorAll('img.r1t-fig')];return i.length===6&&i.every(x=>x.complete&&x.naturalWidth>0);}", timeout=20000)
            r = pg.evaluate("""async ()=>{
              const vids=[...document.querySelectorAll('video.r1t-fig')];
              const assets=[];
              for(const v of vids){assets.push(v.querySelector('source').src);assets.push(new URL(v.poster,location.href).href);}
              const codes={};
              for(const a of assets){try{const resp=await fetch(a,{method:'HEAD'});codes[a.split('/').pop()]=resp.status;}catch(e){codes[a.split('/').pop()]='ERR';}}
              const rollTbl=[...document.querySelectorAll('table.ds-table')].find(t=>{const c=t.querySelector('caption');return c&&/^表 2/.test(c.textContent);});
              const vcards=[...document.querySelectorAll('.r1t-card')].filter(c=>c.querySelector('video'));
              return {
                imgCards:document.querySelectorAll('.r1t-card img.r1t-fig').length,
                videos:vids.length,
                vidOk:vids.every(v=>v.controls&&v.preload==='none'&&v.poster.includes('roll_poster')&&v.querySelector('source').src.includes('.mp4')),
                assetCodes:codes,
                rows:document.querySelectorAll('#tbl tbody tr').length,
                rollRows:rollTbl?rollTbl.querySelectorAll('tbody tr').length:-1,
                captions:[...document.querySelectorAll('caption')].map(c=>(c.textContent.match(/表 \\d+/)||['?'])[0]),
                figcaps:[...document.querySelectorAll('.ds-figcap')].map(f=>(f.textContent.match(/图 \\d+/)||['?'])[0]),
                seqBadges:document.querySelectorAll('.r1t-seq').length,
                seqTexts:[...document.querySelectorAll('.r1t-seq')].map(x=>x.textContent).join(','),
                tblSeq:document.querySelectorAll('.r1t-seqsm').length,
                tblHeader:document.querySelector('#tbl thead th').textContent,
                seqIntro:document.body.textContent.includes('序号 1–6'),
                rollIntro:document.body.textContent.includes('滚动口径（表 2）')&&document.body.textContent.includes('边前进边预测'),
                methodRoll:document.body.textContent.includes('滚动视频版'),
                formula:document.body.textContent.includes('平均位移误差')&&document.body.textContent.includes('终点位移误差'),
                intro:document.body.textContent.includes('位移分解'),
                overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
                theme:document.documentElement.getAttribute('data-theme'),
                bg:getComputedStyle(document.body).backgroundColor,
                tagOF:[...document.querySelectorAll('.r1t-tag,.r1t-metrics')].filter(t=>t.scrollWidth>t.clientWidth+1).length,
                vidTops:vcards.map(c=>Math.round(c.getBoundingClientRect().top))
              };}""")
            ce = [e for e in errs if 'favicon' not in e.lower()]
            r['console_err'] = ce
            print(f"[{theme}] imgs={r['imgCards']} videos={r['videos']} vidOk={r['vidOk']} rows={r['rows']} rollRows={r['rollRows']} overflow={r['overflow']} theme={r['theme']} bg={r['bg']}")
            print(f"   captions={r['captions']}")
            print(f"   figcaps={r['figcaps']}")
            print(f"   assets={r['assetCodes']}")
            print(f"   seq=[{r['seqTexts']}] 表序号={r['tblSeq']} 表头={r['tblHeader']} seqIntro={r['seqIntro']} rollIntro={r['rollIntro']} methodRoll={r['methodRoll']} formula={r['formula']} intro={r['intro']} tagOF={r['tagOF']} vidTops={r['vidTops']} console_err={len(ce)} {ce[:3]}")
            b.close(); return r
    l = run('light'); d = run('dark')
    exp_fig = ['图 %d' % i for i in range(1, 12)]
    ok = (l['imgCards'] == 6 and l['videos'] == 4 and l['vidOk'] and l['rows'] == 6 and l['rollRows'] == 12
          and len(l['assetCodes']) == 8 and all(v == 200 for v in l['assetCodes'].values())
          and l['captions'] == ['表 1', '表 2', '表 3'] and l['figcaps'] == exp_fig
          and l['seqBadges'] == 6 and l['seqTexts'] == '1,2,3,4,5,6' and l['tblSeq'] == 6 and l['tblHeader'] == '示例'
          and l['seqIntro'] and l['rollIntro'] and l['methodRoll'] and l['formula'] and l['intro'] and l['tagOF'] == 0
          and l['overflow'] <= 1 and d['overflow'] <= 1 and d['theme'] == 'dark'
          and l['bg'] == 'rgb(248, 250, 252)' and d['bg'] == 'rgb(15, 23, 42)'
          and len(l['console_err']) == 0 and len(d['console_err']) == 0
          and len(l['vidTops']) == 4 and l['vidTops'][0] == l['vidTops'][1] and l['vidTops'][2] == l['vidTops'][3] and l['vidTops'][2] > l['vidTops'][0])
    print('\nVERIFY', 'PASS' if ok else 'FAIL', url)
    return ok

if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else LOCAL) else 1)
