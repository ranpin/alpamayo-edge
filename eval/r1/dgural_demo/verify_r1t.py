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
            # 联络表默认折叠（display:none 内 img 不触发懒加载）——先展开再滚页
            pg.evaluate("()=>{const d=document.querySelector('.r1t-sheetdet');if(d)d.open=true;}")
            # 滚全页触发懒加载：静态图在折叠线下、超出 Chrome lazy 阈值，不滚不加载
            pg.evaluate("""async()=>{let h=document.body.scrollHeight;for(let y=0,i=0;y<h&&i<60;y+=400,i++){window.scrollTo(0,y);await new Promise(r=>setTimeout(r,60));h=document.body.scrollHeight;}window.scrollTo(0,0);}""")
            pg.wait_for_function("()=>{const i=[...document.querySelectorAll('img.r1t-fig')];const s=document.querySelector('img.r1t-sheet');const p=[...document.querySelectorAll('.r1t-posterwrap img')];return i.length===6&&p.length===4&&[...i,...p].every(x=>x.complete&&x.naturalWidth>0)&&s&&s.complete&&s.naturalWidth>0;}", timeout=60000)
            r = pg.evaluate("""async ()=>{
              const wraps=[...document.querySelectorAll('.r1t-posterwrap')];
              const assets=[];
              for(const w of wraps){assets.push(new URL(w.dataset.src,location.href).href);assets.push(new URL(w.querySelector('img').src,location.href).href);}
              const sheetImg=document.querySelector('img.r1t-sheet'); if(sheetImg) assets.push(new URL(sheetImg.src,location.href).href);
              const codes={};
              for(const a of assets){try{const resp=await fetch(a,{method:'HEAD'});codes[a.split('/').pop()]=resp.status;}catch(e){codes[a.split('/').pop()]='ERR';}}
              const rollTbl=[...document.querySelectorAll('table.ds-table')].find(t=>{const c=t.querySelector('caption');return c&&/^表 2/.test(c.textContent);});
              const vcards=[...document.querySelectorAll('.r1t-card')].filter(c=>c.querySelector('.r1t-posterwrap'));
              return {
                imgCards:document.querySelectorAll('.r1t-card img.r1t-fig').length,
                sheet:document.querySelectorAll('img.r1t-sheet').length,
                sheetDet:document.querySelectorAll('.r1t-sheetdet').length,
                cotLists:document.querySelectorAll('.r1t-cotlist').length,
                cotRows:document.querySelectorAll('.r1t-cotlist div:not(.r1t-cotlabel)').length,
                cotWarns:document.querySelectorAll('.r1t-cotwarn').length,
                posterBeforeGrid:(document.querySelector('.r1t-posterwrap').compareDocumentPosition(document.getElementById('gridL'))&4)>0,
                posters:wraps.length,
                videosInitial:document.querySelectorAll('video').length,
                posterOk:wraps.every(w=>w.dataset.src.includes('.mp4')&&w.querySelector('img').src.includes('roll_poster')&&w.querySelector('.r1t-play')&&w.getAttribute('role')==='button'&&w.tabIndex===0),
                details:document.querySelectorAll('details.r1t-details').length,
                detailsOpen:document.querySelectorAll('details.r1t-details[open]').length,
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
                vidTops:vcards.map(c=>Math.round(c.getBoundingClientRect().top)),
                // 静态图分组：上排 3 例左转 / 下排 3 例右转，各成一行
                gridL:document.querySelectorAll('#gridL .r1t-card').length,
                gridR:document.querySelectorAll('#gridR .r1t-card').length,
                sTops:[...document.querySelectorAll('#gridL .r1t-card,#gridR .r1t-card')].map(c=>Math.round(c.getBoundingClientRect().top)),
                sTurns:[...document.querySelectorAll('#gridL .r1t-card,#gridR .r1t-card')].map(c=>c.querySelector('.r1t-figcap').textContent.includes('左转')?'L':'R').join(''),
                tblTurns:[...document.querySelectorAll('#tbl tbody tr')].slice(0,6).map(tr=>tr.children[1].textContent.includes('左转')?'L':'R').join(''),
                tblIds:[...document.querySelectorAll('#tbl tbody tr')].slice(0,6).map(tr=>tr.querySelector('code').textContent).join(','),
                tblBest:document.querySelectorAll('#tbl td.r1t-best').length,
                tblWorst:document.querySelectorAll('#tbl td.r1t-worst').length,
                avgRow:[...document.querySelectorAll('#tbl tbody tr.r1t-avgrow td')].map(td=>td.textContent).join('|'),
                legendBestBg:getComputedStyle(document.querySelector('.r1t-tlegend .r1t-best')).backgroundColor,
                legendWorstBg:getComputedStyle(document.querySelector('.r1t-tlegend .r1t-worst')).backgroundColor,
                cellBestBg:getComputedStyle(document.querySelector('#tbl td.r1t-best')).backgroundColor,
                cellWorstBg:getComputedStyle(document.querySelector('#tbl td.r1t-worst')).backgroundColor,
                cellBestFg:getComputedStyle(document.querySelector('#tbl td.r1t-best')).color,
                cellWorstFg:getComputedStyle(document.querySelector('#tbl td.r1t-worst')).color,
                // 正文整合：结论速览卡 + 小标题 + 要点列表 + 可折叠口径块
                findings:document.querySelectorAll('.r1t-fcard').length,
                fnums:[...document.querySelectorAll('.r1t-fnum')].map(n=>n.textContent.trim()),
                fnumOF:[...document.querySelectorAll('.r1t-fnum')].filter(n=>n.scrollWidth>n.clientWidth+1).length,
                subheads:[...document.querySelectorAll('.r1t-subh')].map(s=>s.textContent.trim()),
                calDets:document.querySelectorAll('details.r1t-caldet').length,
                calDetOpen:document.querySelectorAll('details.r1t-caldet[open]').length,
                points:document.querySelectorAll('ul.r1t-points').length
              };}""")
            # 点击播放交互：首个海报卡 → 换成 video 并自动播放（远程 mp4 需缓冲，有界等待）
            pg.click('.r1t-posterwrap')
            try:
                pg.wait_for_function("()=>{const v=document.querySelector('video');return v&&!v.paused&&v.currentTime>0;}", timeout=15000)
            except Exception:
                pass
            r2 = pg.evaluate("""()=>{const v=document.querySelector('video');return {
              videos:document.querySelectorAll('video').length,
              postersLeft:document.querySelectorAll('.r1t-posterwrap').length,
              srcOk:v?v.currentSrc.includes('roll_straight_0dd3eb7f.mp4'):false,
              playing:v?(!v.paused&&v.currentTime>0):false};}""")
            # 折叠详情交互
            pg.click('details.r1t-details summary')
            pg.wait_for_timeout(300)
            r3 = pg.evaluate("()=>({open:document.querySelectorAll('details.r1t-details[open]').length,cot:!!document.querySelector('details[open] .r1t-cotlist, details[open] .r1t-cot')})")
            # 口径折叠块交互：展开首个 .r1t-caldet，确认列表项真的可见（非空壳）
            pg.click('details.r1t-caldet summary')
            pg.wait_for_timeout(200)
            r4 = pg.evaluate("()=>{const d=document.querySelector('details.r1t-caldet');const li=d.querySelector('li');return {open:d.open,items:d.querySelectorAll('li').length,liVisible:li?li.getBoundingClientRect().height>0:false};}")
            r.update(click=r2, details_click=r3, caldet_click=r4)
            ce = [e for e in errs if 'favicon' not in e.lower()]
            r['console_err'] = ce
            print(f"[{theme}] imgs={r['imgCards']} posters={r['posters']} videosInit={r['videosInitial']} posterOk={r['posterOk']} details={r['details']}/{r['detailsOpen']} sheet={r['sheet']} cotLists={r['cotLists']} cotRows={r['cotRows']} cotWarns={r['cotWarns']} posterBeforeGrid={r['posterBeforeGrid']} rows={r['rows']} rollRows={r['rollRows']} overflow={r['overflow']} theme={r['theme']} bg={r['bg']}")
            print(f"   click={r['click']} details_click={r['details_click']}")
            print(f"   captions={r['captions']}")
            print(f"   figcaps={r['figcaps']}")
            print(f"   assets={r['assetCodes']}")
            print(f"   seq=[{r['seqTexts']}] 表序号={r['tblSeq']} 表头={r['tblHeader']} seqIntro={r['seqIntro']} rollIntro={r['rollIntro']} methodRoll={r['methodRoll']} formula={r['formula']} intro={r['intro']} tagOF={r['tagOF']} vidTops={r['vidTops']} console_err={len(ce)} {ce[:3]}")
            print(f"   静态分组 gridL={r['gridL']} gridR={r['gridR']} sTurns={r['sTurns']} sTops={r['sTops']} 表转向={r['tblTurns']} 表序={r['tblIds']}")
            print(f"   表1配色 best={r['tblBest']} worst={r['tblWorst']} 平均行={r['avgRow']}")
            print(f"   底色 图例best={r['legendBestBg']} 单元格best={r['cellBestBg']} 图例worst={r['legendWorstBg']} 单元格worst={r['cellWorstBg']} 字色={r['cellBestFg']}/{r['cellWorstFg']}")
            print(f"   正文整合 findings={r['findings']} fnums={r['fnums']} fnumOF={r['fnumOF']} calDets={r['calDets']}/{r['calDetOpen']} points={r['points']} caldet_click={r['caldet_click']}")
            print(f"   小标题={r['subheads']}")
            b.close(); return r
    l = run('light'); d = run('dark')
    exp_fig = ['图 %d' % i for i in range(1, 13)]
    ok = (l['imgCards'] == 6 and l['posters'] == 4 and l['videosInitial'] == 0 and l['posterOk'] and l['posterBeforeGrid']
          and l['details'] == 10 and l['detailsOpen'] == 0
          and l['click']['videos'] == 1 and l['click']['postersLeft'] == 3 and l['click']['srcOk'] and l['click']['playing']
          and l['details_click']['open'] == 1 and l['details_click']['cot']
          and l['rows'] == 7 and l['rollRows'] == 12
          and l['sheet'] == 1 and l['sheetDet'] == 1 and l['cotLists'] == 4 and l['cotRows'] == 12 and l['cotWarns'] == 2
          and len(l['assetCodes']) == 9 and all(v == 200 for v in l['assetCodes'].values())
          and l['captions'] == ['表 1', '表 2', '表 3'] and l['figcaps'] == exp_fig
          and l['seqBadges'] == 6 and l['seqTexts'] == '1,2,3,4,5,6' and l['tblSeq'] == 6 and l['tblHeader'] == '示例'
          and l['seqIntro'] and l['rollIntro'] and l['methodRoll'] and l['formula'] and l['intro'] and l['tagOF'] == 0
          and l['overflow'] <= 1 and d['overflow'] <= 1 and d['theme'] == 'dark'
          and l['bg'] == 'rgb(248, 250, 252)' and d['bg'] == 'rgb(15, 23, 42)'
          and len(l['console_err']) == 0 and len(d['console_err']) == 0
          and len(l['vidTops']) == 4 and l['vidTops'][0] == l['vidTops'][1] and l['vidTops'][2] == l['vidTops'][3] and l['vidTops'][2] > l['vidTops'][0]
          # 静态图按转向分组（上排 3 左转 / 下排 3 右转，各成一行）+ 表 1 同序 + 最优/最差底色
          and l['gridL'] == 3 and l['gridR'] == 3 and l['sTurns'] == 'LLLRRR' and l['tblTurns'] == 'LLLRRR'
          and l['tblIds'] == 'ecafce84,46003675,058bf960,76f4a312,1f7ef408,fa83bcb8'
          and len(l['sTops']) == 6 and l['sTops'][0] == l['sTops'][1] == l['sTops'][2]
          and l['sTops'][3] == l['sTops'][4] == l['sTops'][5] and l['sTops'][3] > l['sTops'][0]
          and l['tblBest'] == 15 and l['tblWorst'] == 15 and d['tblBest'] == 15 and d['tblWorst'] == 15
          and l['avgRow'] == '平均（6 片段）|—|—|0.88|3.10|9.27|0.93|3.41|10.33|0.86|2.39|8.31'
          and l['legendBestBg'] == l['cellBestBg'] and l['legendWorstBg'] == l['cellWorstBg']
          and d['legendBestBg'] == d['cellBestBg'] and d['legendWorstBg'] == d['cellWorstBg']
          and l['cellBestBg'] != l['cellWorstBg'] and l['cellBestFg'] != l['cellWorstFg']
          and 'rgba(0, 0, 0, 0)' not in (l['cellBestBg'], l['cellWorstBg'], d['cellBestBg'], d['cellWorstBg'])
          and d['cellBestBg'] != l['cellBestBg'] and d['cellWorstBg'] != l['cellWorstBg']
          # 正文整合：结论速览卡 + 小标题 + 要点列表 + 可折叠口径块
          and l['findings'] == 4 and l['fnumOF'] == 0 and d['fnumOF'] == 0
          and l['calDets'] == 2 and l['calDetOpen'] == 0 and l['points'] == 6
          and l['caldet_click']['open'] and l['caldet_click']['items'] >= 3 and l['caldet_click']['liVisible']
          and len(l['subheads']) == 6 and any('四条结论' in s for s in l['subheads'])
          and any('表 1' in s for s in l['subheads']) and any('表 3' in s for s in l['subheads'])
          and any('滚动口径（表 2）' in s for s in l['subheads']))
    print('\nVERIFY', 'PASS' if ok else 'FAIL', url)
    return ok

if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else LOCAL) else 1)
