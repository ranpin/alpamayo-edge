#!/usr/bin/env python3
# 重建图1联络表：4 场景(行) × 视频时刻 t=3/8/13/17s(列)，帧取自三线滚动视频。
# 布局与原图一致：2090×1124 = 90px 左标签列(暗底) + 4×4 格(500×281)。
import subprocess, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROLL = '/tmp/r1vid/roll'
OUT = '/Users/ranpin/projects/alpamayo-edge/docs/site/r1-trajectory/assets/roll_contactsheet.jpg'
ROWS = [('0dd3eb7f','直行'), ('ecafce84','左转'), ('1fcf2fef','右转'), ('1f7ef408','变道')]
COLS = [3, 8, 13, 17]
CW, CH, LABELW = 500, 281, 90
W, H = LABELW + 4*CW, 4*CH

try:
    F = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 26)
except Exception:
    F = ImageFont.load_default()

canvas = Image.new('RGB', (W, H), (13, 14, 19))
D = ImageDraw.Draw(canvas)
for r, (cid8, label) in enumerate(ROWS):
    # 左侧场景标签（竖排居中）
    D.text((LABELW/2, r*CH + CH/2), label, font=F, fill=(225,230,240), anchor='mm')
    for c, t in enumerate(COLS):
        tmp = f'/tmp/r1vid/cs_{cid8}_{t}.jpg'
        subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',f'{ROLL}/{cid8}_roll.mp4',
                        '-frames:v','1','-q:v','2',tmp,'-y'], check=True)
        fr = Image.open(tmp).convert('RGB').resize((CW, CH), Image.LANCZOS)
        canvas.paste(fr, (LABELW + c*CW, r*CH))
        os.remove(tmp)
canvas.save(OUT, quality=88)
print('saved', OUT, canvas.size)
