# 驾驶域 OOD 评测（n=184）：三精度正确率

**第一份驾驶域精度证据**（2026-10-09）。此前全部精度证据来自具身机器人操作
（`../benchmark/`，robovqa/robofail），而项目标题是自动驾驶。

## 结果（n=184，8 选 1，随机基线 12.5%）

| 精度 | 设备 | 正确率 | Wilson 95%CI | vs FP16 | McNemar p |
|---|---|---|---|---|---|
| FP16 参考 | goat | **51.6%** | [44.5, 58.7] | — | — |
| INT8 (SmoothQuant) | dog | 47.8% | [40.7, 55.0] | −3.8 pts | 0.143 不显著 |
| INT4 (AWQ) | dog | 50.5% | [43.4, 57.7] | −1.1 pts | 0.815 不显著 |

**结论：「量化无显著正确率损失」从机器人域迁移到驾驶域成立。**
两个掉点都低于 MDE（n=184 ⇒ ≈7 pts，与 n=210 的 6.7 pts 同量级）。

- 552 条输出全部单字母、end-of-sequence、零截断、零解析失败。
- 不一致对：INT8 b/c = 12/5；INT4 b/c = 10/8。
- 分类分项（每类 n=23，CI ±15–20 pts，不可过度解读）：work zones 87/82.6/82.6、
  pedestrians 78.3/78.3/78.3、intersections 65.2/56.5/56.5、emergency 56.5/56.5/47.8、
  special vehicle 43.5/30.4/43.5、animals 30.4/34.8/26.1、debris 30.4/34.8/34.8、
  **cyclists 21.7/8.7/34.8**（FP16/INT8/INT4）。
- INT8 在 cyclists 的 8.7% 是放大 FP16 自己就有的混淆（FP16 也只答对 5/23，
  多数误判为行人密集 10/23、复杂路口 7/23）——任务固有歧义，非量化新失败模式。
- 三方预测选项分布几乎一致（top-3 选项与计数相近）⇒ 无系统性类别偏好漂移。
- 跨基准：机器人域 INT4 掉得更多（−2.4 vs −1.0），驾驶域 INT8 掉得更多
  （−3.8 vs −1.1）⇒ 谁掉得多是噪声，唯一稳定结论是「都不显著」。

## 协议（与 n=210 基准逐项对齐）

| 项 | 取值 |
|---|---|
| 评测集 | `../../data/ood_eval/manifest.json`（闸门 3 产物；PhysicalAI-AV `reasoning/ood_reasoning.parquet`，人工核验标注） |
| 任务 | 8 选 1 场景分类（8 类 × 23，答案字母 A–H 各 23） |
| 帧 | 每 clip 6 帧，长边缩到 448（448×252，与 `bench/frames/robovqa_*.jpg` 同口径；32 对齐由运行时 processor 完成） |
| 生成参数 | `batch_size 1, temperature 0, top_p 1, top_k 1, max_generate_length 12`（与 `bench/orin_input.json` 逐位相同） |
| 引擎 | dog：`engines/int4`、`engines/int8`（n=210 同款，lm_head fp16）+ 共享 `engines/int4/visual`；goat：`engines/fp16` + 从 dog 拷入的 `engines/int4/visual`（1.1 GB，sm_87 + 同版 TRT ⇒ 引擎互通） |
| prompt | n=210 模板 + manifest 题干与 8 选项 + "Answer with ONLY the single letter of the correct option." |

## 复现

```bash
# 1. 本地：缩放帧 + 生成设备输入 JSON（帧不入 git，可确定性重建）
.venv/bin/python eval/accuracy/driving/build_ood_input.py \
    --frames-root /home/vision/chenrunbin/alpamayo-edge/bench/ood_eval/frames \
    --json-out eval/accuracy/driving/ood_input_dog.json
# goat 版同命令，--frames-root 换 /home/nvidia/...，--json-out 换 ood_input_goat.json

# 2. 传输：frames_448/ 打 tar 分别 scp 到两台设备的 bench/ood_eval/frames/
#    goat 另需视觉引擎：在 dog 上
#    scp -r engines/int4/visual nvidia@30.245.40.73:/home/nvidia/chenrunbin/alpamayo-edge/engines/int4/

# 3. 设备侧运行（示例：dog INT4；INT8 换 --engineDir engines/int8/llm；
#    goat FP16 用 engines/fp16/llm + 本机 trtedge 路径与插件）
export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so
/home/vision/TensorRT-Edge-LLM/build_orin/examples/llm/llm_inference \
    --inputFile bench/ood_eval/ood_input_dog.json \
    --engineDir engines/int4/llm \
    --multimodalEngineDir engines/int4/visual \
    --outputFile bench/ood_eval/out_int4.json

# 4. 打分（fp16 为 McNemar 参考）
.venv/bin/python eval/accuracy/driving/score_ood.py \
    --runs fp16=orin_fp16_outputs_n184.json int8=orin_int8_outputs_n184.json \
           int4=orin_int4_outputs_n184.json \
    --reference fp16 --out RESULTS.json
```

## 文件

| 文件 | 说明 |
|---|---|
| `build_ood_input.py` | 缩放帧 + 生成设备输入 JSON |
| `score_ood.py` | 打分（解析/Wilson/McNemar 复用 `../benchmark/score_mc.py`） |
| `ood_input_{dog,goat}.json` | 设备输入（184 requests，仅帧路径前缀不同） |
| `orin_{fp16,int8,int4}_outputs_n184.json` | 设备原始输出（实测记录，不改写） |
| `RESULTS.json` | 打分结果（总体 + 分类 + 分 split + 字母分布） |

## 诚实边界

- 标注是 clip 级"最具安全相关性的场景类别"（事件聚类，非逐帧感知标签）；
  单 clip 可同时含多类事件 ⇒ 绝对分低估感知能力，跨精度对照不受影响。
- train split 占 154/184，无法排除 CR2 训练见过这些 clip（NVIDIA 未公开训练清单）
  ⇒ 不能声称"未见数据上的绝对能力"；三精度同帧同题，掉点结论不受影响。
- n=184 ⇒ MDE ≈ 7 pts；小于该量级的掉点测不出来。
- 绝对分（51.6%）不能与 n=210（76.2%）比：8 选 1 vs 二选一，随机基线 12.5% vs 50%。
