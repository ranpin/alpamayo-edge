# 问题记录 —— 全流程踩坑与解决（统一）

> 本项目实验过程中遇到的**所有**问题、根因与解决方案，集中在此一处。呈现文档（主页面）不含这些细节，只在此归档。
> 每条：**现象 → 根因 → 解决 → 教训**。均为真实发生、可复现。更新：2026-07-22。

## A. 云端量化 / 导出（Kaggle）

### A1. 8B 加载 CUDA OOM
- **现象**：单卡加载 `Cosmos-Reason2-8B` 直接 OOM。
- **根因**：8B fp16≈16GB，单张 T4(16GB) 装不下权重+激活。
- **解决**：`device_map="auto"` 把模型分片到 2×T4；导出前用 `accelerate.remove_hook` + `model.to("cpu")` 把分片模型合并回 CPU 再 export（quantize.py PATCH A/B/C）。
- **教训**：跨多卡分片加载后，导出前必须先合并到单一设备，否则 ONNX 导出会因跨设备张量报错。

### A2. `No space left on device`
- **现象**：int8 检查点(~10GB)+ONNX(~10GB) 撑爆 `/kaggle/working`。
- **根因**：`/kaggle/working` 是固定 20GB loop 设备。
- **解决**：所有重活写 `/tmp`（挂在 7.9TB overlay `/` 上，1.1TB 空闲）；导出后立刻删检查点，只把最终 tgz 放回 `/kaggle/working`；打包放 `try/finally` 确保晚期失败也不丢已产出物。
- **教训**：认清云环境各挂载点的真实容量，重 IO 一律走大盘。

### A3. `secret HTTP 400` —— API 跑读不到 Notebook Secret
- **现象**：`kaggle kernels push` 触发的后台运行读不到 HF_TOKEN Secret。
- **根因**：API 触发的 run 无法读取 UI 配的 Notebook Secret。
- **解决**：推送的私有 kernel 临时副本里注入 token（env 方式）；**repo 副本保持无 token**。
- **教训**：headless 跑云 kernel 时凭据走 env 注入，且注入副本与版本库副本分离。

### A4. HF 门控 401 / `GatedRepoError`
- **现象**：加载 `nvidia/Cosmos-Reason2-8B` 报 401，t≈47s 快速失败（未浪费 GPU）。
- **根因**：本地默认 token（`~/.cache/huggingface/token`）**未获该门控仓库授权**。
- **解决**：改用已接受许可、有 Cosmos 访问权的 token。
- **教训**：门控模型先确认 token 的仓库授权；鉴权失败会很快暴露，把它当作廉价的前置检查。

### A5. Kaggle GPU 抽签给了 P100，torch 不兼容
- **现象**：FP16 参考跑 `generate()` 崩 `CUDA error: no kernel image is available`（`cudaErrorNoKernelImageForDevice`）。
- **根因**：API 默认 `enable_gpu:true` 给的是**单张 Tesla P100（sm_60）**；Kaggle 预装的新版 PyTorch 只支持 `sm_70..sm_120`，不含 sm_60。
- **解决**：kernel-metadata 加 `"machine_shape":"NvidiaTeslaT4"` 锁定 **T4×2（sm_75）**。
- **教训**：API 推送**默认不给 T4×2**，必须显式指定 `machine_shape`；否则 GPU 类型随机、可能撞上不受支持的架构。

### A6. 把两个"权限报错"误判成"凭据过期"，而两套凭据都是好的（2026-07-31）
- **现象**：闸门 3 开始时，① `whoami` 报 `Invalid user token`；② `kaggle kernels pull ranpin/alpamayo-edge-export` 报 **403 Forbidden**。我据此判定"HF token 与 Kaggle key 均已失效"，并给用户写了一整套轮换步骤。**用户反问"我看网站上凭证没失效呀"，一查全错。**
- **根因（两条独立，都不是凭据问题）**：
  1. **Kaggle owner 用错**：正确 owner 是 **`mamihlapinatapai`**，`ranpin` 是 **HF** 的 whoami。Kaggle 对"该 owner 下不存在的资源"返回 **403**，与"key 无效"同形。
  2. **HF 测的是陈旧本地文件**：`~/.cache/huggingface/token` 是 **2025-09** 的，且据本项目自己的记录**从来就没有** Cosmos 门控权限；真正有权限的 `hf_gbJ…` 实测对 PhysicalAI-AV 与 CR2 门控**都是 `True`**。
- **同一轮里我还犯了同类的第三次**：`hf_hub_download('EPFL-DrivingVQA', 'images.zip')` 报 **401 Unauthorized**，我又想归因到凭据——实际是 repo_id 少写了一半（应为 `EPFL-DrivingVQA/DrivingVQA`）。**HF 对不存在的 repo 一律返回 401**（不泄露存在性），Kaggle 返回 403，两家都把"找不到"伪装成"没权限"。
- **解决**：先跑一条**已知可用**的调用做隔离——`kaggle datasets list`（通 ⇒ key 没问题）、`whoami(token=…)` + `HfApi().file_exists(gated_repo)`（后者才真正验门控）。隔离后 5 分钟内定位完毕，**用户无需做任何轮换**。
- **教训**：**权限形状的报错不等于权限问题。** 401/403 只说明"这次请求不被接受"，可能是**凭据**，也可能是**路径/owner/repo 不存在**。定位顺序必须是"先用已知可用的调用证明凭据活着，再怀疑资源"，否则会把成本转嫁给用户（我差点让用户白轮换两套密钥）。
  这与本文件 G3、I1 的教训同源：**不要把新失败挂到手边那个"已知可疑项"上**——`device_layout.md` 与记忆里其实都写着正确的 owner，我没查。

## B. Orin 构建

### B1.（最硬）FMHA 崩溃 `There must be one kernel to implement the MHA`
- **现象**：推理一进 attention 即崩。
- **根因**：Orin 构建 TensorRT-Edge-LLM **漏传 `-DEMBEDDED_TARGET=jetson-orin`**，CMake 走默认分支按 `sm_80;86;89` 编译并 `-DEXCLUDE_SM_87`，把 **sm_87 的 FMHA kernel 整段编译排除**；插件靠 compute_80 PTX 前向 JIT 才勉强加载，但运行时 attention 查表必落空。**非架构缺 cubin、非模型、非量化。**
- **解决**：按官方 Orin 配方全新重编（`-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6`），配置期 FMHA 排除列表翻转为"只留 sm_87"。
- **教训**：交叉编译/嵌入式目标，构建 flag 缺失会**静默**产出能加载但运行必崩的产物；排查要跨层（运行时 kernel 派发表 ↔ 预编译 cubin ↔ 编译宏 ↔ CMake）。深挖全过程见 [orin_build_notes.md](orin_build_notes.md)。

### B2. 插件路径不受 `LD_LIBRARY_PATH` 控制
- **现象**：换了新 build 目录，推理仍加载旧插件（带旧 sm 列表）。
- **根因**：插件由环境变量 `EDGELLM_PLUGIN_PATH` 决定（默认相对路径 `build/libNvInfer_edgellm_plugin.so`），与 `LD_LIBRARY_PATH` 无关。
- **解决**：`export EDGELLM_PLUGIN_PATH=<绝对路径>`；并 `build→build_orin` 软链 + `~/.bashrc` 持久化。
- **教训**：先搞清框架到底用哪条路径加载插件，别想当然靠 `LD_LIBRARY_PATH`。

## C. Orin 推理 / 评测

### C1. `--multimodalEngineDir is required`
- **现象**：纯文本推理也报错要视觉编码器。
- **根因**：VLM 引擎即使只跑文本也要求带多模态引擎目录。
- **解决**：`--multimodalEngineDir <int4/visual>`（int8 复用 int4 的 fp16 视觉编码器）；VLM 路径 CLI 还需 `--outputFile`。
- **教训**：VLM 的文本推理仍需完整多模态引擎在位。

### C2. llm_bench 双加载逼近 29GB
- **现象**：int8 bench 出现非致命 OOM 告警。
- **根因**：引擎被加载两次，撞 29GB 统一内存；TRT 回退到统一内存，数字仍稳定。
- **风险外推**：FP16 引擎(~16GB) 若双加载→32GB 必爆。
- **解决/对策**：FP16 性能测量改用 **llm_inference 单次加载 + `--dumpProfile`**，避开双加载。
- **教训**：大引擎测性能前，先确认工具的加载次数与显存峰值。

## D. 部署 / 呈现

### D1. 提交里混入明文 HF token
- **现象**：`cloud/kaggle_cosmos_only.py` 内嵌明文 token。
- **解决**：改纯 env 读取、无硬编码回退，amend 后再推 → **token 从未进过 GitHub 历史**。
- **教训**：推送前做全历史密钥自查；凭据只走 env。

### D2. 主站外链 404（`/edge-ai-docs/https://…`）
- **现象**：跨子站项目卡片点击 404。
- **根因**：主站 `DocsSection` 无条件给 `file` 拼 `DOCS_BASE` 前缀。
- **解决**：`DocCard` 判断 `file` 以 `http(s)://`/`//` 开头则原样用（PR #24）。
- **教训**：目录清单要原生支持相对条目与绝对外链两种。

## E. 基准评测（Cosmos-Reason1-Benchmark）

### E1. `kaggle kernels output` 下载 15GB 大文件卡死在 0B
- **现象**：拉云端 FP16 ONNX(15GB) 时进程挂起，文件停在 0B。
- **根因**：kaggle CLI 的 `kernels_output` 用 `requests.get(url, stream=True)` 却又 `out.write(resp.content)`——**把整个 15GB 读进内存**再写盘，内存扛不住。
- **解决**：用 API 取签名 URL，`curl -L --http1.1 -C -` **流式+断点续传**（服务器还会 HTTP/2 断流，故 `--http1.1` + 循环重试直到 `tar tzf` 校验通过）。
- **教训**：大文件别用会全量读内存的下载器；流式+可续传+完整性校验。

#### E1 续（2026-07-28，拉 6.46 GiB 的 lm_head 产物时同一个坑又踩了两次）

1. **kaggle CLI 至今未修**：`kaggle kernels output` 拉 6.46 GiB 的 `edge_artifacts_lmhead.tgz`
   产出 **0 字节文件、退出码 0** —— 静默失败，比报错更坏。
2. **`curl -C -` 配"每次重取的签名 URL"会静默损坏文件**：kaggleusercontent 的签名 URL 短时效，
   我写的重试循环每轮重取新 URL 再 `-C -` 续传。结果文件长到 **8.29 GB**（真实归档只有 6.46 GiB），
   多出的 1.3 GB 是从头重下后追加/错位写入的内容，`zlib` 解压报 `invalid block type`。
   ⇒ **`-C -` 只在"同一个稳定 URL + 服务端确实按 Range 响应"时才安全**；换 URL 续传等于赌运气。
3. **正确做法（本轮最终采用）**：先用 `Range: bytes=0-0` 探测——返回 `206` 且
   `Content-Range: bytes 0-0/6938379933` 就同时拿到了**总长度**与**Range 可用性**
   （注意 kaggleusercontent **不支持 HEAD**，`Content-Length` 取不到）。
   然后**按显式字节区间分段拉**（每段 400 MB，**每段单独重取一个新签名 URL**），
   顺序追加；若某段返回 200（服务端忽略 Range）就立刻中止而不是继续写。全程零额外磁盘。

**我自己犯的一个观察错误，值得单独记**：中途我用
`tar tzf x.tgz > manifest.txt && echo "TAR OK"` 校验，看到 25 条目名就宣布"完整性 OK"——
但 **`echo` 那行根本没打印**（tar 返回非零），我把 `grep` 的输出错当成了校验通过，
并据此把一个**还在被写入**的文件 `mv` 进了归档目录、算了 SHA256。
⇒ **教训：校验必须看退出码/成功标记本身，不能看"有输出"**；
`cmd | head` / `cmd | wc -l` 会把退出码吞掉（返回的是管道最后一个命令的状态），别用它们做校验。
**还有：正在下载的文件不要移动**——`mv` 之后下载进程会按原路径重建，既污染归档又白耗带宽。

### E2. 基准仓库结构非 parquet
- **现象**：按 datasets-server 显示的 parquet 列去读，`No objects to concatenate`。
- **根因**：datasets-server 的 parquet 是自动转换的；**真实仓库是 `*_qa_pairs.json`（标注）+ `clips.tar.gz`（视频，需解包）**。
- **解决**：从 JSON 读记录、解包 clips.tar.gz、按相对路径 `clips/xxx.mp4` 定位。
- **教训**：先看仓库真实文件布局（snapshot 后 `ls`），别只信 viewer。

### E3. 视觉引擎 patch 上限 —— 多帧超限
- **现象**：6 帧全分辨率多图推理，`qwenViTRunner: cuSeqlens 6120 exceeds maxHW=4096`，全部请求失败。
- **根因**：Orin 视觉(ViT)引擎构建上限 **总 patch 数 4096**；6 帧 × ~1020 patch = 6120 超限。
- **解决**：帧**长边缩到 448px**；实测（2026-07-26 复核）每帧 patch 数为 **448**（448×252→448×256，28×16）或 **616**（448×358→448×352，28×22），6 帧 = 2688 / 3696 < 4096 ⇒ 通过，但 5:4 那批已用掉 90% 的 patch 预算。**FP16 与 Orin 用同一批缩放帧**保证精度损失可比。（此前本条写的"每帧 ~256 patch"是估算，实测已修正。）
- **教训**：边缘视觉引擎有固定输入上限；多图/多帧要按 patch 预算设计分辨率与帧数。

### E4. FP16 引擎在 Orin build 期 OOM
- **现象**：`llm_build` FP16 引擎，进程消失、引擎目录空、detached 重试 **exit 137（SIGKILL/OOM）**。
- **根因**：FP16 权重 ~16GB + TensorRT builder workspace 峰值 > 29GB 统一内存（lean `--maxKVCacheCapacity` 只降运行时 KV，不降 build 峰值）。
- **实测根因（2026-07-24 用 64GB goat 验证）**：build 峰值 **54.8GB**（>30GB），运行时激活仅 8MB → 纯 **build 内存墙**，非推理。**解法**：在 64GB orin-goat build（成功，peak 54.8GB），引擎同 sm_87+TRT10.7 可拷回 32GB dog 跑（推理 ~17GB）→ 得到 FP16 基线与量化加速比（decode INT4 2.66×）。
- **教训**：边缘设备上大模型 FP16 常连 build 都过不去；量化不只是提速，而是**能否部署**的前提。

---
> 说明：**量化校准集用的是通用新闻文本（非驾驶域）** 属于**方法局限/口径**而非 bug，其影响分析见 [METHODOLOGY.md](METHODOLOGY.md) §4 与主页面「思考讨论」。


### F1. 换工作目录后引擎"反序列化失败"（2026-07-25）

- **现象**：`llm_inference` 报 `failed to deserialize engine: .../llm.engine`，而同一引擎用 `llm_bench` 一直正常。
- **根因**：插件路径。`~/.bashrc` 开头对**非交互式 shell 会提前 `return`**，所以 ssh 里 `source ~/.bashrc` 并不会设上 `EDGELLM_PLUGIN_PATH`；此前 `llm_bench` 能跑纯属侥幸——命令的 cwd 在 `/home/vision/TensorRT-Edge-LLM/` 下，命中了默认的**相对**路径 `build/libNvInfer_edgellm_plugin.so`（`build` 是指向 `build_orin` 的软链）。这次把 cwd 换到 `/home/vision` 后相对路径失效 → 自定义插件未注册 → 引擎反序列化必失败。
- **解决**：非交互式调用一律**显式** `export EDGELLM_PLUGIN_PATH=/home/vision/TensorRT-Edge-LLM/build_orin/libNvInfer_edgellm_plugin.so`。
- **教训**：报错信息（反序列化失败）与真因（插件没加载）相隔很远；"换个目录就崩"要先怀疑相对路径依赖。同理 `nvcc` 也不在非交互式 ssh 的 PATH 里，需 `export PATH=/usr/local/cuda/bin:$PATH`。


### G1. 上游源码漂移让 Kaggle 量化 run 直接失败（2026-07-26）

- **现象**：新推的 `alpamayo-edge-lmhead` kernel 跑到 169 s 就 `ERROR`，日志末尾：
  `AssertionError: PATCH C needle not found — TRTEdge source changed upstream`（`/kaggle/src/script.py:116`）。
- **根因**：脚本用 `git clone --depth 1`（**未 pin commit**）拉 TensorRT-Edge-LLM，再对
  `quantization/quantize.py` 打三处字符串补丁。上游在 `os.makedirs(output_dir, exist_ok=True)` 与
  `with torch.inference_mode(), _skip_resmooth_for_hybrid(` **之间插入了一段注释**，
  于是要求两行相邻的 PATCH C needle 失配。PATCH A/B 仍匹配。
- **解决**：把 PATCH C 的锚点改成**只锚单行** `    os.makedirs(output_dir, exist_ok=True)`（在当前上游文件中仍唯一，已核对 `grep -c` = 1），
  插入位置改为该行之前。同时确认**上游至今仍未自己做 sharded→CPU 合并**（grep `remove_hook_from_module` / `to("cpu")` 均无命中），
  所以 PATCH C 依然必要。修复已回写到 `cloud/kaggle_lmhead_int4.py` **和**原始的 `cloud/kaggle_cosmos_only.py`。
- **教训**：
  1. **这次 assert 起了正作用**——它让失败在 169 s 内明确暴露，而不是"看似打了补丁"后在几十分钟的量化尾声炸掉、或更糟：**静默产出一个错的模型**。字符串补丁必须配唯一性断言。
  2. **锚点要选最小且稳定的片段**。要求多行相邻＝把上游的排版当契约，注释一改就断。
  3. **`clone --depth 1` 不 pin commit 是可复现性缺口**（§4.1 已要求如实标注）。真正的修法是 pin 到 commit；
     当前保持 main 以便拿上游修复，代价就是这类漂移，已在此明确记录。
  4. 失败 run 的 `/kaggle/working` 残留（含整个 TRTEdge 源码树）可通过 `kaggle kernels output` 拉回，
     **正好用来直接读当前上游源码定位漂移**——比盲猜快得多。日志要用 `kaggle kernels logs`（`kernels output` 不含日志）。


### G2. 量化 lm_head 时 T4 CUDA OOM（2026-07-26）

- **现象**：`alpamayo-edge-lmhead` v2 跑到 845 s 失败：
  `tensorrt-edgellm-quantize ... --lm_head_quantization int4_awq` 退出码 1，
  栈底是 `modelopt/torch/quantization/tensor_quant.py:_quantize_impl` →
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.
  GPU 1 has a total capacity of 14.56 GiB of which 1006.81 MiB is free`。
- **根因**：`device_map="auto"` 把 8B(fp16≈16GB) 尽量塞满两张 T4（每张 14.56 GiB，实际用到 13.58 GiB），
  **不给量化算子留临时空间**。而 lm_head 是 `4096 × 151936 = 622 M` 参数，
  fake-quantize 需要约 **1.16 GiB** 的临时张量（与报错数值吻合）——**恰恰是骨干各层都不需要的那种大块临时内存**，
  所以骨干 36 层全部量化成功、只在最后的 lm_head 上炸。
- **解决**：给 PATCH A 的 `from_pretrained` 加
  `max_memory={0: "11GiB", 1: "11GiB", "cpu": "60GiB"}`，
  每张卡留出 ~3.5 GiB 余量给量化临时张量；11+11=22 GiB 仍足以把 16 GB 模型全放在 GPU 上（不牺牲速度）。
- **教训**：
  1. **"模型能装下"≠"能量化"**——量化过程本身需要额外的峰值内存，且**与被量化张量的最大单层尺寸成正比**。
     这与之前 FP16 引擎的 TRT build 内存墙（权重×~3）是同一类错误的不同版本：
     **算显存只算权重，必然低估**。
  2. 词表投影层（vocab × hidden）通常是整个模型**单层最大**的权重，任何"逐层处理"的流程
     （量化 / 导出 / 建引擎）都最可能在它上面爆——优化它收益也最大（见 K1），风险也最集中。
  3. `device_map="auto"` 的默认目标是"尽量塞满"，凡是后续还要做重活的流程都应显式给 `max_memory` 留头寸。


### G3. 在 2×T4 上量化 8B 的 lm_head：五次实验证明是资源限制导致的不可行（2026-07-26）

**目标**：K1 剖析发现 lm_head 未量化、独占 decode 字节 25.6%，预测量化它可得 +23% decode。
TRTEdge 有现成开关 `--lm_head_quantization int4_awq`，看起来只是加个参数。**实际做了 5 次都没成。**

| run | max_memory | 实测 placement | 失败模式 |
|---|---|---|---|
| v2 | 未设 | 全 GPU | `PATCH C needle not found`（上游漂移，见 G1） |
| v3 | 11 GiB/卡 | 全 GPU | **CUDA OOM**：lm_head fake-quantize 要 1.16 GiB，GPU1 仅剩 1006.81 MiB |
| v4 | 9 GiB/卡 | `{0:16, 1:24, cpu:1}` | **CPU offload → AWQ 校准 `illegal memory access`** |
| v5 | 10 GiB/卡 | `{0:16, 1:25}` 全 GPU | **CUDA OOM，剩余显存与 v3 完全相同（1006.81 MiB）** |
| v6 | 11/6 非对称 | `{0:16, 1:16, cpu:9}` | **同 v4 崩溃** |

**根因（两面夹击，在 14.56 GiB×2 上无解）**：
1. 上限够高到把所有模块留在 GPU ⇒ 承载 lm_head 的那张卡剩不下 1.16 GiB 临时空间 ⇒ **OOM**。
2. 上限低到腾出空间 ⇒ 模块外溢到 CPU ⇒ **ModelOpt 的 AWQ 校准与 accelerate 的 offload hook 不兼容**，
   在 `F.linear` 处 illegal memory access ⇒ 直接崩。
3. **关键认识：`max_memory` 只约束"初始权重放置"**，量化状态之后照样把显存吃到同一水位——
   v3(11 GiB) 与 v5(10 GiB) OOM 时**剩余显存一模一样（1006.81 MiB）**，这就是铁证。
   所以"压总量"根本不是有效杠杆。

**出路**：需要**单卡 ≥24 GB**（L4 / A6000 / A100），让 8B + 1.16 GiB 临时张量无需分片即可容纳，
从而同时避开两个失败模式。仓库已有 `cloud/colab_export_quant.ipynb`（Colab 可给 L4/A100），
但 Colab 需要交互式会话。**+23% 至今仍是"预测未验证"，文档中一律如此标注，不得写成已达成。**

#### G3 续（2026-07-27，v7/v8：两个新失败模式，终于定位到真正的机制）

又试了两次（kernel `alpamayo-edge-lmhead-2b8b`，557 s 内双双失败，日志已归档）：

| run | 配置 | 实测 placement | 失败模式 |
|---|---|---|---|
| v7 | **2B**，`device_map={"": 0}` 单卡 | 全在 GPU0 | **CUDA OOM：要分配 4.64 GiB**，GPU0 仅剩 1.94 GiB（进程已占 12.62 GiB） |
| v8 | **8B**，`max_memory={0:"13GiB", 1:"7GiB"}` 非对称 | **`{0: 16, 1: 19, disk: 6}`** | **6 个模块被 offload 到磁盘** → 同 v4/v6 的 `illegal memory access` |

**两个假设都被自己的实测推翻：**
1. 我以为"非对称且总和 20 GiB > 模型 16.34 GiB 就不会外溢"——**错**。accelerate 每卡还要留激活/账务头寸，
   20 GiB 的 `max_memory` 装不下 16.34 GiB 的模型，**而且这次外溢的不是 CPU 而是 `disk`**。
   任何形式的 offload（CPU 或磁盘）都会让 AWQ 校准挂在 `F.linear`。
2. 我以为"2B fp16 才 4.3 GiB，单卡绝对够"——**错**。量化流程的进程峰值是权重的约 **3 倍**（4.3 → 12.62 GiB）。

**但 v7 的报错数字第一次指向了真正的机制。** 4.64 GiB 是什么？
`校准 batch 16 × seq 512 × vocab 151936 × 4 B(fp32) = 4.64 GiB` —— **与报错数值精确吻合**。
⇒ **量化 lm_head 需要为校准批次materialize 完整的 logits 张量，它的大小由 `batch × seq × vocab` 决定，
与权重大小无关。** 这解释了为什么"换更小的模型"救不了：2B 和 8B 的 vocab 都是 151936，
**logits 张量一样大**，缩小模型完全没有减少这一项。
（对比 8B 早前 v3/v5 报的 1.16 GiB = `4096×151936×2 B`，那是 lm_head **权重**本身——是另一个张量。
两个张量都致命，但机制不同。）

**因此正确的出路多了一条、而且不需要更大的显卡**：把 **AWQ 校准 batch 从 16 降到 1–2**
（§4 记录的默认值是 AWQ=16），logits 张量随之缩小 8–16×（4.64 GiB → 0.29 GiB）。
这比"等一张 ≥24 GB 的卡"便宜得多，是下一次该试的第一件事。

#### G3 结局（2026-07-27，v9：**成功**，第 9 次）

按上面的机制改了一处（`PATCH D`，`quantization/quantize.py:737`）：

```diff
-            batch_size = 16 if quantization in (None, "int4_awq") else 1
+            batch_size = 1
```

同时把 8B 的 `max_memory` 从 v8 的 13/7 改回**对称 12/12**（v8 已证 13/7 会把 6 个模块外溢到磁盘；
batch=1 后临时张量小一个数量级，不必再压总量）。

**结果：2B 与 8B 双双成功。**

| run | 配置 | 实测 placement | 结果 |
|---|---|---|---|
| v9-2B | `device_map={"": 0}` + calib batch 1 | 全 GPU0 | ✅ `COSMOS_2b_int4_awq_lmh_OK` |
| v9-8B | `max_memory={0:"12GiB",1:"12GiB"}` + calib batch 1 | **`{0: 16, 1: 25}`** —— 全在 GPU，**无 CPU、无 disk** | ✅ `COSMOS_8b_int4_awq_lmh_OK` |

产物 `edge_artifacts_lmhead.tgz`（6.5 GB，含两个模型的 ONNX），`DONE ok=True`。
**代价：全程 190 分钟**（v7/v8 失败只用了 9 分钟）——batch 16→1 让 512 条样本逐条过，校准慢约 12×。

**教训（这条是整个 G3 最值钱的部分）**：
1. **报错里的数字要做因式分解，而不是只读"又 OOM 了"。** 前 8 次尝试全在调"显存总量"和"模块放到哪张卡"
   这两个维度上打转，而真正的自变量藏在 `4.64 GiB = 16 × 512 × 151936 × 4 B` 里——**是校准批大小**。
   一旦看对了自变量，**第一次尝试就两个模型全过**。
2. **"换个更小的模型"不是万能降级路径**：瓶颈项 `batch × seq × vocab` 与模型规模无关，
   2B 和 8B 的 vocab 都是 151936 ⇒ 8B→2B 对这一项毫无帮助（v7 的 2B 单卡失败正是这么来的）。
   降级之前先确认瓶颈随什么变化。
3. **失败要便宜**：8 次失败每次 9–15 分钟就明确退出，总成本远小于一次盲目的长跑；
   而唯一成功的那次才花 190 分钟。**先便宜地把机制搞对，再付昂贵的算力。**

**⚠️ 口径变更，未验证收益前不得改写结论**：校准 batch 会影响 AWQ 的 per-channel scale 估计，
所以 **batch=1 量出来的模型与已部署的 batch=16 版 INT4 不是同一个量化配置**（样本数仍是 512，覆盖度不降）。
+19~23% 的 decode 增益**仍未验证**——还需要：① 产物拉回本地 ② 传到 Orin 建引擎
③ `llm_bench` 测 decode ④ **同时**在 n=210 基准上测正确率，与 batch=16 版并列报告。
只报速度不报精度，等于拿一个换了配方的模型宣称优化生效。

**教训**：
1. **报错里的数字要拿去做因式分解。** 5 次失败只看到"又 OOM 了"；第 7 次把 4.64 GiB 拆成
   `batch×seq×vocab×4B` 才看见真正的自变量是**校准批大小**，而不是显存总量或模块放置。
2. **`max_memory` 的语义比想象的弱**：它既不约束量化期的峰值（v3/v5 已证），也不保证"总和够就不外溢"（v8 新证），
   而且外溢目标可能是**磁盘**而非 CPU。**必须打印 placement 才知道真相**——这次正是那行打印暴露了 `disk: 6`。
3. **"换个更小的模型"不是万能的降级路径**：当瓶颈项与模型规模无关（这里是 `vocab`），缩小模型毫无帮助。
   降级前先确认瓶颈到底随什么变化。

**教训**：
1. **"只是加一个 CLI 参数"可能触发完全不同的资源画像**。lm_head 是 622 M 参数、
   骨干最大单层（50 M）的 **12 倍**，逐层处理的流程必然在它上面先爆——
   与 G2 是同一条规律的两个实例。
2. **诊断要打印实际状态而不是相信配置**：加了一行 `hf_device_map` 汇总打印，
   才发现 v4/v6 静默外溢到 CPU；否则只会看到"又崩了"。
3. **同样的错误码不代表同样的原因**（v3/v5 是 OOM，v4/v6 是 illegal memory access），
   但**完全相同的剩余显存数值**反而揭示了"cap 无效"这个更深的机制。异常里的数值巧合值得追。
4. 失败要**便宜**：每次 run 都在 15 分钟内失败并留下可读日志，5 次总成本仍小于一次盲目的长跑。

## I1. ViT INT8：5 次尝试全部卡在管道，结构性问题至今未回答（2026-07-28，**未结案**）

**目标**：把视觉编码器量化到 INT8。它占 TTFT **19–25%**（448px）/ **29–33%**（真实分辨率，§1.10），
而 §1.4 已证权重量化对 prefill 无效 ⇒ 这是**唯一一处量化还能打到主要成本的地方**。

**侦察阶段的产出（有效，且都是本地零成本得到）**
1. 我原来判"不可行"的依据（`--visual_quantization` 只给 fp8）**不成立**：`builderUtils.cpp:257` 的
   `createBuilderConfig` **只设 `kMONITOR_MEMORY`、连 `kFP16` 都没设** ⇒ **精度由 ONNX 决定**
   （现有引擎是 fp16 只因 ONNX 权重是 fp16）。
2. **真正的阻塞**：视觉 `model.onnx` 的 `opset_import` 含自定义域 `trt_edgellm`，图里有
   **27 个 `ViTAttentionPlugin`**。ModelOpt 的 **ONNX** PTQ 要靠 ORT 跑图采激活范围，ORT 无此算子
   ⇒ ONNX 侧校准不可行，必须走 **PyTorch 侧**（出路①）。
3. 收益：116 个 `Gemm` 权重合计 **571.5 M = 1090 MiB ≈ 整个 1.10 GiB 引擎** ⇒ INT8 后引擎约减半、
   算力上限翻倍（sm_87 INT8 85 vs FP16 43 TOPS）。**均为解析预测，无任何实测。**

**失败谱系（出路①，5 次，其中 4 次是我自己的问题）**

| run | 失败 | 根因 |
|---|---|---|
| recon | CUDA OOM | 新写脚本**忘带 PATCH A/B/C** ⇒ 重踩 §A1（本项目第一个问题）。且该问题**本不必上云** |
| v1 | needle not found | PATCH C 是**插入**那行的，而写文件后 `_src` 变量仍是原始未打补丁源码 |
| v2 | needle not found | **字符串补丁是子串匹配、不是行匹配**：PATCH A 插入的 20 空格行里也含"4 空格 + 同样文本" ⇒ count==2 |
| v3 | CUDA OOM（申请 32 MiB，GPU0 剩 **4.81 MiB**） | 发生在**加载/`.to()`**；我误归因为"校准 batch 回到 16" |
| v4 | = v3 | **我自己制造的**：编辑脚本的第二处断言失败 ⇒ 文件根本没写，而我把**未修改的脚本**推了上去 |
| v5 | CUDA OOM，**剩余显存与 v3 字节级相同（4.81 MiB）** | ⇒ **v3 的归因是错的**：校准 batch 不控制这一步，加回 PATCH D 是无效动作 |

**本地 diff 的结论**：用 `ast` 提取两个脚本的补丁字面量逐项比对，与**成功的 lm_head v9 完全一致**
（仅 `_D_new` 注释文字不同）⇒ **脚本侧不存在差异**，v9 成功 / 本次同配置失败只能来自**环境**。
最可能是**依赖未 pin**：OOM 堆栈落在 `transformers/modeling_utils.py:3701`，而云端用
`pip install -U 'transformers>=4.51'`（未 pin）+ TRTEdge `git clone --depth 1`（未 pin commit），
两次 run 相隔一天。旁证：v5 的 OOM **未被 PATCH B 的 `try/except` 捕获** ⇒ 来自另一处 `.to()`，
与"加载路径变了"一致。**此为假设，未验证。**

**根因已定位到"哪段代码没跑"（2026-07-29，v6–v9）**
- **决定性观察**：lm_head v9（成功）的日志有 `[edge-patch] load kwargs: {...}`，而**所有 ViT run
  一行都没打印**。⇒ **PATCH A 替换的 `from_pretrained` 调用点根本没被执行到**（assert 通过说明
  补丁写进了文件，但那段代码没跑）⇒ **多卡分片从未生效**，模型按单卡加载 17.5 GB 到 14.56 GB 的 T4
  ⇒ 每次同一点 OOM，而 `max_memory` / calib batch **当然毫无影响**。
- **这同时解答了"为什么量化一个 0.58 B 的 ViT 会失败"**：**失败根本不在 ViT**。失败在
  `quantize llm` **加载整个 8.77 B 模型**的阶段——8 分钟就死，而 batch=1 的 LLM 校准要 190 分钟，
  连 PATCH E 都没执行到。**ViT 的参数量与此完全无关。**
- **transformers 5.0.0 曾是首要嫌疑，已被自己的实验推翻**：v6 指纹抓到 `transformers = 5.0.0`
  （脚本一直 `>=4.51` 无上界），ModelOpt 也 warn "5.0 support is experimental"；但 v9 用
  `--force-reinstall` 降到 **4.57.6**（**子进程验证**）后**同一个 OOM 原地重现** ⇒ **不是根因**。
- **新候选假设（未验证且不自洽）**：安装用 `pip install '.[tools]'`（**非 `-e`**）会把 TRTEdge
  拷进 site-packages，而补丁打在 `/kaggle/working/TRTEdge/` 的源码副本上——与"assert 通过但代码
  没执行"吻合；但 lm_head v9 用同一行安装却成功，故须实测才能定论。

**状态：出路① 未跑通，也未证死** —— 9 次尝试全部卡在**模型加载**阶段，从未触及结构性问题
（QDQ 能否活过视觉导出包装器）。**收益数字自始至终是解析预测、零实测。**

**为什么就此结案**：① §1.10 已定案真实分辨率下系统 **0.3 Hz** vs AV 要 10–30 Hz，ViT INT8 最好
给端到端 **10–15%** ⇒ **不改变任何部署结论**，价值只剩"闭环"；② 9 次尝试边际产出接近零。

**将来重启的第一步只需一行**（几十秒即可证实/推翻上述假设）：
`python -c 'import tensorrt_edgellm as m; print(m.__file__)'` —— 确认 CLI 到底加载哪份源码。
**而这本该是第一次失败时就做的事**，我却先后调了 calib batch、max_memory、依赖版本三个
后来都被证明与该 OOM 无关的旋钮。

**下一步（明确、低成本）**：云端脚本先**打印并 pin** `transformers.__version__` 与 TRTEdge commit，
再重试 —— 这是把"环境漂移"从猜测变成证据的最小动作。若 pin 回旧版即通过，出路① 可行；
若仍 OOM，则差异在 PATCH E 的注入本身，再定位那次 `.to()`。

**教训**
1. **同一个证据模式我犯了两次**：§G3 已写下"完全相同的剩余显存 ⇒ 该旋钮没约束到这一步"，
   而 v3→v5 我又按"调 batch"去猜。**自己写过的教训，下次动手前要先查一遍。**
2. **"命令跑过了" ≠ "目标状态达成了"**（v4）。与本文件另两条同源：把 `grep` 输出当 `tar` 校验通过、
   `py_compile` 通过但 `import` 被吃掉（§E1 续）。⇒ **编辑与推送要么原子化，要么推送前硬校验
   "改动是否真落进文件"**（v5 的推送脚本已加）。
3. **字符串补丁是子串匹配**，"缩进不同所以不会撞"是错的（v2）。
4. **先问"这个问题需要云吗"**：视觉 ONNX 的签名花了 444 s 云端时间，而本地解包 + `onnx.load` 零成本。
5. **依赖不 pin 会让"昨天能跑"变成不可复现**（§G1 的另一种形态）。

## H2. 抬高 maxInputLen 会静默让 INT8 的 prefill 慢 15%（2026-07-27）

- **现象**：把 `maxInputLen` 从默认 1024 提到 4096 重建引擎后，**INT8 在 512/1024 处的 prefill 反而慢了
  15.3% / 15.5%**（157.66 → 181.78 ms、312.42 → 360.86 ms），而 **INT4 完全不受影响**（≤0.5%）。
- **根因**：`cpp/builder/llmBuilder.cpp` 把 TensorRT 优化 profile 写成
  `optCtxShape = {maxBatchSize, maxInputLen/2, hidden}`、`maxCtxShape = {maxBatchSize, maxInputLen, hidden}`
  （变量名自证）⇒ **opt 形状被硬编码为 max 的一半**。抬高上限就把 TRT 的调优点从 512 挪到 2048，
  典型长度反而离最优点更远。
- **归因（两个单变量对照，各只改一个参数）**：

  | 配置 | prefill@512 | prefill@1024 | 归因 |
  |---|---|---|---|
  | `int8` len1024 / b4 / KV4096（原） | 157.66 | 312.42 | 基线 |
  | `int8_ctl_kv8k` len1024 / b4 / **KV8192** | 156.38 | 306.64 | **KV 容量：无影响** |
  | `int8_ctl_b1` len1024 / **b1** / KV4096 | 167.12 | 333.33 | **maxBatchSize 4→1：+6.0% / +6.7%** |
  | `int8_len4096` **len4096** / b1 / KV8192 | 181.78 | 360.86 | 再叠加 **opt 位移：+8.8% / +8.3%** |

  两个原因独立且近似可加。**maxBatchSize 4→1 让 batch-1 自己变慢**这点最反直觉——profile 的
  opt 是 `(maxBatchSize, maxInputLen/2)`，batch=4 的 opt 似乎让 TRT 选到了对 batch=1 也更友好的 kernel。
- **为什么只有 INT8**：INT4 是 W4A16（权重反量化后走 fp16 GEMM，tactic 对形状不敏感）；
  INT8 是 W8A8，跑原生 int8 张量核，tactic/tile 空间更大也更挑形状。**机制是推断，未证明。**
- **处置**：老引擎（`engines/int4`、`engines/int8`）**一律保留不动**，所有已发布数字仍出自它们；
  容量升级版另存 `engines/{int4,int8}_len4096`。INT4 的容量升级**零代价**，INT8 要付 ~15%。
- **教训**：
  1. **"把上限调大点"不是免费的**——工具可能把调优点绑在上限上。改容量参数后**必须在原来的工作点上重测**。
  2. 一次改 3 个参数 = 无法归因。**两个单变量对照（共 8 分钟）就把 15% 拆成了 6% + 9% + 0%**，
     还顺手推翻了我自己"batch 4 会 OOM 所以必须降到 1"的猜测：实测 INT8 build 峰值在
     27.6–29.7 GB 之间、与这几个参数几乎无关（由权重 workspace 主导，不是激活 profile），
     len1024→len4096 只多 0.6 GB。**猜的资源账要用实测推翻。**

## H. 口径与约束的事后复核（2026-07-26）

### H1. n=210 基准最长请求距引擎输入上限只剩 9 个 token
> **2026-07-27 追加的更准确说法**：`maxInputLen = 1024` **是 `llm_build` 的默认值**（`--maxInputLen  Default = 1024`），
> 我们从来没有设过它。所以这不是"侥幸通过"，而是**"没做这个选择"**——上限由工具默认值决定，恰好没被撞破。
> 同批发现 `maxBatchSize` 默认 4（我们全程只跑 batch=1）、`maxKVCacheCapacity` 默认 4096。
> 已按实测重建 4096 版引擎，见 §H2。

- **现象**：为回答"我们到底喂了多大的图、多少 token"而复核时发现，基准里 **40 段视频是 5:4**（帧 448×358 → processor 缩到 448×352），
  **154 image token/帧**，而不是此前一律宣称的 112。实测最长的一题（`request_idx 94`）`Computed Tokens = 1015`，
  引擎 `maxInputLen = 1024` ⇒ **余量 9 个 token（99.1% 占满）**。
- **根因**：`Qwen2VLImageProcessor` 把边长缩到 `patch_size × merge_size = 32` 的整数倍，token 数 = (W/32)×(H/32)；
  "112/帧"只对 16:9 的 448×252 成立。我们此前只用 `robovqa_0`（16:9）做视觉扫描，就把结论推广到了全体。
- **影响评估**：基准日志无截断/超限告警、210 条全部返回、正确率可由 `score_mc.py` 复现 ⇒ **已发布结果不受影响**；
  但这是"侥幸通过"，不是"设计通过"。
- **另一处同向约束**：视觉引擎自身 `max_image_tokens = 1024`（`max_image_tokens_per_image = 512`）——
  6×154 = 924 也逼近它。两个 1024 上限（视觉引擎 token 数 / LLM 输入长度）几乎同时到顶。
- **解决**：文档与图表改为按帧形状分别陈述（图 2 给出预处理链与 prompt 预算条形图）；
  要加帧数或加长题干必须以更大 `maxInputLen` 重建引擎。
- **教训**：**"每帧 N token" 不是模型常数，而是 (帧形状, patch, merge) 的函数**。
  凡是"每 X 多少 Y"的结论，都要用**全量数据的实测分布**验证，而不是拿第一个样本外推；
  并且要把**最坏情况**与硬上限一起算出来（我们直到复核才知道余量只有 0.9%）。

## J. 分辨率 Pareto 的两个运行时边界（2026-10-10）

### J1. INT4@原生 1080p×6 帧：16% 的题输出不可解码的空 token（部分依赖运行时状态）

- **现象**：分辨率 Pareto 的 INT4 1080p 档（12,240 图像 token，`engines/int4_len16384`，
  贪心、cap 12，与 448/720p 档逐位同协议）有 **30/184 题 `finish_reason=max-length` 且
  `output_text` 为空**（448/720p 档全部单字母 EOS、零解析失败）。空 = 生成了 12 个
  解码为空串的 token（疑似特殊 token 重复，输出 JSON 不含 token id，未最终确认）。
  这 30 题按打分协议计为答错 ⇒ 头条 44.0% 被拉低（干净 154 题上 INT4@1080p = 52.6%
  = 720p 档 > 448 档 48.7%）。
- **排查**：
  1. 截断题横跨全部 8 个场景类别（非类别聚集）；帧亮度/对比度统计与正常题同分布（排除夜景/强光）。
  2. 运行日志无任何 warning/error（仅例行的 action runner 缺失提示）。
  3. **确定性检验**：把 30 题单独重跑（cap 64）——**12 题恢复正常单字母作答（8 对 4 错）、
     18 题仍产出不可解码 token** ⇒ 不是内容级确定性：全量跑时这些题前面有 ~97 个请求，
     孤立重跑时只有几个，行为随**运行时累积状态**变化；但 18 题在 cap 64 孤立重跑下仍复现
     ⇒ 至少有一部分是题目级稳定的。
- **定性**：12.4k 多模态上下文 + **INT4(AWQ) 特有**的运行时边界现象（疑似长上下文下
  逐请求累积的数值/状态问题），不是分辨率-精度权衡本身。**关键对照**：INT8(W8A8) 用同一
  视觉引擎、同帧同题跑 1080p 档，**552/552 条全部单字母 EOS、零空输出**（KV32768 死前 140 题
  与 KV16448 全程 184 题均干净）⇒ 缺陷与 INT4 权重格式绑定，不是长上下文的通用问题。
  token id 级取证（需运行时 dump logits/ids）未做，机理未明。
- **影响**：INT4@1080p×6 帧**不具备部署可靠性**（16% 请求空响应）；Pareto 的 1080p 档
  正确率按协议值（含空输出计错）与干净题值（154 题）双口径报告（站点 §1.12）。1080p 档
  部署选 INT8（内存前提见 J2）。
- **教训**：**空结果先怀疑工具**——1080p 档"正确率崩了"的第一解释差点被写成"分辨率过高
  反而伤精度"，实际是运行时产物；成簇的失败模式（连续索引块）是指向工具而非模型的信号。

### J2. INT8@1080p：运行时内存 28.6G 边缘，KV 预留 32768 被 OOM 杀（tegrastats 实测）

- **现象**：INT8 1080p 档首次运行（`engines/int8_len16384`，KV 预留 32768）在 **~140/184 题
  被杀**：无完成记录、整条进程链消失、机器 buff/cache 被剥到 0（OOM 危机痕迹）。
- **证据（tegrastats RAM 轨迹）**：INT4 1080p 全程平稳 25.1G（存活）；INT8 **从第一题起
  28.6G 边缘**（比 INT4 高 3.5G ≈ 引擎体积差 8.9−5.7=3.2G），无泄漏（全程平稳 +140MB），
  最后尖峰 30.4G 超物理上限（29G 模组）⇒ OOM。
- **处置（已解决）**：KV 预留 32768 → 16448（147,456 B/token × 16320 = 省 2.41G，不影响
  计算）重建 `engines/int8_len16384_kv16448`（build 峰值 29,152MB）重试——**184/184 完成**，
  全程 RAM 26.2G/30.7G（tegrastats），E2E 11.26s/请求。KV32768 版引擎保留在
  `engines/int8_len16384` 作为 OOM 实物证据。
- **教训**：**外推要分精度验证**——G3 闸门曾按"INT4 的长度增量 +7.1G"外推 INT8 len16384
  build ~37G 判死，实测 INT8 build 内存对长度几乎不敏感（29.7→29.45G），尝试性构建成功；
  而运行时内存的精度差异（引擎+KV+激活）又是另一回事。**build 内存与运行时内存、INT4 与
  INT8 的长度敏感性，四条曲线各不相同，谁都不能替谁背书。**

### J3. Alpamayo config 类的网络副作用 × transformers to_diff_dict：离线加载必炸（2026-10-12，已解决）

- **现象**：闸门 4 在 goat（不可达 huggingface.co）上 `Alpamayo1_5.from_pretrained(本地权重)`，
  config.json 的 `vlm_name_or_path` 已 patch 为本地 processor 目录，却仍向
  `Qwen/Qwen3-VL-8B-Instruct` 发请求（5 轮重试 ×10s 超时 ×5 个文件 ≈ 14 分钟后 OSError）。
- **根因（完整 traceback 定位）**：不是真 config 加载错——是 transformers 的
  `configuration_utils.py: from_dict → __repr__ → to_json_string → to_diff_dict` 会
  **用无参默认值重构一个 config 实例**来算 diff；而 `ReasoningVLAConfig.__init__` 带网络副作用
  （`_initialize_vlm_config → _build_processor → AutoProcessor.from_pretrained(默认值
  "Qwen/Qwen3-VL-8B-Instruct")`）⇒ 默认值走了网络。**上游设计地雷**：config `__init__`
  里做网络 I/O，与 transformers 的日志/repr 机制天然冲突。
- **排查方法（可复用）**：monkeypatch `_build_processor` 打印 `self.vlm_name_or_path` +
  `HF_HUB_OFFLINE=1` 快速失败——两次离线实验证明真 config 路径正确，从而把嫌疑收敛到
  "第二次 config 构造"，再用 `grep 'File "' 日志` 拿到 to_diff_dict 那一帧。
- **解决**：sed 把 `base_model.py:211` 的默认值改为本地 processor 路径 + 离线环境变量
  （记录于 `scripts/orin/gate4_driver.sh` 头注释）。同批还有两个离线坑：
  ① processor 目录需补 Qwen3-VL-8B 的 config.json（`_initialize_qwenvl3_vlm` 要读
  Qwen3VLConfig）；② TRTEdge 导出的 tokenizer_config.json 把 `extra_special_tokens`
  存成 list，transformers 4.57 要求 dict ⇒ 移除该字段（token 在 tokenizer.json 里本就有）。
- **教训**：**离线环境跑 HF 生态代码，"config 默认值"也是网络依赖面**——patch 实例路径不够，
  类默认值、to_diff_dict 重构、AutoXxx 内部回退都可能触发远程请求。`HF_HUB_OFFLINE=1` 要常开
  （把 14 分钟的重试泥潭变成秒级失败）。另：hf 下载 22G 权重时 **Xet/CAS 后端 401**，
  `HF_HUB_DISABLE_XET=1` 走传统 HTTP 即愈（且快一个量级）；`cmd | tail` 再次吞掉真实退出码
  （下载"成功"实则 3/5 分片缺失——靠逐分片字节数校验抓住）。

## K. Cosmos3-Edge on JetPack 6.2：六道墙与打通路径（2026-10-16，W0–W2 实测）

背景：TRTEdge 0.10.x 官方支持 Cosmos3-Edge，但 Orin JetPack 6.2（cuda 12.6 / TRT 10.7 / sm_87）
在支持矩阵里只是 **Compatible 档**（Official 是 JetPack 7.2 / cuda13 / TRT 11）。实测代价 = 六道墙：

### K1. CuTe DSL 预编译件缺 sm_87+cuda12 ⇒ configure FATAL
上游 `kernelSrcs/build_cutedsl_tarballs.sh` 的 cuda13 targets 含 `aarch64:sm_87`，
cuda12 targets 只有 sm_110/121（**刻意排除，非遗漏**）。初解 `-DENABLE_CUTE_DSL=OFF`（见 K5 反噬）。

### K2. nlohmann 子模块缺失（depth-1 clone）
C++ 侧 `nlohmann/json.hpp` 编译失败；python 侧 pyproject `license-files` 匹配不到文件被
scikit-build-core 拒装。修：`git submodule update --init --recursive`；C++ 侧曾以单头文件 `-I` 绕过（等价）。

### K3. XQA JIT NVRTC 缺头 ⇒ llm_build 插件创建失败
0.10.x 的 decode XQA kernel 改为构建期嵌入头 + 运行期 NVRTC JIT；嵌入清单按 CUDA 13 写，
cuda12.6 的 `cuda_fp16.h → vector_types.h → crt/host_defines.h` 链缺失。补清单后又暴露
`vector_functions.h → cuda_runtime_api.h` 巨链 ⇒ **根治：`pluginJitCompiler.cpp` 注入
`-I$CUDA_HOME/include`**（NVRTC 从磁盘解析系统头，一处补丁覆盖全部 JIT kernel）。

### K4. ViTAttentionPlugin 无 legacy 后端 ⇒ visual_build 失败（head_dim 72）
`ViTFMHABackend` 枚举只剩 CuTe DSL 两项（legacy 被移除）；`USE_TRT_NATIVE_ATTN=1` 导出的
`TRT_Attention` 算子 parser 支持 **TRT 11.0 起**（TensorRT CHANGELOG 实证，ragged 11.1）。
解：分解注意力导出补丁（基础算子块对角 ragged attention，`scripts/orin/apply_decomposed_attn_patch.py`）。

### K5. prefill FMHA 无 legacy 后端 ⇒ 引擎能建、推理必挂
`ContextFMHABackend` 枚举同样只剩 CuTe DSL；`ENABLE_CUTE_DSL=OFF` 下 LLM 引擎构建成功
（XQA JIT 属 decode 路径）但运行期 prefill 报 "selected prefill kernel is unavailable (D=128, SM=87)"。
**根治（W1 突破）：用上游脚本自生成 `cutedsl_aarch64_sm_87_cuda12.tar.gz`**——
`CUDA_CTK_VERSION=12.6 CUTE_DSL_TARGETS=aarch64:sm_87 CUTE_DSL_KERNELS=fmha bash kernelSrcs/build_cutedsl_tarballs.sh`
（需 GPU 在场做 AOT 导出；nvidia-cutlass-dsl[cu12]==4.7.0 + cupy-cuda12x 均有 aarch64 wheel）
→ `-DENABLE_CUTE_DSL=fmha` 重配置重建 → **冒烟全绿**（单帧驾驶场景描述连贯、EOS）。

### K6. policy und_prefill/gen 导出 ONNX 标准 Attention/RotaryEmbedding ⇒ TRT 10.7 parser 不认（W2）
`--task policy` 的 und_prefill（理解 prefill）与 gen（扩散动作专家）把注意力/RoPE 经
`trt::attention_onnx`/`trt::rope_onnx` custom op lower 成 **ONNX 标准 `Attention`（opset 23+）
/`RotaryEmbedding` 算子**（und：28 Attention+84 RotaryEmbedding；gen：一批），这两个算子只有
**TRT≥11 parser** 认识；Orin TRT 10.7 建引擎报 "Cannot find plugin: Attention/RotaryEmbedding,
version: 1, namespace:."。与 K4 同源（都是 TRT-11 算子墙），但 K4 是 visual 的 `TRT_Attention`、
这里是 policy 的 ONNX 标准算子；且 K5 的 CuTe DSL 自生成对 policy 图无解（gen 走 custom-op
lowering，不是 FMHA backend 选择）。
**解（W2 突破）**：把 K4 的分解思路推广到 policy + RoPE——`scripts/orin/apply_policy_decomposed_patch.py`
在 `ops.py` 加 `decomposed_attention`（GQA 折批 grouped matmul + arange 下三角 causal 掩码，
覆盖 und 的 is_causal=True 自注意力与 gen 的 is_causal=False 对 cat([und,gen]) 的交叉注意力）
+ `decomposed_rope`（rotate_half；`attention_pos_id` 是恒等 gather（cosmos3PolicyRunner.cpp:257
"identity gather indices"）⇒ 逐 token 直接对齐，但仍用 torch.gather 保留 position_ids 防被 DCE
导致 runtime 找不到 binding），env `USE_DECOMPOSED_POLICY=1` 门控、两个 modeling 文件 import
换包装器（additive，不动已工作的 reasoning 路径）。分解实现先经 `test_policy_decomposed.py`
数值对拍（attention vs `F.scaled_dot_product_attention` err~1e-7、rope vs LLaMA 全维公式 err=0）
再导出 ⇒ und_prefill/gen 引擎建成、`cosmos3_policy_inference` 冒烟 exit=0
（action [1,16,10] 全 finite、5154ms/round）。

**教训**：① Compatible 档的代价是具体、有限、可逐个击破的——六道墙各有工程解；
② **"上游不给预编译件"≠"不能生成"**：构建脚本支持显式 target 覆盖，sm_87+cuda12 生成一次通过；
③ **分解导出补丁是对「TRT-11-only 算子墙」的通用武器**：K4（visual 的 TRT_Attention）与
K6（policy 的 ONNX Attention/RotaryEmbedding）同源同解——用基础算子分解，任何 TRT 版本可解析；
K4 在 K5（自生成 CuTe DSL）解决后成冗余保险，K6 则是 policy 链唯一通路（gen 无 FMHA backend 可选）；
④ 值得向 TRTEdge 上游提 issue：cuda12 targets 补 aarch64:sm_87 + 嵌入头清单补 vector_* 链 +
policy 导出对 TRT<11 提供分解 fallback。

## L. Cosmos3-Edge policy 动作链：部署打通，但 turnkey 路径输出由 seed 主导、conditioning 弱耦合（2026-10-19 W3 / 2026-10-26 W4 实测）

W2 把 policy 四组件引擎建通、冒烟 exit=0（K6）。W3 要把它做成「真实 DROID 数据 → 动作」demo，
先只读侦察 NVIDIA 参考实现（`github.com/NVIDIA/cosmos` + `cosmos-framework`）搞清动作的后处理与
归一化，再用真实数据实测。**W3 初判：turnkey 单帧路径「帧无关」。W4 复测修正：不是「帧无关」，
而是帧/文本/CFG 三个 conditioning 轴一致地只有 1–2%、输出由 diffusion seed 主导（52%）——整条
conditioning 通路弱耦合，根因未定（我们的分解补丁接线 vs 模型属性），见 L4。**

### L1. 动作表示 / 归一化 / 后处理（cosmos-framework 权威定义，已查清）
- **表示**：Edge 用 `midtrain` 10 维 = [eef_pos(3) + rot6d(6) + gripper(1)]、**相对位姿**
  （`action_policy_server_robolab.py:607-612`；action_space 默认 joint_pos=8 维，Edge=midtrain=10 维）。
- **归一化**：quantile(q01/q99)→[−1,1]，stats 在 `cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json`
  （10 维物理量：dim0-2 位置 delta≈±0.015 米、dim3-8 rot6d、dim9 gripper[0,1]）；反归一化
  `0.5(a+1)(q99−q01)+q01`（`action_normalization.py`）。
- **后处理链**：`denormalize_action` → `pose_rel_to_abs`（需当前 state 做参照）。
- **state 注入定论**：Edge 是 **vision+language-only**（`use_state_rows = 1 if use_state else 0`，
  Edge gen 图=16 行、无 +1 state 行 ⇒ use_state=False）。TRTEdge PolicyRunner 把 action_latent 全填
  高斯噪声、action_noisy_mask 全 1、只注入 frame-0 图像 latent + und K/V，无本体状态——**设计如此，非 bug**。

### L2. 实测：单帧反归一化 + 跨帧复测（reverify 确认）
- 真实 DROID 帧（`lerobot/droid_100` ep0 "Put the marker in the pot"）+ 指令喂 goat 引擎，输出反归一化后：
  位置 delta 小（~mm）、有净位移但沿途抖动、rot6d 合法（|v|=1）近恒等、gripper 反归一化超 [0,1]
  （invert+denorm 约定存疑）；4/16/40 步扩散输出范围不变（**非欠去噪**，`DEFAULT_NUM_INFERENCE_STEPS=4`）。
- **跨帧复测（换独立变量=帧，f20/60/100/140）**：输出跨帧近乎相同（mean|fA−fB|≈0.0076–0.0116、max≈0.06）
  ⇒ **单帧 turnkey 路径「帧无关」**：帧内能出连贯位移，但完全不随输入帧变化。vision-only 单帧
  conditioning 不足以产生帧敏感动作。

### L3. 为何「预测 vs GT」对照不成立（表示动物园）
社区 `lerobot/droid_100` GT 是 **7 维笛卡尔 euler** [x,y,z,roll,pitch,yaw,gripper]、lerobot 自归一化 [−1,1]；
Edge 输出是 **10 维 midtrain rot6d**、NVIDIA quantile 归一化；cosmos cookbook 的 droid 样例又是
**8 维关节空间** [joint_pos(7)+gripper(1)]。三套表示 + 三套归一化 + 时间分辨率不同（policy 5fps vs GT 15fps）
⇒ 直接对照是四重错配，「对不上」是必然，非引擎错。

### L4. W4 复测：不是「帧无关」，是「全 conditioning 弱耦合、输出由 seed 主导」（2026-10-26）

W3 只换了「帧」一个变量，且用的是二进制默认 `--guidance 1.0`。读 `cosmos3_policy_inference.cpp`
发现 **1.0 = CFG 关闭**，源码注明「**3.0 matches the reference**」——W3 是在偏离参考配置下测的。
W4 补测三个 W3 没碰的轴（官方 UMI 帧，`engines/cosmos3_policy_droid`，steps 4）：

| 变化的轴 | mean\|diff\| | 占 mean\|action\|=0.7387 |
|---|---|---|
| 确定性底（同参数三次重跑，**逐位相同**） | 0.000000 | 0% |
| guidance 1.0→5.0 | 0.0070 | 0.94% |
| 跨帧（W3） | 0.0094 | 1.27% |
| 跨 prompt（4 条语义不同指令） | 0.0129 | 1.74% |
| **SEED（对照组）** | **0.383** | **51.9%** |
| *参照：chunk 内相邻两步* | *0.389* | *52.7%* |

**SEED / conditioning = 39×**。「去噪不够」已排除：cross-prompt 敏感性 steps 4→16→40 =
0.0129→0.0322→0.0287（**平台化**），steps16≈steps40（扩散已收敛），seed 效应仍 0.45。

⇒ **修正 W3 定性**：不是「vision-only 单帧 conditioning 弱」，而是**帧/文本/CFG 三个轴一致地只有
1–2%**——整条 conditioning 通路没有有效耦合，输出基本是「seed 去噪出的通用轨迹」。三轴**均匀**衰减
更像系统性 attenuation（scale/mask/接线），**「我们的 gen cross-attention 分解补丁 attenuate 了
conditioning」从猜想升级为领先嫌疑**（reasoner 侧同塔同技术却明确输入敏感：n=184 得 42.4%、3.4× chance）。
**根因未定**：补丁接线 vs 模型属性，决定性判据（PyTorch 参考对照）需 Orin 装 cosmos-framework +
GPU torch（实测 `export_venv` 是 CPU-only torch 且无 cosmos 模块），非快速对照。

**附带发现（action space 配置）**：跑 TRTEdge 自带 `robolab/selftest.py` 失败于
`cannot reshape array of size 160 into shape (32,8)`——参考客户端 `Cosmos3JointposClient` 硬编码
**joint_pos 8 维 / chunk 32**，我们引擎是 **midtrain 10 维 / chunk 16**（与 cosmos-framework 的
DROID 定义一致）。`config.json` 无 action_space/dim/chunk 键（只有 `max_action_dim=64`、
`num_embodiment_domains=32`）⇒ action space 由导出侧选、非 checkpoint 决定。selftest 不适用我们的
配置，**既未证明引擎坏、也未解释 seed 主导**，但留下一个未验证变量（换 joint_pos/chunk32 重导出）。

复现：`eval/cosmos3/w4_conditioning_sensitivity.py`（证据 `eval/cosmos3/w4_cfg/` 16 个 action.json
已入库，一键重算 T0 确定性/T1 guidance/T2 prompt/T3 seed/T4-T5 steps 全部差异；帧用官方 UMI 样例，
无需 droid 数据；取数命令见脚本头注）。

**教训**：① turnkey 工具（`cosmos3_policy_inference`）是**简化路径**——vision+language-only、单帧、无 state
锚点，与 cosmos-framework 参考 server（state-conditioned、多帧历史）输入契约不同；拿 turnkey 输出直接比
参考 GT 前，必须先核对**表示 / 归一化 / state / 时间**四者是否对齐。② **反直觉的单次实测要换独立变量复测
再定性，且要换「结论所依赖的那个」变量**：W3 换帧得「帧无关」，W4 换 conditioning 强度（guidance）与
文本才发现真相是「全 conditioning 弱耦合、seed 主导」——只换帧会漏掉它（见
[[reverify-surprising-measurements]] 纪律）。③ **先查工具默认值是否等于参考配置**：W3/W2 都用了
`--guidance` 默认 1.0（CFG 关），而参考是 3.0——偏离参考配置的「能力边界」结论要先排除配置伪影。
④ 测出边界本身有价值，但**「非引擎 bug」这类排除性断言要有对照证据**：W4 之前说「非引擎 bug」只凭
算子数值对拍，没验导出图接线，W4 后降级为「根因未定」。复现：`eval/cosmos3/w3_frame_sensitivity.py`
（证据 `eval/cosmos3/w3_actions/`）与 `w4_conditioning_sensitivity.py`（证据 `eval/cosmos3/w4_cfg/`）。

## M. Alpamayo-R1 action 专家：K6 同源墙，分解补丁第三次打通（2026-10-26，Phase R 最小验证 ✅）

**背景**：闸门 4 曾基于 TRTEdge **0.9.0** 判定「TRTEdge 不支持 Alpamayo」而关闭 AV 轨迹路线。
2026-10-19 静态侦察**推翻该旧判**——`export.py:162 _ACTION_MODEL_TYPES = frozenset(["alpamayo_r1"])`，
0.10.1 **原生支持** `alpamayo_r1`，且官方指南 `docs/source/user_guide/examples/vla/alpamayo.md` 齐备、
`action_build`/`action_inference` 二进制已编好、`compute_minade.py` 带公开基线 **0.82113 m / 644 clips**。
⇒ 路线重开，用户拍板「按最小验证走」：**只导 + 只建 action 单引擎**，不先碰 10B llm build。

### M1. 墙：action 专家撞 K6 同源墙

`models/alpamayo/modeling_alpamayo_action.py` 的 `ActionAttention` 用三个 `trt::` custom op，
lower 成 **ONNX 标准算子**，只有 TRT≥11 parser 认识（Orin JetPack 6.2 = TRT 10.7）：

| custom op | lower 成 | 与既有墙的关系 |
|---|---|---|
| `trt::attention_onnx` | `Attention` | **= K6 同一个** |
| `trt::rope_onnx` | `RotaryEmbedding` | **= K6 同一个** |
| `trt::kv_cache_update_onnx` | `TensorScatter` | **新**（cosmos3 policy 没有这个） |

三个 op 在 `models/ops.py` 里都是 **stub（`return x.clone()`）**——真实计算由 TRT 侧承担，
所以 PyTorch 端数值对拍无法覆盖它们，必须自己写分解实现。

### M2. 解法：分解补丁（`scripts/orin/apply_alpamayo_action_decomposed_patch.py`）

同 K4/K6 思路，用基础 ONNX 算子实现，任何 TRT 版本可解析。门控 **`USE_DECOMPOSED_ALPAMAYO=1`**
（独立于 cosmos3 的 `USE_DECOMPOSED_POLICY`，两套补丁互不干扰）。改两处：`models/ops.py` 追加三个
分解实现，`modeling_alpamayo_action.py:187` 的局部 import 换成走包装器（别名回原名，调用点不动）。

**🔴 不能直接复用 cosmos3 policy 补丁——三个陷阱都只在 B>1 暴露，而导出 dummy 恰好是 `B=1`**
（`get_onnx_export_args` 里 `B = 1`），照抄会在建 `--maxBatchSize 6` 引擎时才炸：

| # | 陷阱（源码实测） | 处理 |
|---|---|---|
| 1 | RoPE 的 cos/sin **只有 S 行、跨 batch 共享**（forward 里 `rope_cos = rope_rotary_cos_sin[0:1].reshape(-1,half)`） | cosmos3 版 `cos.reshape(b,s,half)` 在 B>1 **元素不足**；改为按 `position_ids` 行索引后广播。position_ids 须保留作图输入防 DCE |
| 2 | `attn_mask` 是 **4D `[B,1,S,Sk]`**，而 GQA 折叠后 scores 是 **5D `[B,Hkv,rep,S,Sk]`** | 不升维直接相加只在 B==1 碰巧广播正确 ⇒ `unsqueeze(2)` 升到 `[B,1,1,S,Sk]` |
| 3 | `kv_cache_update` 的 `cache_indices` 是**运行期张量输入**（dummy=`torch.full((B,),3027)`、dynamic_shapes 标 `{0:batch}`） | trace 期值未知 ⇒ `torch.cat` 静态切片不可用；改 `scatter`（索引仅 `[B,H,S,D]`，比 gather+mask 的 `[B,H,C,D]` 省 ~C/S 倍） |

**导出前数值对拍**（`test_alpamayo_action_decomposed.py`，**16/16 全过**）：`kv_cache_update` vs 直接切片赋值
**err=0.000e+00**（逐位精确，含「写入区外逐位不变」护栏，覆盖 start=3027/0/C−S × B=1/2）；
`attention` vs `F.scaled_dot_product_attention` **err~1e-7**；`rope` vs LLaMA 全维公式 **err=0**；dispatch 3 项。
其中陷阱 2 的护栏**实证了陷阱为真**：B=2 时 naive 写法直接抛 `RuntimeError`（不是算错，是根本广播不了）。

### M3. 实测结果：通了

| 阶段 | 实测 |
|---|---|
| 权重下载 | 21 G / 5 shard，hf-mirror + `HF_HUB_DISABLE_XET=1`，~50 MB/s 约 7 分钟 |
| **export**（`--skip-llm --skip-visual`） | 峰值 RSS **10.5 G**、最低 MemAvailable **37 G**、约 5 分钟；`model.onnx` 6,314,629 B（图）+ `model.onnx.data` **4,561,240,064 B**（外置权重） |
| **算子普查** | `Attention=0 RotaryEmbedding=0 TensorScatter=0 KVCacheUpdate=0`；`ScatterElements=72`、`Softmax=36`、`MatMul=325`（共 3218 节点） |
| **action_build**（`--maxBatchSize 6`） | Engine generation **89.9 s**、峰值 **CPU 12088 MiB / GPU 4349 MiB**、**0 error** ⇒ `action.engine` **4,565,059,244 B** |

**自洽性交叉验证**：`action/config.json` 给出 `num_hidden_layers: 36` ⇒ `Softmax=36`（每层 1 个注意力）、
`ScatterElements=72 = 36×2`（每层 k/v 各一次 scatter）。**数字精确对上**，证明分解逐层落进了图里，
不是「碰巧没有墙算子」。

⇒ **结论：Orin sm_87 / JetPack 6.2 / TRT 10.7 上，Alpamayo-R1 的 turnkey action 路径跑得起来。**
资源开销远低于预期（对比闸门 4 当年「8B 级加载峰值 54.8 G」，action 专家构建峰值仅 12.1 G）。
**分解注意力补丁的普适性第三次得到验证**（K4 cosmos3 visual → K6 cosmos3 policy → M2 alpamayo action）。

### M4. 全链已通（2026-10-27）：三引擎建成 + 推理成功 + 轨迹积分

llm / visual 走 TRTEdge 自己的插件（算子普查实测墙算子全 0，无需新补丁）；只有 action 专家需分解补丁。
三引擎在 goat 建成（FP16-only）：

| 引擎 | 大小 | 构建耗时 | 峰值内存 | 注意力实现 |
|---|---|---|---|---|
| `llm.engine` | 14.15 GiB | 172.3 s | **CPU 31.8 G** / GPU 14.5 G | `AttentionPlugin` ×36 |
| `visual.engine` | 1.08 GiB | 112.2 s | CPU 5.4 G / GPU 1.4 G | `ViTAttentionPlugin` ×27 |
| `action.engine` | 4.25 GiB | 89.9 s | CPU 12.1 G / GPU 4.3 G | **分解补丁**（§M2） |

我此前外推「llm build 峰值 ~43 G、可能 OOM」**没有发生**——实测 31.8 G；但 MemAvailable 峰值时确实掉到
**8.5 G**，离危险不远，`--maxBatchSize 1` 的降压决定是对的。推理期 GPU 统一内存 **19919 MiB**、单次 ~6 s。

**推理成功且输出物理合理**（16 张 gate4 多相机帧 + 官方指南真实 ego 历史 ~9 m/s，`estimate_v0`=8.641 m/s）：
`output_text` = *"Keep lane to continue driving since no critical agent needs attention."*；
`output_trajectory [64,2]` 反归一化后 accel mean **−0.712 m/s²**、κ mean −5.7e−5 /m，
积分得 6.4 s 走 **45.1 m**、速度 8.64→4.08 m/s、横向 **+0.12 m（向左）**、航向变化 <0.3°，
全部在 action_space 界内、全 finite、无发散。**横向符号与文本「nudge left」自洽**。

### M5. seed 敏感性对照（W4 教训的 R1 版，`eval/r1/r1_seed_sensitivity.py`）

W4 的教训直接适用：**测 conditioning 必须带一个非 conditioning 轴作对照**。R1 的对照轴是 `--noiseSeed`。
**关键设计：必须 `temperature=0`**——VLM 文本默认 temp=1.0 采样，生成的 CoT token 会进 action 专家
conditioning，不停掉文本采样就量不出干净的 seed 轴。七次推理实测：

| 轴 | mean\|diff\| | 占 mean\|traj\|=0.5527 |
|---|---|---|
| 确定性底（同参两次，**逐位相同**） | 0.00000000 | 0% |
| **SEED**（0/1/2/3 vs 参考 seed5，4 对） | **0.2711** | **49.05%** |
| **EGO 历史**（真实 ~9 m/s vs 全零，seed 固定） | 0.0706 | **12.78%** |
| *参照：轨迹内相邻两步* | *0.0269* | *4.86%* |

**SEED / EGO-HIST = 3.84×**，对比 **W4 Cosmos3 policy 的 39.3×** ⇒ R1 条件耦合比 Cosmos3 policy
**强约一个数量级**，但 seed 仍主导轨迹头。**文本头则干净确定**：temp=0 下 6 次跑输出完全同一句，
换 ego 历史整句翻转（"Keep lane…" ↔ "Nudge to the left…"）⇒ **VLM 文本头强条件驱动且可复现；
扩散轨迹头 seed 主导**。

⚠️ **解读口径（别误判成缺陷）**：`compute_minade.py` 是 `resps[:N]` 采 N 条再 `ade.min()`——
**minADE 的 "min" 本就跨 seed 取最优**，故 seed 间差异大是扩散式多模态轨迹预测器的设计属性，
官方指标已吸收，不等于缺陷。但**做 demo 必须固定 `--noiseSeed`** 才可复现。

### M6. minADE 定量评测（2026-10-27，chunk 3116，18 clips × 6 seeds）

**数据**：`nvidia/PhysicalAI-Autonomous-Vehicles`（`gated: auto` + `license: other`）。🔴 **官方那 644 clips
拿不到**——`compute_minade.py:19` 写 *"Reference: **chefu** PyTorch baseline minADE = 0.82113 m on 644 clips"*，
`chefu` 是 NVIDIA 内部人名、644 条是内部评测集；两个公开 split（`alpamayo_aligned` = train 276,299 /
test 28,317 / alpasim 1,536；`default` = 153,625 / 90,928 / 61,599）里**都没有 644 这个规模**。
⇒ 改用 `alpamayo_aligned` 的 **test** split（数据集 README 确认它 *"follows the train/test partition of
NVIDIA's Alpamayo models"*）——**分布对，但 clip 集不同，不可声称同口径复现**。

**选片**：chunk 3116 是纯 test chunk（100 clips 全 test+valid）。按轨迹形态**分层抽 18 条**
（每 (形态×速度) 格取 |横向| 最大的 3 条）：straight 9 / turn 6 / gentle 3；fast 9 / mid 5 / slow 4；
v0 跨 0.0–26.6 m/s、maxlat 跨 0.00–49.10 m，含静止与起步加速两个边界样本。

**GT 构造**：从 `labels/egomotion/` per-clip parquet（`timestamp` µs + 四元数 + xyz，~100 Hz）按
**t0=5.1 s**（官方口径，见 `gate4_forward.py`）取 16 点历史 + 64 点未来（均 10 Hz），变换到 t0 的 ego 系
`p_ego = R0ᵀ(p−p0)`。🔴 **关键修正**：`p0` 必须在 t0 **精确插值**而非取最近采样点——否则快车（18 m/s）
会因 ~3 ms 采样偏移产生 **5.7 cm 残差**（断言 `|he[-1]|<0.05` 当场抓到）；修正后残差 **0.00000 m**。

**帧抽取**：mp4 是 1920×1080 / 30 fps / hevc / 20.13 s。用随附 `<clip>.<cam>.timestamps.parquet`
（604 帧权威时间戳）定位 t0−0.3/−0.2/−0.1/0 的最近帧 = **索引 144/147/150/153**（相机与 egomotion
共享 epoch，恒定偏移 −10.85 ms）。相机顺序按 `camera_indices [0,1,2,6]`：cross_left → front_wide →
cross_right → front_tele（camera-major）。goat 无 ffmpeg ⇒ 另建 `frames_venv` 装 `imageio-ffmpeg`
（自带 aarch64 静态 ffmpeg v7.0.2），**不碰 export_venv**（它连 pip 都没有）。

**结果**（官方 `compute_minade.py --num_traj_samples 6`）：

| 指标 | 我们（n=18） | 官方参考 | 差异 |
|---|---|---|---|
| **minADE6@6.4s** | **0.80401 m** | **0.82113 m**（chefu PyTorch, 644 clips） | **−2.1%** |
| minADE6@3s | 0.17460 m | 0.254 m（paper Table 6, AR1 **0.5B**+Route） | −31%（**不同模型**，仅参考） |

@6.4s：mean 0.80401 / median 0.60216 / sd 0.6458 / **95% CI [0.5057, 1.1024]** / min 0.0949 / max 2.5043。
**官方基线 0.82113 落在我们 95% CI 内** ⇒ 统计上一致。分层（@6.4s）：gentle(n=3) 0.3617 /
straight(n=9) 0.7316 / **turn(n=6) 1.1338**；fast(n=9,v0≈15.5) 0.8976 / mid(n=5) 0.8495 / slow(n=4) 0.5366。

**物理合理性交叉验证**（「数字可信」的关键旁证）：ADE 与速度/距离**正相关**（corr(v0,@6.4s)=**+0.43**、
corr(fwd,@6.4s)=**+0.61**、corr(maxlat,@6.4s)=+0.27）⇒ 越快越远绝对误差越大，符合运动学预期、不是随机数；
**相对误差中位数仅 1.91%**（ADE@6.4s ÷ 纵向位移）；最差 3 条全是高速/大位移（`d1a20a67` v0=26.6 /
fwd=167.2 m → 2.504 m），最好 3 条全是近静止（→ 0.095 m）。

### M7. 尚未验证 / 诚实边界

- ⚠️ **mRoPE 黄旗（未解决，但间接证据已大幅加强）**：`action/config.json` 写 `rope_type: mrope`、
  `mrope_section: [24,20,20]`、`mrope_interleaved: true`，而分解补丁实现的是 **rotate_half** 约定。
  判断大概率无碍的依据：① `rope_onnx` 接收的是**已算好的 cos/sin**，mRoPE 分段逻辑在上游；
  ② 导出 dummy 的 `attention_pos_id = arange(n_diff)` 是 **1D** 位置，而 mRoPE 需 (t,h,w) 三维
  ⇒ 疑为从 VLM 继承的残留字段。**间接证据（M6 大幅加强）**：18 个不同场景的轨迹 minADE 与官方基线
  一致、ADE 与速度/距离正相关、相对误差中位数仅 1.91%——**若 RoPE 约定错，输出会是垃圾而不可能对上 GT**。
  但**仍无直接对照**（对拍只是「我的分解 vs rotate_half 公式」的自洽检查，不是「rotate_half vs TRT
  `RotaryEmbedding` 真实语义」的检查）。
- ⬜ **官方 644-clip 同口径评测做不到**：那 644 条是 NVIDIA 内部集，公开数据集无此清单。已在
  `alpamayo_aligned` test split 分层抽 18 clips 做了量级一致的评测（M6），但 **n=18 太小**
  （95% CI 半宽 0.30 m）⇒ 只能说「与基线一致」，**不能说「复现了 0.82113」更不能说「优于基线」**；
  −2.1% 的点估计差异远小于 CI 半宽，无统计意义。扩到更大样本仍受数据可得性限制。
- ⬜ **图像轴的 conditioning 敏感性未单独定量测**：M5 的 seed 对照只用了 gate4 单一场景。不过 M6 的
  minADE 跨 **18 个不同场景**且与基线一致，已**间接证明图像 conditioning 有效**（否则不同场景会给出
  同样的错误轨迹、ADE 会爆）。「固定 seed、只换图像」的定量敏感性对照仍未做。
- ⚠️ **两个输入坑**（已入 `eval/r1/README.md` §五）：① `alpamayo_gate4/meta.npz` 的 ego 历史是
  **退化占位数据**（`ego_history_xyz` 全零、`ego_history_rot` 全单位阵，闸门 4 只验加载不验数据），
  直接喂会得 v0=0、积分出「静止车刹车到倒车」的荒谬轨迹，须改用官方指南真实轨迹；
  ② 导出必须设 `HF_ENDPOINT=https://hf-mirror.com`（export 要取 Qwen3-VL config，goat 直连 HF 不通，
  漏设会卡网络重试且 RSS 停在 767 MB 不动）。另 `hf` CLI 只在 `export_venv/bin/`，不在系统 PATH。
- ⚠️ 输入图像来自 **gated + license:other** 数据集，**只在机上跑，不入库、不上站**。

**教训**：① **旧闸门结论要随工具版本复核**——闸门 4 的「不支持」是 0.9.0 时代的事实，0.10.1 已原生支持；
把版本绑定的否定结论当成永久结论，会白白关掉一条主线（本次差点因此错过 AV 轨迹路线）。
② **复用补丁前先核对调用形态，别只看算子名相同**——三个陷阱全是「同名算子、不同张量约定」，
且都被 `B=1` 的导出 dummy 掩盖。③ **对拍要带「反例护栏」**：不只验「我的实现 == 参考」，
还要验「错误实现 != 参考」，否则测不出自己究竟覆盖了没有。④ 长构建/导出**先挂内存哨兵再发射**
（只观察不自动 kill，见 [[observe-before-stopping-jobs]]）：本次探针顺带把导出跑完了，
峰值 10.5 G 远低于预估，避免了在共享机上盲目发射的风险。

复现：`scripts/orin/r1_action_minverify.sh`（下载→补丁→对拍→只导 action→算子普查→只建 action 引擎，
各步判据即上表实测数字）。
