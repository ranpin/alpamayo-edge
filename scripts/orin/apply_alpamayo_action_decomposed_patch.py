#!/usr/bin/env python3
"""Patch TRTEdge 0.10.1 alpamayo action-expert exporter: decomposed attn/RoPE/KV.

Why (R1 最小验证, 2026-10-26):
  Alpamayo-R1 的 action 专家（``models/alpamayo/modeling_alpamayo_action.py``）
  把三个算子 lower 成 ONNX 标准节点，只有 TRT>=11 parser 认识；Orin JetPack 6.2
  是 TRT 10.7，建 action 引擎会报 "Cannot find plugin" —— 与 K6（cosmos3 policy）
  同一道版本墙。解法同 K4/K6：用基础 ONNX 算子分解掉，任何 TRT 版本可解析。
    trt::rope_onnx            -> RotaryEmbedding
    trt::kv_cache_update_onnx -> TensorScatter        (本补丁新增分解)
    trt::attention_onnx       -> Attention

  与 cosmos3 policy 补丁（apply_policy_decomposed_patch.py）的三点不同，均已实测钉死：
  1) RoPE 的 cos/sin 只有 S 行、跨 batch 共享（forward 里 ``rope_cos[0:1].reshape(-1,half)``），
     故不能复用 cosmos3 的 ``cos.reshape(b,s,half)``（B>1 会元素不足），需按
     position_ids 行索引并广播。position_ids=``arange(n_diff)`` 仍是恒等 gather，
     但保留它作图输入防 DCE（否则 runtime 找不到 attention_pos_id binding 硬失败）。
  2) attention 的 attn_mask 是 4D ``[B,1,S,Sk]``，而 GQA 折叠后 scores 是 5D
     ``[B,Hkv,rep,S,Sk]``；直接相加只在 B==1 碰巧广播正确，B>1 会错 ⇒ 需升一维。
  3) kv_cache_update 的 ``cache_indices`` 是运行期张量输入（导出 dummy=3027、dim0
     动态），值在 trace 期未知 ⇒ 不能用 torch.cat 静态切片，改用 scatter 动态写入。

  门控 env：``USE_DECOMPOSED_ALPAMAYO=1``（独立于 cosmos3 的 USE_DECOMPOSED_POLICY）。

Usage (on device, inside trtedge-0.10.1/):
    python3 apply_alpamayo_action_decomposed_patch.py          # 打补丁 (幂等)
    python3 apply_alpamayo_action_decomposed_patch.py --diff   # 只打印将做的修改
Then re-export action with:
    USE_DECOMPOSED_ALPAMAYO=1 tensorrt-edgellm-export <ckpt> <out> \
        --skip-llm --skip-visual --max-kv-cache-capacity 4096
"""
import sys

OPS = "tensorrt_edgellm/models/ops.py"
MODELING = "tensorrt_edgellm/models/alpamayo/modeling_alpamayo_action.py"

# 追加到 ops.py 末尾。
HELPERS = '''

# ---------------------------------------------------------------------------
# R1 alpamayo action-expert decomposed attention / RoPE / KV-cache update
# (JetPack 6.2 / TRT 10.7 compat; gated by USE_DECOMPOSED_ALPAMAYO=1)
# ---------------------------------------------------------------------------
# trt::rope_onnx / trt::kv_cache_update_onnx / trt::attention_onnx 会 lower 成
# ONNX 标准 RotaryEmbedding / TensorScatter / Attention，只有 TRT>=11 parser 能
# 解析；Orin JetPack 6.2 (TRT 10.7) 建 action 引擎报 "Cannot find plugin"。
# 导出时设 USE_DECOMPOSED_ALPAMAYO=1，把三个算子改走基础 ONNX 算子分解。


def _use_decomposed_alpamayo() -> bool:
    import os
    return os.environ.get("USE_DECOMPOSED_ALPAMAYO") == "1"


def decomposed_rope_shared(x, cos, sin, position_ids):
    """rotate_half RoPE，cos/sin 仅 S 行、跨 batch 共享。x [B,H,S,D]。

    alpamayo forward 把 rope_cos 取成 ``rope_rotary_cos_sin[0:1].reshape(-1,half)``，
    故 cos/sin 是 [S,half]（与 B 无关）。用 position_ids 行索引（恒等 gather，但
    保留 position_ids 作图输入防 DCE），再广播到 [B,1,S,half]。
    """
    b, h, s, d = x.shape
    half = d // 2
    cos_rows = cos.reshape(-1, half)
    sin_rows = sin.reshape(-1, half)
    pos = position_ids.to(torch.int64).reshape(b, s)
    cos_g = cos_rows[pos].reshape(b, s, half).unsqueeze(1)
    sin_g = sin_rows[pos].reshape(b, s, half).unsqueeze(1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos_g - x2 * sin_g, x2 * cos_g + x1 * sin_g),
                     dim=-1)


def decomposed_attention_mask4d(query, key, value, attn_mask, is_causal, scale):
    """GQA attention，attn_mask 为 4D [B,1,S,Sk]（alpamayo 的加性掩码）。

    GQA 把 rep 折进 batch 维后 scores 是 5D [B,Hkv,rep,S,Sk]；4D 掩码需 unsqueeze
    到 [B,1,1,S,Sk] 才能对任意 B 正确广播（直接相加只在 B==1 碰巧对）。
    """
    b, hq, s, d = query.shape
    hkv = key.shape[1]
    sk = key.shape[2]
    rep = hq // hkv
    q = query.view(b, hkv, rep, s, d)
    k = key.unsqueeze(2)
    v = value.unsqueeze(2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        iq = torch.arange(s, device=query.device).unsqueeze(1)
        ik = torch.arange(sk, device=query.device).unsqueeze(0)
        keep = ik <= (iq + (sk - s))
        neg = torch.finfo(scores.dtype).min
        scores = torch.where(keep.unsqueeze(0).unsqueeze(0), scores,
                             torch.full_like(scores, neg))
    if attn_mask is not None:
        m = attn_mask
        if m.dim() == scores.dim() - 1:
            m = m.unsqueeze(2)
        scores = scores + m
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.reshape(b, hq, s, d)


def decomposed_kv_cache_update(cache, new_kv, cache_indices):
    """把 new_kv 写进 cache 的「运行期」起始位，纯基础算子。cache [B,H,C,D]。

    cache_indices 是图输入（导出 dummy=3027、dim0 动态），trace 期值未知，故
    torch.cat 静态切片不可用；scatter 用算出的索引张量写入（导出成 ScatterElements）
    且保留 cache_indices 在图里。索引仅 [B,H,S,D]（S=n_diffusion_tokens），
    比 gather+mask 的 [B,H,C,D] 省 ~C/S 倍。
    """
    b, h, c, d = cache.shape
    s = new_kv.shape[2]
    lo = cache_indices.to(torch.int64).reshape(b, 1, 1, 1)
    off = torch.arange(s, device=cache.device,
                       dtype=torch.int64).reshape(1, 1, s, 1)
    idx = (lo + off).expand(b, h, s, d)
    return cache.scatter(2, idx, new_kv)


def alpamayo_rope(x, cos, sin, position_ids):
    if _use_decomposed_alpamayo():
        return decomposed_rope_shared(x, cos, sin, position_ids)
    return rope_onnx(x, cos, sin, position_ids)


def alpamayo_attention(query, key, value, attn_mask, is_causal, scale):
    if _use_decomposed_alpamayo():
        return decomposed_attention_mask4d(query, key, value, attn_mask,
                                           is_causal, scale)
    return attention_onnx(query, key, value, attn_mask, is_causal, scale)


def alpamayo_kv_cache_update(cache, new_kv, cache_indices):
    if _use_decomposed_alpamayo():
        return decomposed_kv_cache_update(cache, new_kv, cache_indices)
    return kv_cache_update_onnx(cache, new_kv, cache_indices)

'''

# ActionAttention.forward 里的局部 import（出现一次），换成走包装器（别名回原名，
# 调用点无需改动）。
IMPORT_OLD = "from ..ops import attention_onnx, kv_cache_update_onnx, rope_onnx"
IMPORT_NEW = ("from ..ops import alpamayo_attention as attention_onnx, "
              "alpamayo_kv_cache_update as kv_cache_update_onnx, "
              "alpamayo_rope as rope_onnx")


def patch_ops(diff_only):
    src = open(OPS).read()
    if "def decomposed_kv_cache_update(" in src:
        print("ops.py already patched")
        return
    patched = src.rstrip("\n") + HELPERS
    if diff_only:
        _diff(src, patched, OPS)
        return
    open(OPS, "w").write(patched)
    print("PATCHED", OPS)


def patch_modeling(diff_only):
    src = open(MODELING).read()
    if IMPORT_NEW in src:
        print(MODELING, "already patched")
        return
    n = src.count(IMPORT_OLD)
    if n != 1:
        print("ANCHOR count=%d in %s — 需人工适配" % (n, MODELING))
        sys.exit(1)
    patched = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    if diff_only:
        _diff(src, patched, MODELING)
        return
    open(MODELING, "w").write(patched)
    print("PATCHED", MODELING)


def _diff(a, b, name):
    import difflib
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                     fromfile=name, tofile=name + ".patched",
                                     lineterm="", n=2):
        print(line)


def main():
    diff_only = "--diff" in sys.argv
    patch_ops(diff_only)
    patch_modeling(diff_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
