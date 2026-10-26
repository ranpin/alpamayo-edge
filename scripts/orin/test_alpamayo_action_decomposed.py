#!/usr/bin/env python3
"""Numerically validate the R1 alpamayo action decomposed ops before export.

导出 action 专家 + 建引擎很贵；若分解实现有语义 bug，建引擎时才暴露就太晚。
本测试在 CPU 上用小张量对拍三个分解算子，**并专门覆盖两个只在 B>1 才暴露的坑**
（B==1 时错误实现会碰巧广播正确，导出 dummy 恰好是 B=1 ⇒ 极易漏过）：

  - decomposed_kv_cache_update  vs  直接切片赋值（真 ground truth）
        cache_indices 是运行期张量（导出 dummy=3027），验多个起始位 + B>1。
  - decomposed_attention_mask4d vs  F.scaled_dot_product_attention
        alpamayo 的 attn_mask 是 4D [B,1,S,Sk]，scores 是 5D [B,Hkv,rep,S,Sk]。
        坑：不升维直接相加只在 B==1 碰巧对 ⇒ 本测试显式断言 B>1 时
        「naive 相加」与 ground truth 不同，证明升维是必需的（回归护栏）。
  - decomposed_rope_shared      vs  LLaMA 全维公式 q*cos_full + rotate_half(q)*sin_full
        alpamayo 的 cos/sin 只有 S 行、跨 batch 共享（forward 取 rope_cos[0:1]）。
        坑：cosmos3 版 decomposed_rope 的 cos.reshape(b,s,half) 在 B>1 会元素不足
        ⇒ 本测试用 B=2 覆盖。

判据打印实际值 vs 期望值（max abs err），阈值 1e-4（fp32）。
Usage (goat, inside trtedge-0.10.1/, export_venv, AFTER applying the patch):
    USE_DECOMPOSED_ALPAMAYO=1 ./export_venv/bin/python \
        ../scripts/orin/test_alpamayo_action_decomposed.py
"""
import os
import sys

os.environ.setdefault("USE_DECOMPOSED_ALPAMAYO", "1")

import torch
import torch.nn.functional as F

from tensorrt_edgellm.models.ops import (alpamayo_attention, alpamayo_kv_cache_update,
                                         alpamayo_rope, decomposed_attention_mask4d,
                                         decomposed_kv_cache_update,
                                         decomposed_rope_shared)

TOL = 1e-4
ok = True

# Alpamayo-R1 expert_cfg 实测维度（config.json）：head_dim=128, hidden=2048,
# num_attention_heads=16；GQA 2:1 与 cosmos3 同构 ⇒ Hkv=8。n_diffusion_tokens
# 取 ActionConfig 默认 64，max_kv_cache_capacity 取官方指南的 4096，
# kvcache_start_index 取导出 dummy 的 3027。
D = 128
H = 16
HKV = 8
S = 64        # n_diffusion_tokens (q_len)
C = 4096      # max_kv_cache_capacity
START = 3027  # 导出 dummy 的 kvcache_start_index


def check(name, got, ref):
    global ok
    err = (got - ref).abs().max().item()
    shape_ok = got.shape == ref.shape
    passed = err < TOL and shape_ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: "
          f"max_abs_err={err:.3e} (tol {TOL:.0e}), "
          f"shape got={tuple(got.shape)} ref={tuple(ref.shape)}")


# --------------------------------------------------------------------------- #
# 1. KV cache update
# --------------------------------------------------------------------------- #
def ref_kv_update(cache, new_kv, start):
    """Ground truth: 直接切片赋值（start 是 Python int）。"""
    out = cache.clone()
    out[:, :, start:start + new_kv.shape[2], :] = new_kv
    return out


def test_kv_update(b, start, label):
    torch.manual_seed(0)
    cache = torch.randn(b, HKV, C, D, dtype=torch.float32)
    new_kv = torch.randn(b, HKV, S, D, dtype=torch.float32)
    idx = torch.full((b,), start, dtype=torch.int32)
    got = decomposed_kv_cache_update(cache, new_kv, idx)
    ref = ref_kv_update(cache, new_kv, start)
    check(f"kv_cache_update/{label}", got, ref)
    # 写入区外必须逐位不变（scatter 不应污染其它位置）
    outside = got.clone()
    outside[:, :, start:start + S, :] = 0
    ref_outside = cache.clone()
    ref_outside[:, :, start:start + S, :] = 0
    check(f"kv_cache_update/{label}/outside-untouched", outside, ref_outside)


# --------------------------------------------------------------------------- #
# 2. Attention with 4D additive mask (GQA)
# --------------------------------------------------------------------------- #
def ref_repeat_kv(x, rep):
    """HF GQA repeat_kv: kv head h//rep serves query head h (interleave)."""
    b, hkv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, hkv, rep, s,
                                      d).reshape(b, hkv * rep, s, d)


def make_alpamayo_mask(b, q_len, kv_cap, start):
    """复刻 modeling_alpamayo_action.py 的掩码：0.0 有效、-inf 超出 valid_length。"""
    valid = (start + q_len)
    pos = torch.arange(kv_cap, dtype=torch.int32).view(1, 1, 1, -1)
    m = torch.zeros(b, 1, q_len, kv_cap, dtype=torch.float32)
    return m.masked_fill(pos >= valid, float("-inf"))


def test_attention(b, label):
    global ok
    torch.manual_seed(1)
    q = torch.randn(b, H, S, D, dtype=torch.float32)
    k = torch.randn(b, HKV, C, D, dtype=torch.float32)
    v = torch.randn(b, HKV, C, D, dtype=torch.float32)
    mask = make_alpamayo_mask(b, S, C, START)
    scale = 1.0 / (D ** 0.5)
    got = decomposed_attention_mask4d(q, k, v, mask, False, scale)
    ref = F.scaled_dot_product_attention(q, ref_repeat_kv(k, H // HKV),
                                         ref_repeat_kv(v, H // HKV),
                                         attn_mask=mask, scale=scale)
    check(f"attention/{label}", got, ref)

    # 回归护栏：B>1 时「不升维直接相加」必须与 ground truth 不同，
    # 否则说明本测试没真正覆盖到那个坑。
    if b > 1:
        hkv, rep = HKV, H // HKV
        qq = q.view(b, hkv, rep, S, D)
        scores = torch.matmul(qq, k.unsqueeze(2).transpose(-1, -2)) * scale
        try:
            naive = torch.softmax(scores + mask, dim=-1)
            naive = torch.matmul(naive, v.unsqueeze(2)).reshape(b, H, S, D)
            naive_err = (naive - ref).abs().max().item()
        except RuntimeError as e:
            naive_err = float("inf")
            print(f"    (naive broadcast raised: {type(e).__name__})")
        guard = naive_err >= TOL
        ok = ok and guard
        print(f"[{'PASS' if guard else 'FAIL'}] attention/{label}/"
              f"naive-4D-mask-is-wrong: err={naive_err:.3e} "
              f"(must be >= tol, else the B>1 trap is not covered)")


# --------------------------------------------------------------------------- #
# 3. RoPE with batch-shared cos/sin
# --------------------------------------------------------------------------- #
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def test_rope(b, label):
    half = D // 2
    torch.manual_seed(2)
    x = torch.randn(b, H, S, D, dtype=torch.float32)
    # alpamayo forward: rope_cos = rope_rotary_cos_sin[0:1].reshape(-1, half)
    # ⇒ 只有 S 行、跨 batch 共享。
    cos_rows = torch.randn(S, half, dtype=torch.float32)
    sin_rows = torch.randn(S, half, dtype=torch.float32)
    pos = torch.arange(S, dtype=torch.int32).unsqueeze(0).expand(b, -1)
    got = decomposed_rope_shared(x, cos_rows, sin_rows, pos)
    # 独立参考：LLaMA 全维公式，按 pos gather（此处恒等）
    idx = pos.long().unsqueeze(-1).expand(b, S, half)
    cos_g = cos_rows.unsqueeze(0).expand(b, S, half).gather(1, idx)
    sin_g = sin_rows.unsqueeze(0).expand(b, S, half).gather(1, idx)
    cos_full = torch.cat([cos_g, cos_g], dim=-1).unsqueeze(1)  # [B,1,S,D]
    sin_full = torch.cat([sin_g, sin_g], dim=-1).unsqueeze(1)
    ref = x * cos_full + rotate_half(x) * sin_full
    check(f"rope/{label}", got, ref)


# --------------------------------------------------------------------------- #
# 4. Dispatch: env=1 时包装器必须走分解实现
# --------------------------------------------------------------------------- #
def test_dispatch():
    b = 2
    torch.manual_seed(3)
    q = torch.randn(b, H, S, D)
    k = torch.randn(b, HKV, C, D)
    v = torch.randn(b, HKV, C, D)
    mask = make_alpamayo_mask(b, S, C, START)
    check("dispatch/attention", alpamayo_attention(q, k, v, mask, False, 1.0),
          decomposed_attention_mask4d(q, k, v, mask, False, 1.0))
    cache = torch.randn(b, HKV, C, D)
    new_kv = torch.randn(b, HKV, S, D)
    idx = torch.full((b,), START, dtype=torch.int32)
    check("dispatch/kv_cache_update",
          alpamayo_kv_cache_update(cache, new_kv, idx),
          decomposed_kv_cache_update(cache, new_kv, idx))
    half = D // 2
    x = torch.randn(b, H, S, D)
    cos = torch.randn(S, half)
    sin = torch.randn(S, half)
    pos = torch.arange(S, dtype=torch.int32).unsqueeze(0).expand(b, -1)
    check("dispatch/rope", alpamayo_rope(x, cos, sin, pos),
          decomposed_rope_shared(x, cos, sin, pos))


print("=== R1 alpamayo action decomposed op validation ===")
print(f"dims: B={{1,2}} H={H} Hkv={HKV} D={D} S={S} C={C} start={START}")
test_kv_update(1, START, "b1_start3027")
test_kv_update(2, START, "b2_start3027")
test_kv_update(2, 0, "b2_start0")
test_kv_update(1, C - S, "b1_start_at_end")
test_attention(1, "b1_gqa_4dmask")
test_attention(2, "b2_gqa_4dmask")
test_rope(1, "b1")
test_rope(2, "b2_shared_cos_sin")
test_dispatch()
print("=== " + ("ALL PASS" if ok else "SOME FAIL") + " ===")
sys.exit(0 if ok else 1)
