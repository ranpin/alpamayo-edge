#!/usr/bin/env python3
"""R1 seed-vs-conditioning sensitivity repro (PROBLEMS §M addendum).

W4 的教训直接适用 here：测 conditioning 敏感性**必须带一个非 conditioning 轴作对照**，
否则「conditioning 弱」与「输出被噪声主导」不可区分。R1 的对照轴是 `--noiseSeed`。

设计要点 —— **必须 temperature=0**：
  VLM 文本头默认 temperature=1.0 采样，而生成出的 CoT token 会进入 action 专家的
  conditioning（UND K/V）。若不停掉文本采样，seed 轴与文本轴会混在一起，量不出干净结论。
  实测 temp=0 下文本完全确定（6 次跑同一句），seed 轴才被隔离出来。

七次推理（goat，engines/alpamayo_r1 三引擎，16 张 gate4 多相机帧）：
  det_a / det_b  temp=0 seed=5 真实 ego 历史   → 确定性底噪（应逐位相同）
  s0/s1/s2/s3    temp=0 seed=0/1/2/3 真实历史  → SEED 轴
  h0             temp=0 seed=5 全零 ego 历史   → EGO-HISTORY 轴（seed 固定）

证据（committed）：seed_sensitivity/r1_seed_{det_a,det_b,s0,s1,s2,s3,h0}.json
  —— 原始归一化 output_trajectory [64,2] = (accel, kappa) + output_text。
  ⚠️ 输入图像是 gated 数据集帧，**只存路径不入库**；输入 JSON 见
  scripts/orin/r1_full_chain_infer.sh 的构造逻辑。

Regeneration (goat)：见 scripts/orin/r1_full_chain_infer.sh（含 7 配置循环）。

Usage:
    .venv/bin/python eval/r1/r1_seed_sensitivity.py
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "seed_sensitivity")
NAMES = ["det_a", "det_b", "s0", "s1", "s2", "s3", "h0"]

# compute_minade.py 的反归一化常数（取自 R1 config.json action_space_cfg）
ACCEL_MEAN, ACCEL_STD = 0.029052734375, 0.6796875
CURV_MEAN, CURV_STD = 0.0002689361572265625, 0.026123046875


def load(name):
    r = json.load(open(os.path.join(D, f"r1_seed_{name}.json")))["responses"][0]
    return np.array(r["output_trajectory"], dtype=float).reshape(64, 2), r["output_text"]


def main():
    raw, txt = {}, {}
    for n in NAMES:
        raw[n], txt[n] = load(n)

    ref = raw["det_a"]
    spread = np.abs(ref).mean()
    within = np.abs(np.diff(ref, axis=0)).mean()

    print("=== R1 seed-vs-conditioning sensitivity (PROBLEMS §M) ===")
    print(f"[0] signal scale (det_a, raw normalized space)")
    print(f"    mean|traj| = {spread:.6f}")
    print(f"    mean|step-to-step within traj| = {within:.6f}")
    print(f"    accel col [{ref[:,0].min():+.4f},{ref[:,0].max():+.4f}]  "
          f"kappa col [{ref[:,1].min():+.5f},{ref[:,1].max():+.5f}]")

    def md(a, b):
        d = np.abs(raw[a] - raw[b])
        return d.mean(), d.max()

    def show(label, a, b):
        m, x = md(a, b)
        print(f"    {label:<22s} mean|d|={m:.6f} ({100*m/spread:5.2f}%)  max|d|={x:.6f}")
        return m

    print("\n[T0] determinism — temp=0, seed=5, identical input, run twice")
    floor = show("det_a vs det_b", "det_a", "det_b")
    bit = np.array_equal(raw["det_a"], raw["det_b"])
    print(f"    bitwise identical: {bit}")

    print("\n[T1] SEED axis — temp=0, real ego-hist, only --noiseSeed varies")
    vs_ref = [show(f"seed5 vs {n}", "det_a", n) for n in ["s0", "s1", "s2", "s3"]]
    cross = []
    for a, b in itertools.combinations(["s0", "s1", "s2", "s3"], 2):
        cross.append(show(f"{a} vs {b}", a, b))

    print("\n[T2] EGO-HISTORY axis — temp=0, seed=5, zero vs real history")
    h = show("real vs zero", "det_a", "h0")

    print("\n[T3] TEXT head (VLM CoT) — deterministic at temp=0?")
    same = len({txt[n] for n in ["det_a", "det_b", "s0", "s1", "s2", "s3"]}) == 1
    print(f"    6 runs (det_a/det_b/s0-s3) all identical text: {same}")
    print(f"      -> {txt['det_a']!r}")
    print(f"    ego-hist flip changes text: {txt['det_a'] != txt['h0']}")
    print(f"      h0 -> {txt['h0']!r}")

    print("\n[T4] denormalized physical units (det_a)")
    acc = ref[:, 0] * ACCEL_STD + ACCEL_MEAN
    kap = ref[:, 1] * CURV_STD + CURV_MEAN
    print(f"    accel  m/s^2 [{acc.min():+.3f},{acc.max():+.3f}] mean {acc.mean():+.3f}"
          f"   (bounds ±9.8: {bool(np.all(np.abs(acc) <= 9.8))})")
    print(f"    kappa  1/m   [{kap.min():+.6f},{kap.max():+.6f}] mean {kap.mean():+.6f}"
          f"   (bounds ±0.33: {bool(np.all(np.abs(kap) <= 0.33))})")

    seed_m, hist_m = float(np.mean(vs_ref)), h
    seed_cross = float(np.mean(cross))
    print("\n=== VERDICT ===")
    print(f"  determinism floor   = {floor:.8f}  (bitwise={bit})")
    print(f"  SEED effect (vs ref seed5, 4 pairs) = {seed_m:.6f}  "
          f"= {100*seed_m/spread:5.2f}% of mean|traj|")
    print(f"  SEED mean pairwise (all 10 pairs)   = {seed_cross:.6f}  "
          f"= {100*seed_cross/spread:5.2f}%")
    print(f"  EGO-HIST effect     = {hist_m:.6f}  = {100*hist_m/spread:5.2f}% of mean|traj|")
    print(f"  within-traj step    = {within:.6f}  = {100*within/spread:5.2f}% of mean|traj|")
    print(f"  SEED / EGO-HIST     = {seed_m/hist_m:.2f}x")
    print(f"\n  [W4 Cosmos3 policy 参照: SEED/conditioning = 39.3x, seed 占 51.9%]")
    print(f"  => R1 的条件耦合比 Cosmos3 policy 强约 {39.3/(seed_m/hist_m):.0f} 倍，"
          f"但 seed 仍主导轨迹头。")
    print("\n  ⚠️ 解读口径：compute_minade.py 是 resps[:N] 采 N 条再 ade.min()——")
    print("     minADE 的 'min' 本就跨 seed 取最优，故 seed 间差异大是扩散式多模态")
    print("     轨迹预测器的设计属性，官方指标已吸收，不等于缺陷。")
    print("     但做 demo 必须固定 --noiseSeed 才可复现。")


if __name__ == "__main__":
    main()
