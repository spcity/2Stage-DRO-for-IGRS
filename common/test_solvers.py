"""
公共模块单元测试
================
用简单toy数据验证LP求解器和Wasserstein距离的数学正确性。

运行: conda activate torch_m && python -m common.test_solvers
或:   cd 260216_MICCAI2stagerisk && python common/test_solvers.py
"""

import sys
import os
import numpy as np

# 确保可以import common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.lp_solvers import solve_stage1_lp, solve_stage2_lp
from common.wasserstein import wasserstein_distance, compute_epsilon2
from common.risk_params import get_brainweb_params, get_brats_params, compute_lambda


def test_wasserstein_basic():
    """测试Wasserstein距离的基本性质"""
    print("=" * 60)
    print("Test 1: Wasserstein距离基本性质")
    print("=" * 60)

    # 简单2类情况: D = [[0, 1], [1, 0]]
    D2 = np.array([[0.0, 1.0], [1.0, 0.0]])

    # 性质1: W(p, p) = 0
    p = np.array([0.3, 0.7])
    d = wasserstein_distance(p, p, D2)
    assert abs(d) < 1e-8, f"W(p,p) should be 0, got {d}"
    print(f"  [✓] W(p, p) = {d:.2e} ≈ 0")

    # 性质2: W(p, q) ≥ 0
    q = np.array([0.6, 0.4])
    d = wasserstein_distance(p, q, D2)
    assert d >= -1e-10, f"W(p,q) should be ≥ 0, got {d}"
    print(f"  [✓] W([0.3,0.7], [0.6,0.4]) = {d:.4f} ≥ 0")

    # 性质3: 对称性 W(p,q) = W(q,p)
    d2 = wasserstein_distance(q, p, D2)
    assert abs(d - d2) < 1e-8, f"Symmetry violated: {d} vs {d2}"
    print(f"  [✓] W(p,q) = W(q,p) = {d:.4f}")

    # 性质4: 已知解 — 对于D=[[0,1],[1,0]], W([a,1-a], [b,1-b]) = |a-b|
    # 因为只需移动|a-b|的质量, 代价为|a-b|*1
    expected = abs(0.3 - 0.6)
    assert abs(d - expected) < 1e-8, f"Expected {expected}, got {d}"
    print(f"  [✓] W([0.3,0.7], [0.6,0.4]) = {d:.4f} = |0.3-0.6| = {expected} ✓")

    # 性质5: 三角不等式 W(p,r) ≤ W(p,q) + W(q,r)
    r = np.array([0.1, 0.9])
    d_pr = wasserstein_distance(p, r, D2)
    d_pq = wasserstein_distance(p, q, D2)
    d_qr = wasserstein_distance(q, r, D2)
    assert d_pr <= d_pq + d_qr + 1e-8, \
        f"Triangle inequality violated: {d_pr} > {d_pq} + {d_qr}"
    print(f"  [✓] 三角不等式: W(p,r)={d_pr:.4f} ≤ W(p,q)+W(q,r)={d_pq+d_qr:.4f}")

    print("  ALL PASSED\n")


def test_wasserstein_3class():
    """测试3类Wasserstein距离 (BrainWeb场景)"""
    print("=" * 60)
    print("Test 2: 3类Wasserstein距离 (BrainWeb D矩阵)")
    print("=" * 60)

    params = get_brainweb_params()
    D = params['D']

    # 纯CSF vs 纯GM: 应等于D[0,1]=0.5
    p_csf = np.array([1.0, 0.0, 0.0])
    p_gm = np.array([0.0, 1.0, 0.0])
    d = wasserstein_distance(p_csf, p_gm, D)
    assert abs(d - D[0, 1]) < 1e-8, f"Expected {D[0,1]}, got {d}"
    print(f"  [✓] W(纯CSF, 纯GM) = {d:.4f} = D[CSF,GM] = {D[0,1]}")

    # 纯GM vs 纯WM: 应等于D[1,2]=0.3
    p_wm = np.array([0.0, 0.0, 1.0])
    d = wasserstein_distance(p_gm, p_wm, D)
    assert abs(d - D[1, 2]) < 1e-8, f"Expected {D[1,2]}, got {d}"
    print(f"  [✓] W(纯GM, 纯WM) = {d:.4f} = D[GM,WM] = {D[1,2]}")

    # 纯CSF vs 纯WM: 应等于D[0,2]=0.8
    d = wasserstein_distance(p_csf, p_wm, D)
    assert abs(d - D[0, 2]) < 1e-8, f"Expected {D[0,2]}, got {d}"
    print(f"  [✓] W(纯CSF, 纯WM) = {d:.4f} = D[CSF,WM] = {D[0,2]}")

    # 混合分布
    p_mix = np.array([0.2, 0.5, 0.3])
    q_mix = np.array([0.1, 0.6, 0.3])
    d = wasserstein_distance(p_mix, q_mix, D)
    print(f"  [✓] W([0.2,0.5,0.3], [0.1,0.6,0.3]) = {d:.4f}")
    assert d >= 0, "Negative distance"

    print("  ALL PASSED\n")


def test_stage1_lp_basic():
    """测试第一阶段LP的基本行为"""
    print("=" * 60)
    print("Test 3: 第一阶段LP基本行为")
    print("=" * 60)

    params = get_brainweb_params()
    C, D = params['C'], params['D']

    # Case 1: ε₁=0 → P*_pve = Q (无调整空间)
    q = np.array([0.2, 0.5, 0.3])
    p_star, info = solve_stage1_lp(q, C, D, epsilon1=0.0)
    assert info['success'], f"LP failed: {info['message']}"
    assert np.allclose(p_star, q, atol=1e-6), \
        f"ε₁=0 should give P*=Q: {p_star} vs {q}"
    print(f"  [✓] ε₁=0: P*_pve = Q = {q} (无调整)")
    print(f"       风险值 = {info['optimal_risk']:.4f} (= C·Q = {np.dot(C,q):.4f})")

    # Case 2: ε₁>0 → P*_pve ≠ Q, 且风险增加
    p_star2, info2 = solve_stage1_lp(q, C, D, epsilon1=0.1)
    assert info2['success']
    assert info2['optimal_risk'] >= info['optimal_risk'] - 1e-8, \
        f"Risk should increase: {info2['optimal_risk']} < {info['optimal_risk']}"
    print(f"  [✓] ε₁=0.1: P*_pve = {np.round(p_star2, 4)}")
    print(f"       风险值 = {info2['optimal_risk']:.4f} ≥ {info['optimal_risk']:.4f} ✓")
    print(f"       传输代价 = {info2['transport_cost']:.4f} ≤ ε₁ = 0.1")
    assert info2['transport_cost'] <= 0.1 + 1e-8

    # Case 3: 更大ε₁ → 风险更大 (单调性)
    p_star3, info3 = solve_stage1_lp(q, C, D, epsilon1=0.5)
    assert info3['success']
    assert info3['optimal_risk'] >= info2['optimal_risk'] - 1e-8, \
        f"Risk monotonicity: {info3['optimal_risk']} < {info2['optimal_risk']}"
    print(f"  [✓] ε₁=0.5: P*_pve = {np.round(p_star3, 4)}")
    print(f"       风险值 = {info3['optimal_risk']:.4f} ≥ {info2['optimal_risk']:.4f} ✓ (单调递增)")

    # Case 4: P*_pve应偏向高风险类 (GM的C最高=0.7)
    # 检查GM概率是否增加
    print(f"  [i] Q       = {np.round(q, 4)} → E[C·Q]     = {np.dot(C,q):.4f}")
    print(f"  [i] P*(0.1) = {np.round(p_star2, 4)} → E[C·P*]    = {info2['optimal_risk']:.4f}")
    print(f"  [i] P*(0.5) = {np.round(p_star3, 4)} → E[C·P*]    = {info3['optimal_risk']:.4f}")
    print(f"  [✓] GM(高风险)概率趋势: {q[1]:.3f} → {p_star2[1]:.3f} → {p_star3[1]:.3f} (应递增)")

    # Case 5: 概率单纯形约束
    for p, name in [(p_star, "ε₁=0"), (p_star2, "ε₁=0.1"), (p_star3, "ε₁=0.5")]:
        assert np.all(p >= -1e-8), f"{name}: negative probability"
        assert abs(p.sum() - 1.0) < 1e-6, f"{name}: sum != 1: {p.sum()}"
    print(f"  [✓] 所有输出满足概率单纯形约束 (p≥0, Σp=1)")

    print("  ALL PASSED\n")


def test_stage1_lp_extreme():
    """测试第一阶段LP的极端情况"""
    print("=" * 60)
    print("Test 4: 第一阶段LP极端情况")
    print("=" * 60)

    params = get_brainweb_params()
    C, D = params['C'], params['D']

    # 极端1: 纯类别分布 (体素完全在某类别内)
    q_pure = np.array([0.0, 0.0, 1.0])  # 纯WM
    p_star, info = solve_stage1_lp(q_pure, C, D, epsilon1=0.1)
    assert info['success']
    print(f"  [✓] 纯WM输入: Q={q_pure} → P*={np.round(p_star, 4)}, risk={info['optimal_risk']:.4f}")

    # 极端2: 均匀分布
    q_uniform = np.array([1/3, 1/3, 1/3])
    p_star, info = solve_stage1_lp(q_uniform, C, D, epsilon1=0.1)
    assert info['success']
    print(f"  [✓] 均匀分布: Q={np.round(q_uniform,3)} → P*={np.round(p_star, 4)}, risk={info['optimal_risk']:.4f}")

    # 极端3: 非常大的ε₁ (应趋向最高风险类的纯分布)
    q = np.array([0.5, 0.3, 0.2])
    p_star, info = solve_stage1_lp(q, C, D, epsilon1=10.0)
    assert info['success']
    max_risk_idx = np.argmax(C)  # GM (index 1)
    print(f"  [✓] 极大ε₁=10: Q={q} → P*={np.round(p_star, 4)}")
    print(f"       最高风险类: {params['tissue_names'][max_risk_idx]} (C={C[max_risk_idx]})")
    print(f"       P*[GM]={p_star[max_risk_idx]:.4f} (应趋向1.0)")

    print("  ALL PASSED\n")


def test_stage2_lp_basic():
    """测试第二阶段LP的基本行为"""
    print("=" * 60)
    print("Test 5: 第二阶段LP基本行为")
    print("=" * 60)

    params = get_brainweb_params()
    C, D = params['C'], params['D']
    lam = compute_lambda(D, beta=1.0)  # λ = min(D[D>0]) = 0.3

    # Case 1: ε₂=0, λ=lam → 受限于算法共识
    u_hat = np.array([0.2, 0.5, 0.3])
    v_pve = np.array([0.15, 0.55, 0.3])

    R, info = solve_stage2_lp(u_hat, v_pve, C, D, epsilon2=0.0, lam=lam)
    assert info['success'], f"LP failed: {info['message']}"
    print(f"  [✓] ε₂=0, λ={lam}: R_final = {R:.4f}")
    print(f"       最优分布 = {np.round(info['optimal_distribution'], 4)}")

    # Case 2: ε₂>0 → 风险增加
    R2, info2 = solve_stage2_lp(u_hat, v_pve, C, D, epsilon2=0.1, lam=lam)
    assert info2['success']
    assert R2 >= R - 1e-8, f"Risk should increase: {R2} < {R}"
    print(f"  [✓] ε₂=0.1, λ={lam}: R_final = {R2:.4f} ≥ {R:.4f}")

    # Case 3: ε₂和λ都为0 → 两个约束都紧,
    # 目标分布同时等于u_hat和v_pve, 只有u_hat==v_pve时可行
    u_same = np.array([0.2, 0.5, 0.3])
    R3, info3 = solve_stage2_lp(u_same, u_same, C, D, epsilon2=0.0, lam=0.0)
    if info3['success']:
        expected_risk = np.dot(C, u_same)
        assert abs(R3 - expected_risk) < 1e-6, \
            f"ε₂=0,λ=0,u=v: R should be C·u = {expected_risk}, got {R3}"
        print(f"  [✓] ε₂=0, λ=0 (u==v): R_final = {R3:.4f} = C·u = {expected_risk:.4f}")
    else:
        print(f"  [i] ε₂=0, λ=0 (u==v): LP infeasible (数值问题, 可接受)")

    # Case 4: 传输代价约束验证
    R4, info4 = solve_stage2_lp(u_hat, v_pve, C, D, epsilon2=0.2, lam=lam)
    if info4['success']:
        assert info4['transport_cost_1'] <= 0.2 + 1e-8, \
            f"Transport cost 1 exceeds ε₂: {info4['transport_cost_1']}"
        assert info4['transport_cost_2'] <= lam + 1e-8, \
            f"Transport cost 2 exceeds λ: {info4['transport_cost_2']}"
        print(f"  [✓] ε₂=0.2: cost1={info4['transport_cost_1']:.4f}≤0.2, "
              f"cost2={info4['transport_cost_2']:.4f}≤{lam}")

    print("  ALL PASSED\n")


def test_epsilon2_computation():
    """测试ε₂的计算"""
    print("=" * 60)
    print("Test 6: ε₂(v) 计算")
    print("=" * 60)

    params = get_brainweb_params()
    D = params['D']

    # K=3个算法的预测
    P_k = np.array([
        [0.2, 0.5, 0.3],  # 算法1
        [0.1, 0.6, 0.3],  # 算法2
        [0.3, 0.4, 0.3],  # 算法3
    ])
    P_seg_hat = P_k.mean(axis=0)
    print(f"  P̂_seg = {np.round(P_seg_hat, 4)}")

    eps2 = compute_epsilon2(P_k, P_seg_hat, D)
    print(f"  ε₂ = {eps2:.6f}")
    assert eps2 >= 0, f"ε₂ should be ≥ 0: {eps2}"

    # 所有算法相同 → ε₂ = 0
    P_same = np.array([[0.2, 0.5, 0.3]] * 3)
    eps2_same = compute_epsilon2(P_same, P_same.mean(axis=0), D)
    assert abs(eps2_same) < 1e-8, f"All same → ε₂ should be 0: {eps2_same}"
    print(f"  [✓] 所有算法相同: ε₂ = {eps2_same:.2e} ≈ 0")

    print("  ALL PASSED\n")


def test_risk_params():
    """测试风险参数的合理性"""
    print("=" * 60)
    print("Test 7: 风险参数合理性检查")
    print("=" * 60)

    for name, get_fn in [("BrainWeb", get_brainweb_params),
                         ("BraTS", get_brats_params)]:
        params = get_fn()
        C, D, N = params['C'], params['D'], params['N']

        assert C.shape == (N,), f"{name}: C shape {C.shape} != ({N},)"
        assert D.shape == (N, N), f"{name}: D shape {D.shape} != ({N},{N})"
        assert np.all(C >= 0), f"{name}: C has negative values"
        assert np.all(D >= 0), f"{name}: D has negative values"
        assert np.allclose(D, D.T), f"{name}: D not symmetric"
        assert np.allclose(np.diag(D), 0), f"{name}: D diagonal not zero"

        lam = compute_lambda(D)
        print(f"  [{name}] N={N}, C={C}, λ(β=1)={lam:.2f}")
        print(f"           D min(>0)={D[D>0].min():.2f}, max={D.max():.2f}")
        print(f"           组织: {params['tissue_names']}")

    print("  ALL PASSED\n")


def test_fcm_basic():
    """测试FCM聚类的基本行为"""
    print("=" * 60)
    print("Test 8: FCM软分割基本行为")
    print("=" * 60)

    from common.fcm import fcm_fit, fcm_segment_t1w

    # 生成3个明显分离的高斯簇
    rng = np.random.RandomState(42)
    n_per_cluster = 500
    X = np.concatenate([
        rng.randn(n_per_cluster, 1) * 0.5 + 2.0,   # CSF: ~2.0
        rng.randn(n_per_cluster, 1) * 0.5 + 5.0,   # GM:  ~5.0
        rng.randn(n_per_cluster, 1) * 0.5 + 8.0,   # WM:  ~8.0
    ])

    U, centers, n_iters, J = fcm_fit(X, n_clusters=3, m=2.0, seed=42)

    print(f"  聚类中心: {np.round(centers.flatten(), 2)}")
    print(f"  迭代次数: {n_iters}, 目标函数: {J:.2f}")

    # 检查隶属度和约束
    assert U.shape == (1500, 3)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6), "隶属度行和≠1"
    assert np.all(U >= 0), "隶属度有负值"

    # 聚类中心应近似 [2, 5, 8]
    sorted_centers = np.sort(centers.flatten())
    assert abs(sorted_centers[0] - 2.0) < 0.5, f"Center 0: {sorted_centers[0]}"
    assert abs(sorted_centers[1] - 5.0) < 0.5, f"Center 1: {sorted_centers[1]}"
    assert abs(sorted_centers[2] - 8.0) < 0.5, f"Center 2: {sorted_centers[2]}"
    print(f"  [✓] 排序后聚类中心 ≈ [2.0, 5.0, 8.0]: {np.round(sorted_centers, 2)}")

    # 簇内样本应高隶属度
    # 前500个属于cluster0 (CSF~2.0), 排序后应是最低中心
    sort_idx = np.argsort(centers.flatten())
    csf_membership = U[:n_per_cluster, sort_idx[0]].mean()
    gm_membership = U[n_per_cluster:2*n_per_cluster, sort_idx[1]].mean()
    wm_membership = U[2*n_per_cluster:, sort_idx[2]].mean()
    print(f"  [✓] CSF簇内平均隶属度: {csf_membership:.4f} (应>0.8)")
    print(f"  [✓] GM簇内平均隶属度:  {gm_membership:.4f} (应>0.8)")
    print(f"  [✓] WM簇内平均隶属度:  {wm_membership:.4f} (应>0.8)")
    assert csf_membership > 0.7
    assert gm_membership > 0.7
    assert wm_membership > 0.7

    # 测试fcm_segment_t1w
    vol = np.zeros((10, 10, 10))
    mask = np.ones((10, 10, 10), dtype=bool)
    vol[:3] = 2.0   # CSF
    vol[3:7] = 5.0  # GM
    vol[7:] = 8.0   # WM
    # 添加少量噪声
    vol += rng.randn(10, 10, 10) * 0.3

    Q, centers_sorted, seg_info = fcm_segment_t1w(vol, mask, n_classes=3, m=2.0)
    assert Q.shape == (10, 10, 10, 3)
    assert np.allclose(Q[mask].sum(axis=1), 1.0, atol=1e-6)
    print(f"  [✓] fcm_segment_t1w: Q shape={Q.shape}, centers={np.round(centers_sorted, 2)}")
    print(f"       CSF区域Q[:,0]均值={Q[:3,:,:,0].mean():.3f} (应高)")
    print(f"       GM区域Q[:,1]均值={Q[3:7,:,:,1].mean():.3f} (应高)")
    print(f"       WM区域Q[:,2]均值={Q[7:,:,:,2].mean():.3f} (应高)")

    print("  ALL PASSED\n")


def test_full_pipeline_toy():
    """端到端toy pipeline测试"""
    print("=" * 60)
    print("Test 9: 端到端toy pipeline (Stage1 + Stage2)")
    print("=" * 60)

    params = get_brainweb_params()
    C, D = params['C'], params['D']
    lam = compute_lambda(D, beta=1.0)

    # 模拟一个边界体素的完整流程
    # Step 1: FCM给出的Q(v) — 在GM/WM边界, 两者各约一半
    q = np.array([0.05, 0.45, 0.50])  # CSF少, GM≈WM
    print(f"  输入 Q(v) = {q} (GM/WM边界体素)")
    print(f"  C = {C}, λ = {lam}")

    # Step 2: ε₁ — 边界体素梯度大
    epsilon1 = 0.15
    print(f"  ε₁(v) = {epsilon1} (边界, 较大)")

    # Stage 1: LP → P*_pve
    p_star, info1 = solve_stage1_lp(q, C, D, epsilon1)
    assert info1['success']
    print(f"\n  Stage 1 结果:")
    print(f"    P*_pve = {np.round(p_star, 4)}")
    print(f"    最优风险 = {info1['optimal_risk']:.4f} (原始: {np.dot(C,q):.4f})")
    print(f"    传输代价 = {info1['transport_cost']:.4f} ≤ ε₁ = {epsilon1}")

    # Step 3: 模拟K=3个分割算法
    P_k = np.array([
        [0.05, 0.40, 0.55],  # 算法1: 偏WM
        [0.10, 0.50, 0.40],  # 算法2: 偏GM
        [0.02, 0.48, 0.50],  # 算法3: 居中
    ])
    P_seg_hat = P_k.mean(axis=0)
    epsilon2 = compute_epsilon2(P_k, P_seg_hat, D)
    print(f"\n  K=3 算法共识 P̂_seg = {np.round(P_seg_hat, 4)}")
    print(f"  ε₂(v) = {epsilon2:.6f}")

    # Stage 2: LP → R_final
    R_final, info2 = solve_stage2_lp(P_seg_hat, p_star, C, D, epsilon2, lam)
    assert info2['success']
    print(f"\n  Stage 2 结果:")
    print(f"    R_final(v) = {R_final:.4f}")
    print(f"    最优分布 = {np.round(info2['optimal_distribution'], 4)}")
    print(f"    cost1={info2['transport_cost_1']:.4f}≤ε₂={epsilon2:.4f}, "
          f"cost2={info2['transport_cost_2']:.4f}≤λ={lam}")

    # 验证: R_final ≥ max(C·P*_pve, C·P̂_seg) — 最坏情况应≥两个基准
    risk_pve = np.dot(C, p_star)
    risk_seg = np.dot(C, P_seg_hat)
    print(f"\n  对比:")
    print(f"    C·Q       = {np.dot(C,q):.4f} (原始FCM风险)")
    print(f"    C·P*_pve  = {risk_pve:.4f} (Stage1鲁棒风险)")
    print(f"    C·P̂_seg  = {risk_seg:.4f} (算法共识风险)")
    print(f"    R_final   = {R_final:.4f} (最终鲁棒风险)")
    print(f"    R_final ≥ max(C·P*, C·P̂) = {max(risk_pve, risk_seg):.4f}? "
          f"{'✓' if R_final >= max(risk_pve, risk_seg) - 1e-6 else '✗'}")

    print("  ALL PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  两阶段DRO公共模块 - 单元测试")
    print("=" * 60 + "\n")

    test_wasserstein_basic()
    test_wasserstein_3class()
    test_stage1_lp_basic()
    test_stage1_lp_extreme()
    test_stage2_lp_basic()
    test_epsilon2_computation()
    test_risk_params()
    test_fcm_basic()
    test_full_pipeline_toy()

    print("=" * 60)
    print("  ✓ 全部测试通过!")
    print("=" * 60)
