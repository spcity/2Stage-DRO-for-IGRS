"""
两阶段DRO线性规划求解器
========================
严格实现 paper.md 中的公式3（第一阶段）和公式7（第二阶段）。

第一阶段LP: 在Wasserstein球内寻找最大风险分布 → 鲁棒物理先验 P*_pve(v)
第二阶段LP: 双重约束模糊集内的最大风险 → 最终风险值 R_final(v)

依赖: numpy, scipy (均为标准科学计算库)
"""

import numpy as np
from scipy.optimize import linprog


def solve_stage1_lp(q, C, D, epsilon1):
    """第一阶段LP：在Wasserstein球内寻找最大风险分布 (paper.md 公式3)

    max  Σ_i C_i * (Σ_j π_ij)
    s.t. Σ_i π_ij = q_j,  ∀j        (质量守恒: 源边缘=观测分布)
         Σ_ij d_ij * π_ij ≤ ε₁      (Wasserstein距离约束)
         π_ij ≥ 0

    Args:
        q: np.ndarray [N], 观测分布 (来自FCM), 需满足 q>=0, sum(q)=1
        C: np.ndarray [N], 风险成本向量
        D: np.ndarray [N,N], 基准距离矩阵 (对称, 非负, 对角线为0)
        epsilon1: float, PVE鲁棒半径 (≥0)

    Returns:
        p_star: np.ndarray [N], 鲁棒物理先验 P*_pve(v), 满足概率单纯形约束
        info: dict, 包含求解状态和最优风险值
    """
    N = len(q)
    assert q.shape == (N,), f"q shape mismatch: {q.shape} vs ({N},)"
    assert C.shape == (N,), f"C shape mismatch: {C.shape} vs ({N},)"
    assert D.shape == (N, N), f"D shape mismatch: {D.shape} vs ({N},{N})"
    assert epsilon1 >= 0, f"epsilon1 must be non-negative: {epsilon1}"

    # 决策变量: π_ij 展平为 [π_00, π_01, ..., π_{N-1,N-1}], 共N²个
    n_vars = N * N

    # 目标函数: max Σ_i C_i * (Σ_j π_ij) → linprog求min, 故取负
    # p_i = Σ_j π_ij, 所以目标 = Σ_i C_i * p_i = Σ_i Σ_j C_i * π_ij
    c_obj = np.zeros(n_vars)
    for i in range(N):
        for j in range(N):
            c_obj[i * N + j] = -C[i]

    # 等式约束: Σ_i π_ij = q_j, ∀j (N个约束)
    # 这保证源边缘分布 = 观测分布Q(v)
    A_eq = np.zeros((N, n_vars))
    for j in range(N):
        for i in range(N):
            A_eq[j, i * N + j] = 1.0
    b_eq = q.copy()

    # 不等式约束: Σ_ij d_ij * π_ij ≤ ε₁ (1个约束)
    A_ub = np.zeros((1, n_vars))
    for i in range(N):
        for j in range(N):
            A_ub[0, i * N + j] = D[i, j]
    b_ub = np.array([epsilon1])

    # 求解
    result = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, None)] * n_vars,
        method='highs'
    )

    info = {
        'success': result.success,
        'status': result.status,
        'message': result.message,
    }

    if result.success:
        Pi = result.x.reshape(N, N)
        p_star = Pi.sum(axis=1)  # p*_i = Σ_j π*_ij (目标边缘分布)
        # 数值修正: 确保严格在概率单纯形内
        p_star = np.clip(p_star, 0, None)
        p_star /= p_star.sum()
        info['optimal_risk'] = -result.fun  # 最大风险值
        info['transport_plan'] = Pi
        info['transport_cost'] = np.sum(D * Pi)  # 实际Wasserstein距离
        return p_star, info
    else:
        # LP求解失败, 返回原始分布作为fallback
        info['optimal_risk'] = np.dot(C, q)
        return q.copy(), info


def solve_stage2_lp(u_hat, v_pve, C, D, epsilon2, lam):
    """第二阶段LP：双重约束模糊集内的最大风险 (paper.md 公式7)

    max  Σ_i C_i * (Σ_j π^(1)_ij)
    s.t. Σ_i π^(1)_ij = û_j,  ∀j           (源1: 算法共识边缘)
         Σ_i π^(2)_ik = v_k,  ∀k           (源2: 物理先验边缘)
         Σ_j π^(1)_ij = Σ_k π^(2)_ik, ∀i   (边缘分布耦合: 目标分布一致)
         Σ_ij d_ij π^(1)_ij ≤ ε₂           (认知距离约束)
         Σ_ik d_ik π^(2)_ik ≤ λ            (物理锚定约束)
         π^(1), π^(2) ≥ 0

    Args:
        u_hat: np.ndarray [N], 算法共识分布 P̂_seg(v)
        v_pve: np.ndarray [N], 鲁棒物理先验 P*_pve(v)
        C: np.ndarray [N], 风险成本向量
        D: np.ndarray [N,N], 基准距离矩阵
        epsilon2: float, 认知不确定性半径 (≥0)
        lam: float, 物理锚定半径 λ (≥0)

    Returns:
        R_final: float, 最终风险值 (标量)
        info: dict, 包含求解状态和最优分布
    """
    N = len(u_hat)
    assert u_hat.shape == (N,), f"u_hat shape mismatch: {u_hat.shape}"
    assert v_pve.shape == (N,), f"v_pve shape mismatch: {v_pve.shape}"
    assert C.shape == (N,), f"C shape mismatch: {C.shape}"
    assert D.shape == (N, N), f"D shape mismatch: {D.shape}"
    assert epsilon2 >= 0, f"epsilon2 must be non-negative: {epsilon2}"
    assert lam >= 0, f"lambda must be non-negative: {lam}"

    # 决策变量: π^(1)[0 : N²] 和 π^(2)[N² : 2N²]
    n_vars = 2 * N * N

    # 目标: max Σ_i C_i * (Σ_j π^(1)_ij) → linprog min取负
    c_obj = np.zeros(n_vars)
    for i in range(N):
        for j in range(N):
            c_obj[i * N + j] = -C[i]  # 只对π^(1)部分

    # === 等式约束 (3N个) ===
    A_eq = np.zeros((3 * N, n_vars))
    b_eq = np.zeros(3 * N)

    # (a) 源1边缘: Σ_i π^(1)_ij = û_j, ∀j (N个约束)
    for j in range(N):
        for i in range(N):
            A_eq[j, i * N + j] = 1.0
        b_eq[j] = u_hat[j]

    # (b) 源2边缘: Σ_i π^(2)_ik = v_k, ∀k (N个约束)
    for k in range(N):
        for i in range(N):
            A_eq[N + k, N * N + i * N + k] = 1.0
        b_eq[N + k] = v_pve[k]

    # (c) 耦合约束: Σ_j π^(1)_ij = Σ_k π^(2)_ik, ∀i (N个约束)
    # 即: Σ_j π^(1)_ij - Σ_k π^(2)_ik = 0, ∀i
    for i in range(N):
        for j in range(N):
            A_eq[2 * N + i, i * N + j] = 1.0          # π^(1)_ij
            A_eq[2 * N + i, N * N + i * N + j] = -1.0  # -π^(2)_ij

    # === 不等式约束 (2个) ===
    A_ub = np.zeros((2, n_vars))
    b_ub = np.zeros(2)

    # Σ_ij d_ij π^(1)_ij ≤ ε₂
    for i in range(N):
        for j in range(N):
            A_ub[0, i * N + j] = D[i, j]
    b_ub[0] = epsilon2

    # Σ_ik d_ik π^(2)_ik ≤ λ
    for i in range(N):
        for k in range(N):
            A_ub[1, N * N + i * N + k] = D[i, k]
    b_ub[1] = lam

    # 求解
    result = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, None)] * n_vars,
        method='highs'
    )

    info = {
        'success': result.success,
        'status': result.status,
        'message': result.message,
    }

    if result.success:
        R_final = -result.fun
        Pi1 = result.x[:N * N].reshape(N, N)
        Pi2 = result.x[N * N:].reshape(N, N)
        p_final = Pi1.sum(axis=1)  # 最终最坏情况分布
        p_final = np.clip(p_final, 0, None)
        p_final /= p_final.sum()
        info['optimal_distribution'] = p_final
        info['transport_plan_1'] = Pi1
        info['transport_plan_2'] = Pi2
        info['transport_cost_1'] = np.sum(D * Pi1)
        info['transport_cost_2'] = np.sum(D * Pi2)
        return R_final, info
    else:
        # fallback: 用算法共识的期望风险
        R_fallback = np.dot(C, u_hat)
        info['optimal_distribution'] = u_hat.copy()
        return R_fallback, info


def solve_stage1_batch(Q_batch, C, D, epsilon1_batch, n_jobs=1):
    """批量求解第一阶段LP (用于处理大量体素)

    Args:
        Q_batch: np.ndarray [M, N], M个体素的观测分布
        C: np.ndarray [N], 风险成本向量
        D: np.ndarray [N,N], 基准距离矩阵
        epsilon1_batch: np.ndarray [M], 每个体素的PVE鲁棒半径
        n_jobs: int, 并行作业数 (1=串行, -1=全部CPU)

    Returns:
        P_star_batch: np.ndarray [M, N], M个体素的鲁棒物理先验
        risk_batch: np.ndarray [M], 每个体素的最优风险值
        success_batch: np.ndarray [M], 布尔数组, 每个LP是否成功
    """
    M, N = Q_batch.shape
    assert epsilon1_batch.shape == (M,)

    if n_jobs == 1:
        # 串行求解
        P_star_batch = np.zeros_like(Q_batch)
        risk_batch = np.zeros(M)
        success_batch = np.zeros(M, dtype=bool)
        for idx in range(M):
            p_star, info = solve_stage1_lp(Q_batch[idx], C, D, epsilon1_batch[idx])
            P_star_batch[idx] = p_star
            risk_batch[idx] = info['optimal_risk']
            success_batch[idx] = info['success']
        return P_star_batch, risk_batch, success_batch
    else:
        from joblib import Parallel, delayed

        def _solve_one(idx):
            p_star, info = solve_stage1_lp(Q_batch[idx], C, D, epsilon1_batch[idx])
            return p_star, info['optimal_risk'], info['success']

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_solve_one)(idx) for idx in range(M)
        )
        P_star_batch = np.array([r[0] for r in results])
        risk_batch = np.array([r[1] for r in results])
        success_batch = np.array([r[2] for r in results])
        return P_star_batch, risk_batch, success_batch


def solve_stage2_batch(U_hat_batch, V_pve_batch, C, D,
                       epsilon2_batch, lam, n_jobs=1):
    """批量求解第二阶段LP

    Args:
        U_hat_batch: np.ndarray [M, N], M个体素的算法共识分布
        V_pve_batch: np.ndarray [M, N], M个体素的鲁棒物理先验
        C: np.ndarray [N], 风险成本向量
        D: np.ndarray [N,N], 基准距离矩阵
        epsilon2_batch: np.ndarray [M], 每个体素的认知不确定性半径
        lam: float, 物理锚定半径 λ
        n_jobs: int, 并行作业数

    Returns:
        R_final_batch: np.ndarray [M], 最终风险值
        success_batch: np.ndarray [M], 布尔数组
    """
    M, N = U_hat_batch.shape
    assert V_pve_batch.shape == (M, N)
    assert epsilon2_batch.shape == (M,)

    if n_jobs == 1:
        R_final_batch = np.zeros(M)
        success_batch = np.zeros(M, dtype=bool)
        for idx in range(M):
            r, info = solve_stage2_lp(
                U_hat_batch[idx], V_pve_batch[idx],
                C, D, epsilon2_batch[idx], lam
            )
            R_final_batch[idx] = r
            success_batch[idx] = info['success']
        return R_final_batch, success_batch
    else:
        from joblib import Parallel, delayed

        def _solve_one(idx):
            r, info = solve_stage2_lp(
                U_hat_batch[idx], V_pve_batch[idx],
                C, D, epsilon2_batch[idx], lam
            )
            return r, info['success']

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_solve_one)(idx) for idx in range(M)
        )
        R_final_batch = np.array([r[0] for r in results])
        success_batch = np.array([r[1] for r in results])
        return R_final_batch, success_batch
