"""
Wasserstein距离计算模块
========================
实现 paper.md 公式1 中的最优传输Wasserstein距离。
用于计算ε₂(v) = (1/K) Σ_k W(P_k(v), P̂_seg(v))。

依赖: numpy, scipy
"""

import numpy as np
from scipy.optimize import linprog


def wasserstein_distance(p, q, D):
    """计算两个离散分布间的1-Wasserstein距离 (paper.md 公式1)

    D_W(P, Q) = min_{Π≥0} Σ_ij d_ij π_ij
    s.t. Σ_j π_ij = p_i, ∀i  (P的边缘约束)
         Σ_i π_ij = q_j, ∀j  (Q的边缘约束)

    Args:
        p: np.ndarray [N], 概率分布P
        q: np.ndarray [N], 概率分布Q
        D: np.ndarray [N,N], 基准距离矩阵 (非负, 对称)

    Returns:
        dist: float, Wasserstein距离 (≥0)
    """
    N = len(p)
    assert p.shape == (N,) and q.shape == (N,), \
        f"Shape mismatch: p={p.shape}, q={q.shape}"
    assert D.shape == (N, N), f"D shape mismatch: {D.shape}"

    # 特判: 两个分布完全相同
    if np.allclose(p, q, atol=1e-12):
        return 0.0

    # 目标: min Σ_ij d_ij π_ij
    c_obj = D.flatten()

    # 等式约束: 2N个 (P边缘 + Q边缘)
    A_eq = np.zeros((2 * N, N * N))
    b_eq = np.zeros(2 * N)

    # Σ_j π_ij = p_i, ∀i
    for i in range(N):
        for j in range(N):
            A_eq[i, i * N + j] = 1.0
        b_eq[i] = p[i]

    # Σ_i π_ij = q_j, ∀j
    for j in range(N):
        for i in range(N):
            A_eq[N + j, i * N + j] = 1.0
        b_eq[N + j] = q[j]

    result = linprog(
        c_obj,
        A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, None)] * (N * N),
        method='highs'
    )

    if result.success:
        return max(result.fun, 0.0)  # 数值保护
    else:
        # fallback: L1距离的下界
        return 0.0


def wasserstein_distance_batch(P_batch, q, D):
    """批量计算多个分布到同一参考分布q的Wasserstein距离

    Args:
        P_batch: np.ndarray [K, N], K个分布
        q: np.ndarray [N], 参考分布
        D: np.ndarray [N,N], 基准距离矩阵

    Returns:
        distances: np.ndarray [K], 每个分布到q的Wasserstein距离
    """
    K = P_batch.shape[0]
    distances = np.zeros(K)
    for k in range(K):
        distances[k] = wasserstein_distance(P_batch[k], q, D)
    return distances


def compute_epsilon2(P_k_list, P_seg_hat, D):
    """计算认知不确定性半径 ε₂(v)

    ε₂(v) = (1/K) Σ_k W(P_k(v), P̂_seg(v))

    Args:
        P_k_list: np.ndarray [K, N], K个算法的预测分布
        P_seg_hat: np.ndarray [N], 算法共识分布 (K个的均值)
        D: np.ndarray [N,N], 基准距离矩阵

    Returns:
        epsilon2: float, 认知不确定性半径
    """
    distances = wasserstein_distance_batch(P_k_list, P_seg_hat, D)
    return distances.mean()
