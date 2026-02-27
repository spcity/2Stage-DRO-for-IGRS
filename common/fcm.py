"""
模糊C均值(FCM)软分割模块
=========================
对MRI图像体素进行软分割，生成观测概率分布 Q(v)。
支持任意N类，通过参数m控制模糊程度。

依赖: numpy
"""

import numpy as np


def fcm_fit(X, n_clusters, m=2.0, max_iter=300, tol=1e-6, seed=42):
    """模糊C均值聚类

    最小化目标函数: J = Σ_i Σ_c u_ic^m * ||x_i - μ_c||²
    其中 u_ic 是隶属度, μ_c 是聚类中心, m 是模糊指数。

    Args:
        X: np.ndarray [M, D], M个样本, D维特征 (通常D=1为T1w强度)
        n_clusters: int, 聚类数 (N)
        m: float, 模糊指数 (>1, 默认2.0, 越大越模糊)
        max_iter: int, 最大迭代次数
        tol: float, 收敛阈值 (隶属度变化的最大范数)
        seed: int, 随机种子

    Returns:
        U: np.ndarray [M, n_clusters], 隶属度矩阵 (每行和=1)
        centers: np.ndarray [n_clusters, D], 聚类中心
        n_iters: int, 实际迭代次数
        J: float, 最终目标函数值
    """
    rng = np.random.RandomState(seed)
    M, D = X.shape
    K = n_clusters

    assert m > 1.0, f"Fuzziness m must be > 1, got {m}"
    assert K >= 2, f"n_clusters must be >= 2, got {K}"

    # 初始化隶属度: 随机 + 归一化
    U = rng.rand(M, K)
    U = U / U.sum(axis=1, keepdims=True)

    centers = np.zeros((K, D))
    power = 2.0 / (m - 1.0)

    for iteration in range(max_iter):
        U_old = U.copy()

        # 更新聚类中心: μ_c = Σ_i u_ic^m * x_i / Σ_i u_ic^m
        U_m = U ** m  # [M, K]
        for c in range(K):
            w = U_m[:, c]  # [M]
            w_sum = w.sum()
            if w_sum > 0:
                centers[c] = (w[:, np.newaxis] * X).sum(axis=0) / w_sum
            else:
                # 退化情况: 随机重新初始化
                centers[c] = X[rng.randint(M)]

        # 更新隶属度: u_ic = 1 / Σ_c' (||x_i-μ_c|| / ||x_i-μ_c'||)^power
        # 先计算距离
        dist2 = np.zeros((M, K))  # [M, K]
        for c in range(K):
            diff = X - centers[c]  # [M, D]
            dist2[:, c] = (diff * diff).sum(axis=1)  # 欧氏距离平方

        # 处理距离为0的情况 (样本恰好在中心)
        dist2 = np.maximum(dist2, 1e-16)
        dist = np.sqrt(dist2)  # [M, K]

        # 隶属度更新
        U = np.zeros((M, K))
        for c in range(K):
            denom = np.zeros(M)
            for c2 in range(K):
                denom += (dist[:, c] / dist[:, c2]) ** power
            U[:, c] = 1.0 / denom

        # 数值修正
        U = np.clip(U, 1e-16, None)
        U = U / U.sum(axis=1, keepdims=True)

        # 检查收敛
        change = np.abs(U - U_old).max()
        if change < tol:
            break

    # 计算最终目标函数值
    U_m = U ** m
    J = 0.0
    for c in range(K):
        diff = X - centers[c]
        d2 = (diff * diff).sum(axis=1)
        J += (U_m[:, c] * d2).sum()

    return U, centers, iteration + 1, J


def fcm_segment_t1w(t1w_data, brain_mask, n_classes=3, m=2.0,
                    max_iter=300, tol=1e-6, seed=42):
    """对T1w图像执行FCM软分割

    对brain_mask内的体素进行聚类，输出每个体素的组织隶属度。
    聚类中心按强度排序对齐: CSF(最低) → GM(中间) → WM(最高)

    Args:
        t1w_data: np.ndarray [H,W,D], T1w图像 (已归一化)
        brain_mask: np.ndarray [H,W,D], 脑掩膜 (bool)
        n_classes: int, 组织类别数 (默认3: CSF, GM, WM)
        m: float, 模糊指数
        max_iter: int, 最大迭代次数
        tol: float, 收敛阈值
        seed: int, 随机种子

    Returns:
        Q: np.ndarray [H,W,D,N], 每个体素的组织隶属度 (掩膜外为0)
        centers: np.ndarray [N], 排序后的聚类中心
        info: dict, 包含迭代次数、目标函数值等
    """
    assert t1w_data.shape == brain_mask.shape, \
        f"Shape mismatch: t1w={t1w_data.shape}, mask={brain_mask.shape}"

    vol_shape = t1w_data.shape
    N = n_classes

    # 提取掩膜内体素
    voxels = t1w_data[brain_mask].reshape(-1, 1)  # [M, 1]
    M = voxels.shape[0]

    # 执行FCM
    U, centers_raw, n_iters, J = fcm_fit(
        voxels, N, m=m, max_iter=max_iter, tol=tol, seed=seed
    )

    # 按聚类中心强度排序: CSF(暗) < GM(中) < WM(亮)
    center_values = centers_raw.flatten()  # [N]
    sort_idx = np.argsort(center_values)
    centers_sorted = center_values[sort_idx]
    U_sorted = U[:, sort_idx]  # 重排列隶属度

    # 填充到3D体积
    Q = np.zeros(vol_shape + (N,), dtype=np.float64)
    Q[brain_mask] = U_sorted

    info = {
        'n_iters': n_iters,
        'objective': J,
        'centers_raw': center_values,
        'centers_sorted': centers_sorted,
        'sort_idx': sort_idx,
        'n_voxels': M,
        'm': m,
    }

    return Q, centers_sorted, info
