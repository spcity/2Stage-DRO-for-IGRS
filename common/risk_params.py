"""
风险参数定义模块
================
定义两个实验的风险成本向量C和基准距离矩阵D。
参数基于临床知识和MRI信号特性设计。

BrainWeb (N=3): CSF, GM, WM — 神经外科手术场景
BraTS   (N=4): BG, NCR/NET, ED, ET — 肿瘤手术场景
"""

import numpy as np


def get_brainweb_params():
    """返回BrainWeb实验的风险参数 (N=3)

    组织类别: [CSF, GM, WM]
    临床场景: 神经外科手术路径规划

    Returns:
        params: dict, 包含 N, tissue_names, C, D
    """
    N = 3
    tissue_names = ['CSF', 'GM', 'WM']

    # 风险成本向量 C
    # CSF:  低风险, 脑脊液空间可通行 (S3)
    # GM:   高风险, 灰质含功能区 (S2)
    # WM:   中等风险, 白质纤维束 (S3)
    C = np.array([0.1, 0.7, 0.3])

    # 基准距离矩阵 D ∈ R^{3×3}
    # d_ij: 将组织j误判为组织i的代价
    # 基于MRI T1w信号相似度 + 解剖邻近性
    D = np.array([
        # CSF   GM    WM
        [0.0,  0.5,  0.8],   # CSF: 与GM邻近(脑沟), 与WM较远
        [0.5,  0.0,  0.3],   # GM:  与WM紧邻(皮层), 与CSF次之
        [0.8,  0.3,  0.0],   # WM:  与GM紧邻, 与CSF远
    ])

    return {
        'N': N,
        'tissue_names': tissue_names,
        'C': C,
        'D': D,
    }


def get_brats_params():
    """返回BraTS2019实验的风险参数 (N=4)

    组织类别: [BG, NCR/NET, ED, ET]
    临床场景: 脑肿瘤手术路径规划

    Returns:
        params: dict, 包含 N, tissue_names, C, D
    """
    N = 4
    tissue_names = ['BG', 'NCR/NET', 'ED', 'ET']

    # 风险成本向量 C (肿瘤手术场景)
    C = np.array([0.0,    # BG:      安全区 (S4)
                  0.5,    # NCR/NET: 中等风险 (S2), 坏死核心
                  0.3,    # ED:      较低风险 (S3), 水肿可恢复
                  1.0])   # ET:      最高风险 (S1), 活跃肿瘤/禁区

    # 基准距离矩阵 D ∈ R^{4×4}
    D = np.array([
        #  BG    NCR   ED    ET
        [0.0,  0.8,  0.5,  1.0],   # BG:  距所有肿瘤组织较远
        [0.8,  0.0,  0.4,  0.3],   # NCR: 与ET紧邻(核心区), 与ED次之
        [0.5,  0.4,  0.0,  0.5],   # ED:  水肿包绕肿瘤, 与NCR/ET均相邻
        [1.0,  0.3,  0.5,  0.0],   # ET:  活跃肿瘤, 与NCR紧邻
    ])

    return {
        'N': N,
        'tissue_names': tissue_names,
        'C': C,
        'D': D,
    }


def compute_lambda(D, beta=1.0):
    """计算物理锚定半径 λ

    λ = β · min(D[D>0])

    Args:
        D: np.ndarray [N,N], 基准距离矩阵
        beta: float, 比例系数, ∈ [0.5, 2.0]

    Returns:
        lam: float, 物理锚定半径
    """
    nonzero_dists = D[D > 0]
    assert len(nonzero_dists) > 0, "D has no non-zero elements"
    return beta * nonzero_dists.min()
