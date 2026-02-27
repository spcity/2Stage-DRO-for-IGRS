"""
两阶段分布鲁棒风险量化 - 公共模块
==================================
提供LP求解器、Wasserstein距离、FCM软分割、风险参数定义等通用组件。
所有模块独立于具体数据集（BrainWeb / BraTS），通过参数N适配不同实验。

运行环境: conda activate torch_m
"""

from .lp_solvers import solve_stage1_lp, solve_stage2_lp
from .wasserstein import wasserstein_distance
from .risk_params import get_brainweb_params, get_brats_params
