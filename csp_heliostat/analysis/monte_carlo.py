"""
改进 #7：蒙特卡洛置信带。

对阴影遮挡算法的随机栅格采样做多次重复，量化随机性对结果的影响。
"""

import numpy as np
from typing import Dict, List

from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
)
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.simulation.annual import annual_simulation


def run_monte_carlo(
    mirrors: List[Mirror],
    tower_xy: tuple = (0.0, 0.0),
    n_runs: int = 10,
    seed_start: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    对同一镜场配置做 n_runs 次独立仿真，每次使用不同随机种子。

    Args:
        mirrors: 定日镜列表
        tower_xy: 塔位置
        n_runs: 重复次数
        seed_start: 起始种子值
        verbose: 是否输出进度

    Returns:
        dict: {
            'P_mean_MW': float,
            'P_std_MW': float,
            'eta_mean': float,
            'eta_std': float,
            'cv_pct': float,           # 变异系数 = std/mean * 100
            'all_P_MW': [float, ...],
            'all_eta': [float, ...],
        }
    """
    all_P = []
    all_eta = []

    for run_idx in range(n_runs):
        seed = seed_start + run_idx
        np.random.seed(seed)

        if verbose:
            print(f"  MC run {run_idx + 1}/{n_runs} (seed={seed})...")

        result = annual_simulation(
            mirrors,
            tower_xy=tower_xy,
            tower_height=TOWER_HEIGHT_M,
            receiver_height=RECEIVER_HEIGHT_M,
            receiver_radius=RECEIVER_RADIUS_M,
            verbose=False,
        )

        all_P.append(result['P_field_avg_MW'])
        all_eta.append(result['eta_field_avg'])

    P_arr = np.array(all_P)
    eta_arr = np.array(all_eta)

    P_mean = float(P_arr.mean())
    P_std = float(P_arr.std(ddof=1))
    eta_mean = float(eta_arr.mean())
    eta_std = float(eta_arr.std(ddof=1))
    cv_pct = float(P_std / P_mean * 100) if P_mean > 0 else 0.0

    return {
        'P_mean_MW': P_mean,
        'P_std_MW': P_std,
        'eta_mean': eta_mean,
        'eta_std': eta_std,
        'cv_pct': cv_pct,
        'all_P_MW': all_P,
        'all_eta': all_eta,
    }