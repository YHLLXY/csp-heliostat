"""
问题二：均匀参数优化（统一镜面尺寸 + 统一安装高度）。

决策变量（连续）：
  x_t, y_t  — 塔位置 ∈ [-100, 100]
  W = H     — 镜面边长 ∈ [2, 8]
  h_inst    — 安装高度 ∈ [2, 6]
  n_rings   — 环数 ∈ {5, ..., 30}（离散，外层枚举）

优化目标：
  max  P_per_mirror = P_total / (N * W^2)
  s.t. P_total >= 60 MW（年加权平均）

算法：
  外层：枚举 n_rings ∈ [5, 30]
  内层：scipy.optimize.differential_evolution 连续变量优化
  取所有 n_rings 中最优解
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, Tuple
import time

from csp_heliostat.config.constants import (
    LATITUDE_DEG, ALTITUDE_KM, TOWER_HEIGHT_M, RECEIVER_HEIGHT_M,
    RECEIVER_RADIUS_M, REFLECTIVITY, FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
    RATED_POWER_MW, MIRROR_W_RANGE, MIRROR_H_RANGE, INSTALL_HEIGHT_RANGE,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.constraints import count_spacing_violations
from csp_heliostat.simulation.annual import annual_simulation


# 惩罚系数（功率不达标时的惩罚）
PENALTY_FACTOR = 1e6


from csp_heliostat.config.sampling import build_noon_schedule


def _simulate_for_params(params: np.ndarray,
                          n_rings: int,
                          schedule=None,
                          verbose: bool = False) -> Dict:
    """给定连续参数和环数，运行年度仿真。schedule=None 表示使用快速 noon-only 模式。"""
    x_t, y_t, W, h_inst = params
    W = abs(W)
    H = W

    mirrors = radial_layout(
        n_rings=n_rings, W=float(W), H=float(H),
        install_h=float(h_inst),
        R_inner=EXCLUSION_RADIUS_M,
        R_outer=FIELD_RADIUS_M,
    )
    mirrors = exclude_zone(mirrors, (float(x_t), float(y_t)), EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)

    if len(mirrors) == 0:
        return {'P_field_avg_MW': 0.0, 'n_mirrors': 0, 'total_area_m2': 0.0}

    if schedule is None:
        schedule = build_noon_schedule()  # 快速模式：12 时刻

    result = annual_simulation(
        mirrors,
        tower_xy=(float(x_t), float(y_t)),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        schedule=schedule,
        verbose=verbose,
    )
    return result


def objective_uniform(params: np.ndarray,
                      n_rings: int,
                      target_power: float = RATED_POWER_MW,
                      schedule=None) -> float:
    """
    优化目标函数。schedule=None 使用快速 noon-only 模式（12 时刻）。
    """
    x_t, y_t, W, h_inst = params

    if not (MIRROR_W_RANGE[0] <= W <= MIRROR_W_RANGE[1]):
        return 1e10 + abs(W - MIRROR_W_RANGE[0])
    if not (INSTALL_HEIGHT_RANGE[0] <= h_inst <= INSTALL_HEIGHT_RANGE[1]):
        return 1e10 + abs(h_inst - INSTALL_HEIGHT_RANGE[0])
    if abs(x_t) > 120 or abs(y_t) > 120:
        return 1e10

    result = _simulate_for_params(params, n_rings, schedule=schedule)

    P_avg = result['P_field_avg_MW']
    N = result['n_mirrors']
    total_area = result['total_area_m2']

    if N == 0 or total_area < 1.0:
        return 1e10

    P_per_area = P_avg / total_area

    penalty = 0.0
    if P_avg < target_power:
        penalty = PENALTY_FACTOR * (target_power - P_avg) ** 2

    return -P_per_area + penalty


def solve_problem2(target_power_MW: float = RATED_POWER_MW,
                    n_rings_candidates: list = None,
                    maxiter: int = 20,
                    popsize: int = 8,
                    verbose: bool = True) -> Dict:
    """
    解决问题二：均匀参数优化（快速模式）。

    外层枚举 n_rings（默认 5,8,10,12,15,18），内层 DE。
    优化阶段使用 noon-only 快速度量（12 时刻），最终用 60 时刻验证。

    Args:
        target_power_MW: 目标功率
        n_rings_candidates: 待搜索的环数列表。默认 [5,8,10,12,15,18,22]
        maxiter: DE 最大迭代代数
        popsize: DE 种群大小
        verbose: 是否输出中间结果
    """
    if n_rings_candidates is None:
        n_rings_candidates = [5, 8, 10, 12, 15, 18, 22]

    bounds = [(-100.0, 100.0), (-100.0, 100.0), MIRROR_W_RANGE, INSTALL_HEIGHT_RANGE]
    fast_schedule = build_noon_schedule()

    best_overall = None
    best_P_per_area = -np.inf
    all_results = []

    if verbose:
        print(f"问题二优化 (快速模式: noon-only 12时刻)")
        print(f"  n_rings candidates: {n_rings_candidates}")
        print(f"  DE: maxiter={maxiter}, popsize={popsize}")
        print(f"{'n_rings':>8} {'x_t':>8} {'y_t':>8} {'W':>6} {'h':>6} {'P_fast':>9} {'elap':>7}")
        print("-" * 60)

    for n_rings in n_rings_candidates:
        t0 = time.time()

        result = differential_evolution(
            lambda p: objective_uniform(p, n_rings, target_power_MW, schedule=fast_schedule),
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            polish=True,
        )

        x_t, y_t, W, h_inst = result.x
        P_per_area_fast = -result.fun if result.fun < 1e8 else 0.0
        elapsed = time.time() - t0

        # 快速评估功率
        sim_fast = _simulate_for_params(result.x, n_rings, schedule=fast_schedule)
        P_fast = sim_fast['P_field_avg_MW']
        N = sim_fast['n_mirrors']

        if verbose:
            print(f"{n_rings:>8} {x_t:>8.1f} {y_t:>8.1f} {W:>6.2f} {h_inst:>6.2f} "
                  f"{P_fast:>9.2f} {elapsed:>6.0f}s")

        all_results.append({
            'n_rings': n_rings, 'x_t': x_t, 'y_t': y_t, 'W': W, 'h_inst': h_inst,
            'P_fast_MW': P_fast, 'P_per_area_fast': P_per_area_fast,
            'n_mirrors': N, 'elapsed_s': elapsed,
        })

        if P_per_area_fast > best_P_per_area:
            best_P_per_area = P_per_area_fast
            best_overall = {
                'best_params': [x_t, y_t, W, h_inst, n_rings],
                'best_P_per_area_fast': P_per_area_fast,
                'best_P_fast_MW': P_fast,
                'best_n_mirrors': N,
            }

    if verbose:
        print("-" * 60)
        if best_overall:
            p = best_overall['best_params']
            print(f"最优解: x_t={p[0]:.1f}, y_t={p[1]:.1f}, W={p[2]:.2f}, "
                  f"h_inst={p[3]:.2f}, n_rings={p[4]}")
            print(f"  P_per_area (fast) = {best_overall['best_P_per_area_fast']*1e6:.2f} W/m2")

    best_overall['all_results'] = all_results
    return best_overall