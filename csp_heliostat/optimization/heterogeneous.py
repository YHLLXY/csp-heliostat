"""
问题三：异构优化（分环分区 + warm-start）。

策略：
  Phase 1: 用问题二的均匀最优解作为初值（warm start）
  Phase 2: 按"径向环"分组，组内共享参数（zoned）
    决策变量 = [(x_t, y_t), (W_k, h_k) for k in 0..n_rings-1]
    算法: 差分进化（DE）— 适合中等维度（~2+2*n_rings）
  Phase 3: 微调 — 在 Phase 2 解附近做局部搜索
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple, Optional

from csp_heliostat.config.constants import (
    LATITUDE_DEG, ALTITUDE_KM, TOWER_HEIGHT_M, RECEIVER_HEIGHT_M,
    RECEIVER_RADIUS_M, FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
    RATED_POWER_MW, MIRROR_W_RANGE, MIRROR_H_RANGE, INSTALL_HEIGHT_RANGE,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.simulation.annual import annual_simulation


PENALTY_FACTOR = 1e6


def _simulate_zoned(params: np.ndarray,
                     n_rings: int,
                     schedule=None,
                     verbose: bool = False) -> Dict:
    """
    用分环参数运行年度仿真。

    params 布局: [x_t, y_t, W_0, h_0, W_1, h_1, ..., W_{K-1}, h_{K-1}]
    共 2 + 2*n_rings 个变量。

    Args:
        params: 分环参数数组
        n_rings: 环数
        schedule: 采样时间表。None = 默认 60 时刻
        verbose: 是否输出进度

    Returns:
        年度仿真结果字典
    """
    x_t, y_t = params[0], params[1]

    # 生成分环布局
    dr = (FIELD_RADIUS_M - EXCLUSION_RADIUS_M) / n_rings
    mirrors = []

    for k in range(n_rings):
        r_k = EXCLUSION_RADIUS_M + (k + 0.5) * dr
        W_k = float(np.clip(params[2 + 2*k], MIRROR_W_RANGE[0], MIRROR_W_RANGE[1]))
        h_k = float(np.clip(params[2 + 2*k + 1], INSTALL_HEIGHT_RANGE[0], INSTALL_HEIGHT_RANGE[1]))
        H_k = W_k  # 正方形镜面

        n_k = max(1, int(np.floor(2.0 * np.pi * r_k / (W_k + 5.0))))
        dtheta = 2.0 * np.pi / n_k

        offset = 0.0
        if k % 2 == 1:
            offset = dtheta / 2.0

        for j in range(n_k):
            theta = offset + j * dtheta
            x = r_k * np.cos(theta)
            y = r_k * np.sin(theta)

            mirrors.append(Mirror(
                x=float(x), y=float(y),
                width=W_k, height=H_k,
                install_height=h_k,
            ))

    mirrors = exclude_zone(mirrors, (float(x_t), float(y_t)), EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)

    if len(mirrors) == 0:
        return {'P_field_avg_MW': 0.0, 'n_mirrors': 0, 'total_area_m2': 0.0}

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


def objective_zoned(params: np.ndarray,
                     n_rings: int,
                     target_power: float = RATED_POWER_MW,
                     schedule=None) -> float:
    """
    分环优化目标函数。

    f = -P_per_area + penalty

    Args:
        params: 分环参数 [x_t, y_t, W_0, h_0, ...]
        n_rings: 环数
        target_power: 目标功率（MW）
        schedule: 采样时间表。None = 默认 60 时刻

    Returns:
        目标函数值（越小越好）
    """
    x_t, y_t = params[0], params[1]

    # 塔位置边界检查
    if abs(x_t) > 120 or abs(y_t) > 120:
        return 1e10

    # 运行仿真
    result = _simulate_zoned(params, n_rings, schedule=schedule)

    P_avg = result['P_field_avg_MW']
    N = result['n_mirrors']
    total_area = result['total_area_m2']

    if N == 0 or total_area < 1.0:
        return 1e10

    P_per_area = P_avg / total_area

    # 功率约束惩罚
    penalty = 0.0
    if P_avg < target_power:
        penalty = PENALTY_FACTOR * (target_power - P_avg) ** 2

    return -P_per_area + penalty


def build_initial_zoned_params(warm_start: Dict,
                                n_rings: int,
                                perturbation: float = 0.1) -> np.ndarray:
    """
    从问题二的最优解构建分环初始参数。

    初始所有环使用相同 W 和 h，加小扰动打破对称性。

    Args:
        warm_start: 问题二的 best_overall 结果
        n_rings: 环数
        perturbation: 扰动幅度（相对比例）

    Returns:
        初始参数数组 [x_t, y_t, W_0, h_0, ...]
    """
    params = warm_start['best_params']  # [x_t, y_t, W, h_inst, n_rings_opt]
    x_t, y_t, W_base, h_base = params[:4]

    init = np.zeros(2 + 2 * n_rings)
    init[0] = x_t
    init[1] = y_t

    np.random.seed(42)
    for k in range(n_rings):
        # 外环比内环稍大（物理直觉：外环接收低角太阳更多，需要更精确）
        W_factor = 1.0 + perturbation * (np.random.random() - 0.5)
        h_factor = 1.0 + perturbation * (np.random.random() - 0.5)
        init[2 + 2*k] = np.clip(W_base * W_factor, *MIRROR_W_RANGE)
        init[2 + 2*k + 1] = np.clip(h_base * h_factor, *INSTALL_HEIGHT_RANGE)

    return init


def solve_problem3(warm_start: Optional[Dict] = None,
                    n_rings: int = None,
                    target_power_MW: float = RATED_POWER_MW,
                    maxiter: int = 50,
                    popsize: int = 10,
                    schedule=None,
                    verbose: bool = True) -> Dict:
    """
    解决问题三：分环异构优化。

    从问题二 warm-start，每环独立参数。

    Args:
        warm_start: 问题二的 best_overall 结果。如果为 None，使用默认值。
        n_rings: 环数。如果为 None，从 warm_start 中读取。
        target_power_MW: 目标功率
        maxiter: DE 最大迭代代数
        popsize: DE 种群大小
        schedule: 采样时间表。None = 默认 60 时刻。传入 noon-only 可用快速模式。
        verbose: 是否输出中间结果

    Returns:
        dict: {
            'best_params': ndarray,  # 最优参数
            'best_P_per_area': float,
            'best_P_avg_MW': float,
            'best_n_mirrors': int,
            'per_ring_params': [(W, h), ...],  # 每环参数
            'convergence_history': list,
        }
    """
    if warm_start is None:
        # 默认 warm-start (无问题二结果时的回退)
        warm_start = {
            'best_params': [0.0, 0.0, 6.0, 4.0, 18],
            'best_P_per_area': 0.0,
            'best_P_avg_MW': 0.0,
            'best_n_mirrors': 0,
        }

    if n_rings is None:
        n_rings = int(warm_start['best_params'][4])

    if verbose:
        print(f"问题三：分环异构优化 (warm-start from P2)")
        print(f"  环数: {n_rings}")
        print(f"  DE: maxiter={maxiter}, popsize={popsize}")
        print(f"  初值: x_t={warm_start['best_params'][0]:.1f}, "
              f"y_t={warm_start['best_params'][1]:.1f}, "
              f"W={warm_start['best_params'][2]:.2f}m, "
              f"h={warm_start['best_params'][3]:.2f}m")

    # 构建变量边界
    bounds = [
        (-100.0, 100.0),   # x_t
        (-100.0, 100.0),   # y_t
    ]
    for _ in range(n_rings):
        bounds.append(MIRROR_W_RANGE)       # W_k
        bounds.append(INSTALL_HEIGHT_RANGE) # h_k

    # 构建初始种群
    init_params = build_initial_zoned_params(warm_start, n_rings)

    # 在初始解附近生成种群
    init_pop = [init_params]
    np.random.seed(42)
    for _ in range(popsize - 1):
        perturbed = init_params.copy()
        perturbed[0] += np.random.uniform(-10, 10)      # x_t
        perturbed[1] += np.random.uniform(-10, 10)      # y_t
        for k in range(n_rings):
            perturbed[2 + 2*k] *= np.random.uniform(0.85, 1.15)
            perturbed[2 + 2*k + 1] *= np.random.uniform(0.85, 1.15)
        # clip to bounds
        perturbed[0] = np.clip(perturbed[0], -100, 100)
        perturbed[1] = np.clip(perturbed[1], -100, 100)
        perturbed[2::2] = np.clip(perturbed[2::2], *MIRROR_W_RANGE)
        perturbed[3::2] = np.clip(perturbed[3::2], *INSTALL_HEIGHT_RANGE)
        init_pop.append(perturbed)

    # 收敛历史记录
    convergence_history = []
    _gen_counter = [0]  # 用列表避免闭包问题

    def _callback(xk, convergence):
        """DE 每代回调：记录最优目标值"""
        gen = _gen_counter[0]
        _gen_counter[0] += 1
        # 在当前最优解处评估目标
        best_f = objective_zoned(xk, n_rings, target_power_MW, schedule=schedule)
        convergence_history.append({
            'generation': gen,
            'best_fitness': float(best_f),
            'convergence': float(convergence),
        })
        if verbose:
            P_per_area = -best_f if best_f < 1e8 else 0.0
            print(f"  [DE] gen {gen:3d}: best_f={best_f:.4f}, "
                  f"P/A={P_per_area*1e6:.2f} W/m2, conv={convergence:.6f}")

    # 差分进化
    result = differential_evolution(
        lambda p: objective_zoned(p, n_rings, target_power_MW, schedule=schedule),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        seed=42,
        init=np.array(init_pop[:popsize]),
        polish=True,
        callback=_callback,
    )

    # 最终精确仿真（使用传入的 schedule，None=完整60时刻）
    sim_result = _simulate_zoned(result.x, n_rings, schedule=schedule)

    # 提取每环参数
    per_ring = []
    for k in range(n_rings):
        per_ring.append((
            float(result.x[2 + 2*k]),
            float(result.x[2 + 2*k + 1]),
        ))

    if verbose:
        print(f"\n问题三 结果:")
        print(f"  x_t={result.x[0]:.1f}, y_t={result.x[1]:.1f}")
        print(f"  P_avg = {sim_result['P_field_avg_MW']:.2f} MW")
        print(f"  P_per_area = {sim_result['P_field_avg_MW']/sim_result['total_area_m2']*1e6:.2f} W/m2")
        print(f"  镜面数: {sim_result['n_mirrors']}")
        print(f"  总反射面积: {sim_result['total_area_m2']:.0f} m2")
        print(f"  首环: W={per_ring[0][0]:.2f}, h={per_ring[0][1]:.2f}")
        print(f"  末环: W={per_ring[-1][0]:.2f}, h={per_ring[-1][1]:.2f}")

    return {
        'best_params': result.x,
        'best_P_per_area': -result.fun if result.fun < 1e8 else 0.0,
        'best_P_avg_MW': sim_result['P_field_avg_MW'],
        'best_n_mirrors': sim_result['n_mirrors'],
        'per_ring_params': per_ring,
        'convergence_history': convergence_history,
    }