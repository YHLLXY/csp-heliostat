"""
改进 #1：P/A 敏感性分析。

在最优解附近做单变量扫描，验证解的鲁棒性。
扫描变量：x_t, y_t, W, h, n_rings
"""

import numpy as np
import time
from typing import Dict, List, Optional

from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.constraints import total_reflective_area
from csp_heliostat.simulation.annual import annual_simulation
from csp_heliostat.config.sampling import build_noon_schedule


def _build_and_simulate(x_t, y_t, W, h, n_rings, schedule):
    """构建镜场并运行仿真。"""
    mirrors = radial_layout(
        n_rings=n_rings, W=float(W), H=float(W),
        install_h=float(h),
        R_inner=EXCLUSION_RADIUS_M,
        R_outer=FIELD_RADIUS_M,
    )
    mirrors = exclude_zone(mirrors, (float(x_t), float(y_t)), EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)

    if len(mirrors) == 0:
        return {'P_field_avg_MW': 0.0, 'eta_field_avg': 0.0,
                'total_area_m2': 0.0, 'n_mirrors': 0}

    result = annual_simulation(
        mirrors,
        tower_xy=(float(x_t), float(y_t)),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        schedule=schedule,
        verbose=False,
    )
    return result


def run_sensitivity(
    best_params: List[float],
    n_rings: int,
    use_fast: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    对最优解做单变量敏感性扫描。

    Args:
        best_params: [x_t, y_t, W, h]（最优连续变量）
        n_rings: 最优环数
        use_fast: 是否用 noon-only 快速模式（推荐 True）
        verbose: 是否输出进度

    Returns:
        dict: 每个变量的扫描结果
    """
    x_t0, y_t0, W0, h0 = best_params

    schedule = build_noon_schedule() if use_fast else None

    # 扫描定义
    scan_defs = {
        'x_t': {
            'values': np.linspace(-50, 50, 11),
            'fixed': {'y_t': y_t0, 'W': W0, 'h': h0},
        },
        'y_t': {
            'values': np.linspace(-50, 50, 11),
            'fixed': {'x_t': x_t0, 'W': W0, 'h': h0},
        },
        'W': {
            'values': np.linspace(5.0, 7.0, 11),
            'fixed': {'x_t': x_t0, 'y_t': y_t0, 'h': h0},
        },
        'h': {
            'values': np.linspace(2.0, 4.0, 11),
            'fixed': {'x_t': x_t0, 'y_t': y_t0, 'W': W0},
        },
        'n_rings': {
            'values': np.array([22, 23, 24, 25, 26, 27, 28], dtype=int),
            'fixed': {'x_t': x_t0, 'y_t': y_t0, 'W': W0, 'h': h0},
        },
    }

    results = {}

    for var_name, scan_def in scan_defs.items():
        if verbose:
            print(f"\n扫描变量: {var_name} ({len(scan_def['values'])} 个点)")

        points = []
        for val in scan_def['values']:
            t0 = time.time()

            if var_name == 'n_rings':
                result = _build_and_simulate(
                    x_t0, y_t0, W0, h0, int(val), schedule
                )
            else:
                kwargs = dict(scan_def['fixed'])
                kwargs[var_name] = float(val)
                result = _build_and_simulate(
                    kwargs['x_t'], kwargs['y_t'],
                    kwargs['W'], kwargs['h'],
                    n_rings, schedule,
                )

            elapsed = time.time() - t0
            P_avg = result['P_field_avg_MW']
            total_area = result.get('total_area_m2', 0.0)
            P_per_area = (P_avg / total_area * 1e6) if total_area > 0 else 0.0

            points.append({
                'value': float(val) if var_name != 'n_rings' else int(val),
                'P_avg_MW': P_avg,
                'P_per_area_W_m2': P_per_area,
                'eta_avg': result['eta_field_avg'],
                'n_mirrors': result['n_mirrors'],
                'elapsed_s': elapsed,
            })

            if verbose:
                flag = " [OK]" if P_avg >= 60.0 else " [!!]"
                print(f"  {var_name}={val:7.2f}: P={P_avg:7.2f} MW, "
                      f"P/A={P_per_area:7.2f} W/m2, N={result['n_mirrors']}{flag}")

        results[var_name] = {
            'values': scan_def['values'].tolist() if var_name != 'n_rings'
                      else [int(v) for v in scan_def['values']],
            'points': points,
        }

    return results