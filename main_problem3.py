"""
问题三：异构优化（分环差异化参数）

策略：
  Phase 1: warm-start 问题二的最优解
  Phase 2: 分环优化 — 每环独立 W_k, h_k
  Phase 3: 验证最终结果

必须用问题二的输出作为输入！
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows GBK encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import time
import json

from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    RATED_POWER_MW, FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.field.constraints import total_reflective_area
from csp_heliostat.simulation.annual import annual_simulation
from csp_heliostat.optimization.heterogeneous import solve_problem3, _simulate_zoned


def load_problem2_result(path: str = 'outputs/data/problem2_result.json') -> dict:
    """加载问题二的结果作为 warm-start。"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        params = data['params']
        warm_start = {
            'best_params': [
                params['x_t'], params['y_t'], params['W'],
                params['h_inst'], params['n_rings'],
            ],
            'best_P_per_area': data['P_per_area_W_m2'],
            'best_P_avg_MW': data['P_avg_MW'],
            'best_n_mirrors': data['n_mirrors'],
        }
        print(f"已加载问题二结果: x_t={params['x_t']:.1f}, y_t={params['y_t']:.1f}, "
              f"W={params['W']:.2f}, h_inst={params['h_inst']:.2f}, "
              f"n_rings={params['n_rings']}")
        return warm_start
    else:
        print("未找到问题二结果文件，使用默认 warm-start")
        return None


def main():
    print("=" * 60)
    print("问题三：分环异构优化 (Warm-Start from Problem 2)")
    print("=" * 60)
    print()

    # ---- 加载问题二结果 ----
    warm_start = load_problem2_result()

    if warm_start is None:
        # 回退：手动指定
        print("使用默认初始参数")
        warm_start = {
            'best_params': [0.0, 0.0, 5.5, 4.0, 15],
            'best_P_per_area': 0.0,
            'best_P_avg_MW': 0.0,
            'best_n_mirrors': 0,
        }

    n_rings = int(warm_start['best_params'][4])
    print(f"  分环数: {n_rings}")
    print(f"  决策变量维度: {2 + 2*n_rings}")
    print()

    # ---- 运行分环优化 ----
    print("优化策略：")
    print(f"  DE: popsize=8, maxiter=30")
    print(f"  变量: 塔位置(2) + 每环(W_k, h_k) × {n_rings} = {2+2*n_rings} 维")
    print(f"  初始种群来自问题二最优解 + 扰动")
    print()

    t0 = time.time()

    result = solve_problem3(
        warm_start=warm_start,
        n_rings=n_rings,
        target_power_MW=RATED_POWER_MW,
        maxiter=30,
        popsize=8,
        verbose=True,
    )

    elapsed = time.time() - t0

    # ---- 输出每环参数 ----
    print()
    print("=" * 60)
    print("分环参数详情")
    print("=" * 60)
    print(f"{'环号':>6} {'r (m)':>8} {'W (m)':>8} {'h (m)':>8}")
    print("-" * 35)
    dr = (FIELD_RADIUS_M - EXCLUSION_RADIUS_M) / n_rings
    for k, (w, h) in enumerate(result['per_ring_params']):
        r_k = EXCLUSION_RADIUS_M + (k + 0.5) * dr
        print(f"{k:>6} {r_k:>8.1f} {w:>8.2f} {h:>8.2f}")

    # ---- 对比问题二 ----
    print()
    print("=" * 60)
    print("问题二 vs 问题三 对比")
    print("=" * 60)
    if warm_start:
        print(f"  问题二 P_avg: {warm_start['best_P_avg_MW']:.2f} MW")
        print(f"  问题二 P_area: {warm_start['best_P_per_area']*1e6:.2f} W/m2")
    print(f"  问题三 P_avg: {result['best_P_avg_MW']:.2f} MW")
    print(f"  问题三 P_area: {result['best_P_per_area']*1e6:.2f} W/m2")
    print(f"  优化耗时: {elapsed:.1f} s ({elapsed/60:.1f} min)")

    # ---- 保存结果 ----
    output = {
        'problem': 3,
        'warm_start_from_problem2': warm_start is not None,
        'params': {
            'x_t': float(result['best_params'][0]),
            'y_t': float(result['best_params'][1]),
            'n_rings': n_rings,
            'per_ring': [
                {'ring': k, 'W': float(w), 'h': float(h)}
                for k, (w, h) in enumerate(result['per_ring_params'])
            ],
        },
        'P_per_area_W_m2': float(result['best_P_per_area']),
        'P_avg_MW': float(result['best_P_avg_MW']),
        'n_mirrors': int(result['best_n_mirrors']),
    }

    os.makedirs('outputs/data', exist_ok=True)
    with open('outputs/data/problem3_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至 outputs/data/problem3_result.json")
    print("问题三完成")

    return result


if __name__ == '__main__':
    main()