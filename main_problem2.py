"""
问题二：均匀参数优化

决策变量：塔位置 (x_t, y_t)、镜面边长 W=H、安装高度 h_inst、环数 n_rings
目标：max P_per_area（单位面积功率），约束 P_avg >= 60 MW

算法：外层枚举 n_rings，内层 scipy.optimize.differential_evolution

注意：完整年度仿真耗时较长，默认使用"快速模式"（仅正午12时刻）
进行优化，最终用完整 60 时刻验证最优解。
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
    RATED_POWER_MW, FIELD_RADIUS_M, EXCLUSION_RADIUS_M, REFLECTIVITY,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.constraints import total_mirrors_count, total_reflective_area
from csp_heliostat.simulation.annual import annual_simulation
from csp_heliostat.optimization.uniform import solve_problem2


def main():
    print("=" * 60)
    print("问题二：均匀参数优化")
    print("=" * 60)
    print()

    # ---- 快速模式说明 ----
    print("优化策略：")
    print("  外层枚举 n_rings ∈ [5,8,10,12,15,18,22]")
    print("  内层 DE: popsize=8, maxiter=20")
    print("  优化阶段使用 fast noon-only 12时刻 (~5x speed)")
    print("  最终验证使用完整 60 时刻")
    print("  预计总耗时：~10-20 分钟")
    print()

    # ---- 运行优化 ----
    t0 = time.time()

    result = solve_problem2(
        target_power_MW=RATED_POWER_MW,
        n_rings_candidates=[5, 8, 10, 12, 15, 18, 22],
        maxiter=20,
        popsize=8,
        verbose=True,
    )

    elapsed = time.time() - t0

    # ---- 输出最终结果 ----
    print()
    print("=" * 60)
    print("问题二 最终最优解")
    print("=" * 60)

    params = result['best_params']
    x_t, y_t, W, h_inst, n_rings = params

    print(f"  塔位置:       x_t = {x_t:.2f} m, y_t = {y_t:.2f} m")
    print(f"  镜面边长:     W = H = {W:.2f} m")
    print(f"  安装高度:     h_inst = {h_inst:.2f} m")
    print(f"  环数:         n_rings = {n_rings}")
    print()
    print(f"  (快速评估) P_per_area = {result['best_P_per_area_fast']*1e6:.2f} W/m2")
    print(f"  (快速评估) P_avg      = {result['best_P_fast_MW']:.2f} MW")
    print(f"  镜面总数:             = {result['best_n_mirrors']}")
    print(f"  优化总耗时:            = {elapsed:.1f} s ({elapsed/60:.1f} min)")

    # ---- 验证最优解 ----
    print()
    print("=" * 60)
    print("验证：用最优参数生成镜场并做完整仿真")
    print("=" * 60)

    mirrors = radial_layout(
        n_rings=n_rings, W=W, H=W, install_h=h_inst,
        R_inner=EXCLUSION_RADIUS_M, R_outer=FIELD_RADIUS_M,
    )
    mirrors = exclude_zone(mirrors, (x_t, y_t), EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0, 0), FIELD_RADIUS_M)

    print(f"  镜面数: {len(mirrors)}")

    verify_result = annual_simulation(
        mirrors,
        tower_xy=(x_t, y_t),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        verbose=True,
    )

    print()
    print(f"  验证 P_avg = {verify_result['P_field_avg_MW']:.2f} MW")
    print(f"  验证 eta   = {verify_result['eta_field_avg']:.6f}")

    verify_area = float(total_reflective_area(mirrors))
    # ---- 保存结果 ----
    output = {
        'problem': 2,
        'params': {'x_t': float(x_t), 'y_t': float(y_t), 'W': float(W),
                    'h_inst': float(h_inst), 'n_rings': int(n_rings)},
        'P_per_area_W_m2': float(verify_result['P_field_avg_MW'] / verify_area * 1e6) if verify_area > 0 else 0.0,
        'P_avg_MW': float(verify_result['P_field_avg_MW']),
        'eta_avg': float(verify_result['eta_field_avg']),
        'n_mirrors': len(mirrors),
        'total_area_m2': verify_area,
    }

    os.makedirs('outputs/data', exist_ok=True)
    with open('outputs/data/problem2_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至 outputs/data/problem2_result.json")
    print("问题二完成")

    return result


if __name__ == '__main__':
    main()