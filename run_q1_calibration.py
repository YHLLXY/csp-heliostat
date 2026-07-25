"""
Q1 步骤1：校准 noon-only 与 full-60 的功率关系。

在 n_rings=23, 25, 27 三个点采样，确定 noon→full 的转换系数 k。
输出: outputs/data/problem3_calibration.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.simulation.annual import annual_simulation
from csp_heliostat.config.sampling import build_noon_schedule, build_sampling_schedule
from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
)

def run_calibration():
    """采样 n_rings=23, 25, 27 确定校准系数"""
    print("=" * 60)
    print("Q1 步骤1：校准 noon-only → full-60 功率关系")
    print("=" * 60)

    noon_schedule = build_noon_schedule()
    full_schedule = build_sampling_schedule()

    results = []
    for n_rings in [23, 25, 27]:
        # 均一布局：W=6, h=3
        mirrors = radial_layout(
            n_rings=n_rings, W=6.0, H=6.0, install_h=3.0,
            R_inner=EXCLUSION_RADIUS_M, R_outer=FIELD_RADIUS_M,
        )
        mirrors = exclude_zone(mirrors, (0.0, 0.0), EXCLUSION_RADIUS_M)
        mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)

        print(f"\nn_rings={n_rings}, N={len(mirrors)}")

        # Noon-only (12 时刻)
        r_noon = annual_simulation(
            mirrors, tower_xy=(0.0, 0.0),
            tower_height=TOWER_HEIGHT_M,
            receiver_height=RECEIVER_HEIGHT_M,
            receiver_radius=RECEIVER_RADIUS_M,
            schedule=noon_schedule, verbose=False,
        )
        P_noon = r_noon['P_field_avg_MW']
        eta_noon = r_noon['eta_field_avg']

        # Full-60 (60 时刻)
        r_full = annual_simulation(
            mirrors, tower_xy=(0.0, 0.0),
            tower_height=TOWER_HEIGHT_M,
            receiver_height=RECEIVER_HEIGHT_M,
            receiver_radius=RECEIVER_RADIUS_M,
            schedule=full_schedule, verbose=False,
        )
        P_full = r_full['P_field_avg_MW']
        eta_full = r_full['eta_field_avg']

        ratio = P_full / P_noon if P_noon > 0 else 0.0
        results.append({
            'n_rings': n_rings,
            'P_noon_MW': round(P_noon, 4),
            'P_full_MW': round(P_full, 4),
            'eta_noon': round(eta_noon, 6),
            'eta_full': round(eta_full, 6),
            'ratio': round(ratio, 6),
        })
        print(f"  P_noon={P_noon:.4f} MW, P_full={P_full:.4f} MW, ratio={ratio:.6f}")

    # 计算平均校准系数
    ratios = [r['ratio'] for r in results]
    mean_ratio = np.mean(ratios)
    results.append({
        'n_rings': 'mean',
        'P_noon_MW': '',
        'P_full_MW': '',
        'eta_noon': '',
        'eta_full': '',
        'ratio': round(mean_ratio, 6),
    })

    df = pd.DataFrame(results)
    os.makedirs('outputs/data', exist_ok=True)
    df.to_csv('outputs/data/problem3_calibration.csv', index=False)

    print(f"\n校准系数 CALIBRATION_RATIO = {mean_ratio:.6f}")
    print(f"即 P_full ≈ {mean_ratio:.4f} × P_noon")
    print(f"约束阈值: P_noon >= {60.0/mean_ratio:.2f} MW（等效 P_full >= 60 MW）")
    print(f"结果已保存到 outputs/data/problem3_calibration.csv")

    return mean_ratio

if __name__ == '__main__':
    k = run_calibration()