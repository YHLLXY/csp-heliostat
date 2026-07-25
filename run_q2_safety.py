"""
Q2：安全裕度测试 — n_rings=25,26,27 四种加权方法验证。

运行完整60采样，对每种配置做4种加权对照。
输出:
  - outputs/data/weighting_comparison_safe.csv
  - outputs/data/problem2_result_safe.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import pandas as pd
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.simulation.annual import annual_simulation
from csp_heliostat.config.sampling import build_sampling_schedule
from csp_heliostat.analysis.weighting import compare_weighting_methods
from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M, RATED_POWER_MW,
)


def run_safety_margin():
    """测试 n_rings=25,26,27 在各种加权下是否达标"""
    print("=" * 60)
    print("Q2：安全裕度测试 — 4种加权方法 × 3种环数")
    print("=" * 60)

    full_schedule = build_sampling_schedule()
    all_rows = []

    for n_rings in [25, 26, 27]:
        print(f"\nn_rings={n_rings}:")
        mirrors = radial_layout(
            n_rings=n_rings, W=6.0, H=6.0, install_h=3.0,
            R_inner=EXCLUSION_RADIUS_M, R_outer=FIELD_RADIUS_M,
        )
        mirrors = exclude_zone(mirrors, (0.0, 0.0), EXCLUSION_RADIUS_M)
        mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)
        print(f"  N = {len(mirrors)}")

        # 完整 60 采样
        result = annual_simulation(
            mirrors, tower_xy=(0.0, 0.0),
            tower_height=TOWER_HEIGHT_M,
            receiver_height=RECEIVER_HEIGHT_M,
            receiver_radius=RECEIVER_RADIUS_M,
            schedule=full_schedule, verbose=False,
        )

        # 四种加权
        weighting = compare_weighting_methods(result)

        for method, label, P, eta in zip(
            weighting['methods'], weighting['labels'],
            weighting['P_avg_MW'], weighting['eta_avg']
        ):
            satisfied = "Yes" if P >= RATED_POWER_MW else "No"
            all_rows.append({
                'n_rings': n_rings,
                'method': label,
                'P_avg_MW': round(P, 4),
                'eta_avg': round(eta, 6),
                'satisfied': satisfied,
            })
            status = "PASS" if satisfied == "Yes" else "FAIL"
            print(f"  {label:20s}: P={P:.4f} MW, eta={eta:.6f} -> {status}")

    # 保存
    os.makedirs('outputs/data', exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv('outputs/data/weighting_comparison_safe.csv', index=False)
    print(f"\n加权对照表已保存: outputs/data/weighting_comparison_safe.csv")

    # 分析：找到"4种方法全达标"的最小 n_rings
    print(f"\n{'='*40}")
    print("安全裕度分析")
    print(f"{'='*40}")
    best_n = None
    for n_rings in [25, 26, 27]:
        subset = df[df['n_rings'] == n_rings]
        all_ok = all(subset['satisfied'] == 'Yes')
        min_P = subset['P_avg_MW'].min()
        print(f"  n_rings={n_rings}: all_pass={all_ok}, min_P={min_P:.2f} MW")
        if all_ok and best_n is None:
            best_n = n_rings

    if best_n:
        print(f"\n[OK] 推荐 n_rings={best_n}：4 种加权方法全达标")
    else:
        print(f"\n[WARN] n_rings=27 仍未全达标，需论证 sin α_s 加权的物理依据")

    # 获取选定方案详情
    chosen_n = best_n if best_n else 25
    chosen_subset = df[df['n_rings'] == chosen_n]

    # 重新跑一次获取完整指标
    mirrors = radial_layout(
        n_rings=chosen_n, W=6.0, H=6.0, install_h=3.0,
        R_inner=EXCLUSION_RADIUS_M, R_outer=FIELD_RADIUS_M,
    )
    mirrors = exclude_zone(mirrors, (0.0, 0.0), EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)
    result = annual_simulation(
        mirrors, tower_xy=(0.0, 0.0),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        schedule=full_schedule, verbose=False,
    )
    P_chosen = result['P_field_avg_MW']
    weighting_chosen = compare_weighting_methods(result)

    result_json = {
        'problem': 2,
        'type': 'uniform_safe_margin',
        'params': {
            'x_t': 0, 'y_t': 0, 'W': 6.0, 'h_inst': 3.0,
            'n_rings': chosen_n,
        },
        'P_avg_MW': round(P_chosen, 4),
        'eta_avg': round(result['eta_field_avg'], 6),
        'n_mirrors': result['n_mirrors'],
        'total_area_m2': result['total_area_m2'],
        'P_per_area_W_m2': round(P_chosen / result['total_area_m2'] * 1e6, 2),
        'all_weighting_methods_satisfied': best_n is not None,
        'weighting_min_P': round(min(chosen_subset['P_avg_MW']), 4),
        'weighting_details': {
            m: round(p, 4) for m, p in zip(
                weighting_chosen['labels'], weighting_chosen['P_avg_MW']
            )
        },
    }

    with open('outputs/data/problem2_result_safe.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"\n安全方案结果已保存: outputs/data/problem2_result_safe.json")

    return result_json


if __name__ == '__main__':
    run_safety_margin()