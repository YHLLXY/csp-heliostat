"""
续跑脚本：从改进 #4 开始（#1-#3 已完成）。
用法：python main_improvements_part2.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import time
import json

from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.constraints import total_reflective_area
from csp_heliostat.simulation.annual import annual_simulation

from csp_heliostat.analysis.monthly import compute_monthly_stats
from csp_heliostat.analysis.diurnal import (
    compute_diurnal_curve, SUMMER_SOLSTICE_DOY, WINTER_SOLSTICE_DOY,
)

from csp_heliostat.visualization.layout_plot import (
    plot_field_with_normals,
    plot_convergence_curve,
    plot_monthly_power,
    plot_diurnal_curves,
)

FIGURE_DIR = 'outputs/figure'
DATA_DIR = 'outputs/data'
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

OPTIMAL_PARAMS = {
    'x_t': 0.0, 'y_t': 0.0, 'W': 6.0, 'h': 3.0, 'n_rings': 25,
}


def build_optimal_field():
    mirrors = radial_layout(
        n_rings=OPTIMAL_PARAMS['n_rings'],
        W=OPTIMAL_PARAMS['W'], H=OPTIMAL_PARAMS['W'],
        install_h=OPTIMAL_PARAMS['h'],
        R_inner=EXCLUSION_RADIUS_M,
        R_outer=FIELD_RADIUS_M,
    )
    mirrors = exclude_zone(mirrors, (OPTIMAL_PARAMS['x_t'], OPTIMAL_PARAMS['y_t']),
                           EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)
    return mirrors


def main():
    tower_xy = (OPTIMAL_PARAMS['x_t'], OPTIMAL_PARAMS['y_t'])

    print("构建最优镜场...")
    mirrors = build_optimal_field()
    print(f"  镜面数: {len(mirrors)}")

    # ---- 改进 #4：法向量彩图 ----
    print("\n" + "=" * 60)
    print("改进 #4：法向量彩图")
    print("=" * 60)

    from csp_heliostat.core.solar_position import sun_state_batch

    D_spring = np.array([0.0])
    D_summer = np.array([92.0])
    t_noon = np.array([12.0])

    sun_spring = sun_state_batch(39.4, D_spring, t_noon, convention="from_spring_equinox")
    plot_field_with_normals(
        mirrors,
        sun_alt_deg=float(sun_spring.altitude_deg[0]),
        sun_az_deg=float(sun_spring.azimuth_deg[0]),
        title='Mirror Normals — Spring Equinox Noon',
        save_path=os.path.join(FIGURE_DIR, 'layout_with_normals_spring_noon.png'),
    )
    print(f"  -> {FIGURE_DIR}/layout_with_normals_spring_noon.png")

    sun_summer = sun_state_batch(39.4, D_summer, t_noon, convention="from_spring_equinox")
    plot_field_with_normals(
        mirrors,
        sun_alt_deg=float(sun_summer.altitude_deg[0]),
        sun_az_deg=float(sun_summer.azimuth_deg[0]),
        title='Mirror Normals — Summer Solstice Noon',
        save_path=os.path.join(FIGURE_DIR, 'layout_with_normals_summer_noon.png'),
    )
    print(f"  -> {FIGURE_DIR}/layout_with_normals_summer_noon.png")

    # ---- 改进 #5：收敛曲线 ----
    print("\n" + "=" * 60)
    print("改进 #5：问题三收敛曲线")
    print("=" * 60)

    existing_convergence = os.path.join(DATA_DIR, 'problem3_convergence.csv')
    if os.path.exists(existing_convergence):
        print("  已有收敛数据，直接出图...")
        import csv
        history = []
        with open(existing_convergence, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({
                    'generation': int(row['generation']),
                    'best_fitness': float(row['best_fitness']),
                    'convergence': float(row['convergence']),
                })
    else:
        print("  重跑问题三 DE 优化...")
        from csp_heliostat.optimization.heterogeneous import solve_problem3

        p2_path = os.path.join(DATA_DIR, 'problem2_result.json')
        if os.path.exists(p2_path):
            with open(p2_path, 'r', encoding='utf-8') as f:
                p2_data = json.load(f)
            warm_start = {
                'best_params': [
                    p2_data['params']['x_t'], p2_data['params']['y_t'],
                    p2_data['params']['W'], p2_data['params']['h_inst'],
                    p2_data['params']['n_rings'],
                ],
            }
        else:
            warm_start = None

        t0 = time.time()
        result = solve_problem3(
            warm_start=warm_start,
            n_rings=OPTIMAL_PARAMS['n_rings'],
            maxiter=50, popsize=10, verbose=True,
        )
        print(f"  问题三重跑完成，耗时 {time.time() - t0:.1f}s")
        history = result.get('convergence_history', [])

        import csv
        with open(existing_convergence, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'best_fitness', 'convergence'])
            for h in history:
                writer.writerow([h['generation'], f'{h["best_fitness"]:.6f}',
                                f'{h["convergence"]:.6f}'])
        print(f"  -> {existing_convergence}")

    if history:
        plot_convergence_curve(
            history, uniform_baseline=525.85,
            save_path=os.path.join(FIGURE_DIR, 'problem3_convergence.png'),
        )
        print(f"  -> {FIGURE_DIR}/problem3_convergence.png")

    # ---- 年度仿真（供 #6 使用）----
    print("\n运行年度仿真...")
    annual_result = annual_simulation(
        mirrors, tower_xy=tower_xy,
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        verbose=True,
    )

    # ---- 改进 #6：月度均值 ----
    print("\n" + "=" * 60)
    print("改进 #6：月度均值表")
    print("=" * 60)

    monthly = compute_monthly_stats(annual_result)
    for s in monthly['monthly']:
        flag = " [!!]" if s['below_60'] else ""
        print(f"  {s['month']:>2}: P={s['P_mean_MW']:7.2f} +- {s['P_std_MW']:5.2f} MW  "
              f"eta={s['eta_mean']:.4f}{flag}")

    print(f"\n  低于 60MW: {monthly['months_below_60']}")
    print(f"  最低月均: {monthly['min_monthly_P_MW']:.2f} MW")
    print(f"  储能需求: {monthly['storage_required_MWh']:.0f} MWh")

    import csv
    csv_path = os.path.join(DATA_DIR, 'monthly_power.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['month', 'P_mean_MW', 'P_std_MW', 'eta_mean', 'below_60'])
        for s in monthly['monthly']:
            writer.writerow([s['month'], f'{s["P_mean_MW"]:.4f}',
                           f'{s["P_std_MW"]:.4f}', f'{s["eta_mean"]:.6f}',
                           'Yes' if s['below_60'] else 'No'])
    print(f"  -> {csv_path}")

    plot_monthly_power(monthly['monthly'],
                       save_path=os.path.join(FIGURE_DIR, 'monthly_power_bar.png'))
    print(f"  -> {FIGURE_DIR}/monthly_power_bar.png")

    # ---- 改进 #8：全天曲线 ----
    print("\n" + "=" * 60)
    print("改进 #8：夏至/冬至全天曲线")
    print("=" * 60)

    summer = compute_diurnal_curve(mirrors, tower_xy=tower_xy, doy=SUMMER_SOLSTICE_DOY)
    print(f"  夏至峰值: {max(summer['P_MW']):.2f} MW")
    winter = compute_diurnal_curve(mirrors, tower_xy=tower_xy, doy=WINTER_SOLSTICE_DOY)
    print(f"  冬至峰值: {max(winter['P_MW']):.2f} MW")

    plot_diurnal_curves(
        summer, winter,
        save_path_single=os.path.join(FIGURE_DIR, 'diurnal_curve_{label}.png'),
        save_path_comparison=os.path.join(FIGURE_DIR, 'diurnal_comparison.png'),
    )
    print(f"  -> {FIGURE_DIR}/diurnal_curve_summer.png")
    print(f"  -> {FIGURE_DIR}/diurnal_curve_winter.png")
    print(f"  -> {FIGURE_DIR}/diurnal_comparison.png")

    print("\n" + "=" * 60)
    print("改进 #4-#8 全部完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()