"""
改进清单统一执行入口。

按优先级依次执行 8 项改进，输出所有 CSV + PNG 到 outputs/ 目录。

用法：
    python main_improvements.py              # 执行全部改进
    python main_improvements.py --skip-mc    # 跳过蒙特卡洛（省时间）
    python main_improvements.py --fast       # 快速模式：跳过耗时项
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
import argparse

from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M, REFLECTIVITY, RATED_POWER_MW,
)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.constraints import total_reflective_area
from csp_heliostat.simulation.annual import annual_simulation

# ---- 分析模块 ----
from csp_heliostat.analysis.efficiency_breakdown import (
    compute_efficiency_breakdown, efficiency_breakdown_table,
)
from csp_heliostat.analysis.weighting import compare_weighting_methods
from csp_heliostat.analysis.sensitivity import run_sensitivity
from csp_heliostat.analysis.monthly import compute_monthly_stats
from csp_heliostat.analysis.monte_carlo import run_monte_carlo
from csp_heliostat.analysis.diurnal import (
    compute_diurnal_curve, SUMMER_SOLSTICE_DOY, WINTER_SOLSTICE_DOY,
)

# ---- 可视化 ----
from csp_heliostat.visualization.layout_plot import (
    plot_efficiency_breakdown,
    plot_weighting_comparison,
    plot_sensitivity_curves,
    plot_field_with_normals,
    plot_convergence_curve,
    plot_monthly_power,
    plot_diurnal_curves,
)

# ---- 输出目录 ----
FIGURE_DIR = 'outputs/figure'
DATA_DIR = 'outputs/data'
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---- 最优参数（来自问题二） ----
OPTIMAL_PARAMS = {
    'x_t': 0.0, 'y_t': 0.0, 'W': 6.0, 'h': 3.0, 'n_rings': 25,
}


def build_optimal_field():
    """构建最优镜场（问题二结果）。"""
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


def run_improvement_2_3(annual_result):
    """改进 #2 分项效率拆解 + 改进 #3 加权方法对照。"""
    print("\n" + "=" * 60)
    print("改进 #2：分项效率拆解")
    print("=" * 60)

    breakdown = compute_efficiency_breakdown(annual_result)
    print(efficiency_breakdown_table(breakdown))

    # 保存 CSV
    import csv
    csv_path = os.path.join(DATA_DIR, 'efficiency_components.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['component', 'value'])
        for k, v in breakdown.items():
            writer.writerow([k, f'{v:.6f}'])
    print(f"  -> {csv_path}")

    # 出图
    plot_efficiency_breakdown(
        breakdown,
        save_path_pie=os.path.join(FIGURE_DIR, 'efficiency_breakdown_pie.png'),
        save_path_bar=os.path.join(FIGURE_DIR, 'efficiency_breakdown_bar.png'),
    )
    print(f"  -> {FIGURE_DIR}/efficiency_breakdown_pie.png")
    print(f"  -> {FIGURE_DIR}/efficiency_breakdown_bar.png")

    print("\n" + "=" * 60)
    print("改进 #3：加权方法对照")
    print("=" * 60)

    weighting = compare_weighting_methods(annual_result)
    for label, P, eta in zip(weighting['labels'], weighting['P_avg_MW'], weighting['eta_avg']):
        print(f"  {label:25s}: P_avg = {P:.4f} MW, eta_avg = {eta:.6f}")
    print(f"  最大偏差: {weighting['max_deviation_pct']:.2f}%")
    print(f"  -> {'PASS' if weighting['max_deviation_pct'] < 5.0 else 'WARNING: deviation > 5%'}")

    csv_path = os.path.join(DATA_DIR, 'weighting_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'P_avg_MW', 'eta_avg'])
        for label, P, eta in zip(weighting['labels'], weighting['P_avg_MW'], weighting['eta_avg']):
            writer.writerow([label, f'{P:.4f}', f'{eta:.6f}'])
    print(f"  -> {csv_path}")

    plot_weighting_comparison(
        weighting,
        save_path=os.path.join(FIGURE_DIR, 'weighting_comparison.png'),
    )
    print(f"  -> {FIGURE_DIR}/weighting_comparison.png")

    return breakdown, weighting


def run_improvement_1():
    """改进 #1：敏感性分析。"""
    print("\n" + "=" * 60)
    print("改进 #1：P/A 敏感性分析")
    print("=" * 60)
    print("（使用快速 noon-only 模式加速，预计 ~5-10 分钟）")

    t0 = time.time()

    sensitivity = run_sensitivity(
        best_params=[OPTIMAL_PARAMS['x_t'], OPTIMAL_PARAMS['y_t'],
                     OPTIMAL_PARAMS['W'], OPTIMAL_PARAMS['h']],
        n_rings=OPTIMAL_PARAMS['n_rings'],
        use_fast=True,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\n敏感性分析完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 保存 CSV
    import csv
    csv_path = os.path.join(DATA_DIR, 'sensitivity_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variable', 'value', 'P_avg_MW', 'P_per_area_W_m2',
                         'eta_avg', 'n_mirrors', 'elapsed_s'])
        for var_name, data in sensitivity.items():
            for pt in data['points']:
                writer.writerow([
                    var_name, pt['value'], f'{pt["P_avg_MW"]:.4f}',
                    f'{pt["P_per_area_W_m2"]:.4f}', f'{pt["eta_avg"]:.6f}',
                    pt['n_mirrors'], f'{pt["elapsed_s"]:.1f}',
                ])
    print(f"  -> {csv_path}")

    # 出图
    plot_sensitivity_curves(sensitivity, save_dir=FIGURE_DIR)
    print(f"  -> {FIGURE_DIR}/sensitivity_xy.png")
    print(f"  -> {FIGURE_DIR}/sensitivity_W.png")
    print(f"  -> {FIGURE_DIR}/sensitivity_h.png")
    print(f"  -> {FIGURE_DIR}/sensitivity_n_rings.png")


def run_improvement_4(mirrors):
    """改进 #4：法向量彩图。"""
    print("\n" + "=" * 60)
    print("改进 #4：法向量彩图")
    print("=" * 60)

    # 春分正午：D=0, t=12 → 太阳在正南，高度角 = 90°-39.4° = 50.6°
    # 夏至正午：D=92, t=12 → 太阳高度角 = 90°-39.4°+23.45° = 74.05°

    from csp_heliostat.core.solar_position import sun_state_batch

    # 春分正午
    D_spring = np.array([0.0])
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

    # 夏至正午
    D_summer = np.array([92.0])  # 夏至 D = 172 - 80 = 92
    sun_summer = sun_state_batch(39.4, D_summer, t_noon, convention="from_spring_equinox")
    plot_field_with_normals(
        mirrors,
        sun_alt_deg=float(sun_summer.altitude_deg[0]),
        sun_az_deg=float(sun_summer.azimuth_deg[0]),
        title='Mirror Normals — Summer Solstice Noon',
        save_path=os.path.join(FIGURE_DIR, 'layout_with_normals_summer_noon.png'),
    )
    print(f"  -> {FIGURE_DIR}/layout_with_normals_summer_noon.png")


def run_improvement_5():
    """改进 #5：问题三收敛曲线。"""
    print("\n" + "=" * 60)
    print("改进 #5：问题三收敛曲线")
    print("=" * 60)
    print("（需重跑问题三以记录收敛历史，预计 ~2-5 分钟）")

    # 尝试从已有结果加载
    json_path = os.path.join(DATA_DIR, 'problem3_result.json')
    existing_convergence = os.path.join(DATA_DIR, 'problem3_convergence.csv')

    # 如果已存在收敛数据，直接出图
    if os.path.exists(existing_convergence):
        print("  检测到已有收敛数据，直接出图...")
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

        # 加载问题二 warm-start
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
            maxiter=50,
            popsize=10,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  问题三重跑完成，耗时 {elapsed:.1f}s")

        history = result.get('convergence_history', [])

        # 保存收敛历史 CSV
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
            history,
            uniform_baseline=525.85,
            save_path=os.path.join(FIGURE_DIR, 'problem3_convergence.png'),
        )
        print(f"  -> {FIGURE_DIR}/problem3_convergence.png")

        # 检查是否有上穿
        max_fitness = max(h['best_fitness'] for h in history)
        # fitness 值是 MW/m² 单位，baseline 525.85 W/m² = 525.85e-6 MW/m²
        if max_fitness < -525.85e-6:
            print("  [OK] 收敛曲线未超过均一基线，结论稳健")
        else:
            print("  [!!] 警告：某代目标值上穿均一基线，可能需要扩大搜索")
    else:
        print("  [!!] 无收敛历史数据")


def run_improvement_6(annual_result):
    """改进 #6：月度均值表。"""
    print("\n" + "=" * 60)
    print("改进 #6：月度均值表")
    print("=" * 60)

    monthly = compute_monthly_stats(annual_result)

    print(f"  {'Month':>5}  {'P_mean':>8}  {'P_std':>8}  {'eta_mean':>9}  {'<60MW':>6}")
    print("  " + "-" * 45)
    for s in monthly['monthly']:
        flag = " [!!]" if s['below_60'] else ""
        print(f"  {s['month']:>5}  {s['P_mean_MW']:>8.2f}  {s['P_std_MW']:>8.2f}  "
              f"{s['eta_mean']:>9.4f}  {'Yes' if s['below_60'] else 'No':>6}{flag}")

    print(f"\n  低于 60MW 的月份: {monthly['months_below_60']}")
    print(f"  最低月均功率: {monthly['min_monthly_P_MW']:.2f} MW")
    print(f"  估算储能需求: {monthly['storage_required_MWh']:.0f} MWh")

    # 保存 CSV
    import csv
    csv_path = os.path.join(DATA_DIR, 'monthly_power.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['month', 'P_mean_MW', 'P_std_MW', 'eta_mean', 'below_60'])
        for s in monthly['monthly']:
            writer.writerow([
                s['month'], f'{s["P_mean_MW"]:.4f}', f'{s["P_std_MW"]:.4f}',
                f'{s["eta_mean"]:.6f}', 'Yes' if s['below_60'] else 'No',
            ])
    print(f"  -> {csv_path}")

    plot_monthly_power(
        monthly['monthly'],
        save_path=os.path.join(FIGURE_DIR, 'monthly_power_bar.png'),
    )
    print(f"  -> {FIGURE_DIR}/monthly_power_bar.png")


def run_improvement_7(mirrors):
    """改进 #7：蒙特卡洛置信带。"""
    print("\n" + "=" * 60)
    print("改进 #7：蒙特卡洛置信带")
    print("=" * 60)
    print("（10 次重复仿真，预计 ~8-10 分钟）")

    t0 = time.time()
    mc = run_monte_carlo(
        mirrors,
        tower_xy=(OPTIMAL_PARAMS['x_t'], OPTIMAL_PARAMS['y_t']),
        n_runs=10,
        seed_start=0,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  蒙特卡洛结果 ({mc['cv_pct']:.3f}% 变异系数):")
    print(f"    P_mean = {mc['P_mean_MW']:.4f} +- {mc['P_std_MW']:.4f} MW")
    print(f"    eta_mean = {mc['eta_mean']:.6f} +- {mc['eta_std']:.6f}")
    print(f"    耗时: {elapsed:.1f}s")
    print(f"    -> {'PASS' if mc['cv_pct'] < 1.0 else 'WARNING: CV > 1%'}")

    # 保存 CSV
    import csv
    csv_path = os.path.join(DATA_DIR, 'monte_carlo_confidence.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'P_MW', 'eta_avg'])
        for i, (P, eta) in enumerate(zip(mc['all_P_MW'], mc['all_eta'])):
            writer.writerow([i, f'{P:.6f}', f'{eta:.8f}'])
    print(f"  -> {csv_path}")


def run_improvement_8(mirrors):
    """改进 #8：夏至/冬至全天曲线。"""
    print("\n" + "=" * 60)
    print("改进 #8：夏至/冬至全天曲线")
    print("=" * 60)

    tower_xy = (OPTIMAL_PARAMS['x_t'], OPTIMAL_PARAMS['y_t'])

    # 夏至
    print("  计算夏至全天曲线 (6:00-18:00)...")
    summer = compute_diurnal_curve(
        mirrors, tower_xy=tower_xy,
        doy=SUMMER_SOLSTICE_DOY,
    )
    print(f"    夏至峰值功率: {max(summer['P_MW']):.2f} MW")

    # 冬至
    print("  计算冬至全天曲线 (6:00-18:00)...")
    winter = compute_diurnal_curve(
        mirrors, tower_xy=tower_xy,
        doy=WINTER_SOLSTICE_DOY,
    )
    print(f"    冬至峰值功率: {max(winter['P_MW']):.2f} MW")

    # 出图
    plot_diurnal_curves(
        summer, winter,
        save_path_single=os.path.join(FIGURE_DIR, 'diurnal_curve_{label}.png'),
        save_path_comparison=os.path.join(FIGURE_DIR, 'diurnal_comparison.png'),
    )
    print(f"  -> {FIGURE_DIR}/diurnal_curve_summer.png")
    print(f"  -> {FIGURE_DIR}/diurnal_curve_winter.png")
    print(f"  -> {FIGURE_DIR}/diurnal_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='运行改进清单分析')
    parser.add_argument('--skip-mc', action='store_true', help='跳过蒙特卡洛（省 ~10min）')
    parser.add_argument('--skip-p3', action='store_true', help='跳过问题三重跑')
    parser.add_argument('--fast', action='store_true', help='快速模式：跳过所有耗时项')
    args = parser.parse_args()

    if args.fast:
        args.skip_mc = True
        args.skip_p3 = True

    print("=" * 60)
    print("定日镜场优化 — 改进清单执行")
    print("=" * 60)
    print(f"  最优参数: x_t={OPTIMAL_PARAMS['x_t']}, y_t={OPTIMAL_PARAMS['y_t']}, "
          f"W={OPTIMAL_PARAMS['W']}m, h={OPTIMAL_PARAMS['h']}m, "
          f"n_rings={OPTIMAL_PARAMS['n_rings']}")
    print(f"  输出目录: {FIGURE_DIR}/, {DATA_DIR}/")

    # ---- 构建镜场 ----
    print("\n构建最优镜场...")
    t0 = time.time()
    mirrors = build_optimal_field()
    print(f"  镜面数: {len(mirrors)}, 总面积: {total_reflective_area(mirrors):.0f} m2")

    # ---- 运行一次完整年度仿真（供 #2, #3, #6 复用）----
    print("\n运行年度仿真（60 时刻，供后续分析复用）...")
    annual_result = annual_simulation(
        mirrors,
        tower_xy=(OPTIMAL_PARAMS['x_t'], OPTIMAL_PARAMS['y_t']),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        verbose=True,
    )
    print(f"  P_avg = {annual_result['P_field_avg_MW']:.4f} MW")
    print(f"  eta_avg = {annual_result['eta_field_avg']:.6f}")

    # ---- 执行改进 ----
    total_start = time.time()

    # 第一批：改进 #2 + #3（复用 annual_result）
    run_improvement_2_3(annual_result)

    # 第二批：改进 #1（需要独立扫描）
    run_improvement_1()

    # 第三批：改进 #4（复用 mirrors，无需仿真）
    run_improvement_4(mirrors)

    # 第四批：改进 #5（可能需重跑问题三）+ 改进 #6（复用 annual_result）
    if not args.skip_p3:
        run_improvement_5()
    else:
        print("\n[跳过] 改进 #5：问题三收敛曲线")

    run_improvement_6(annual_result)

    # 第五批：改进 #7 + #8
    if not args.skip_mc:
        run_improvement_7(mirrors)
    else:
        print("\n[跳过] 改进 #7：蒙特卡洛置信带")

    run_improvement_8(mirrors)

    # ---- 汇总 ----
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("全部改进完成！")
    print(f"  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  输出图表: {FIGURE_DIR}/")
    print(f"  输出数据: {DATA_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()