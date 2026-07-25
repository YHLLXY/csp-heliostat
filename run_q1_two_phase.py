"""
Q1 步骤2-5：两阶段 DE 重跑问题三。

阶段 1：noon-only 快搜（40代, popsize=15, 校准约束）
阶段 2：full-60 精修（10代, popsize=5, warm-start, 真实约束）

输出:
  - outputs/data/problem3_convergence_v2.csv
  - outputs/data/problem3_result_v2.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import pandas as pd
from scipy.optimize import differential_evolution

from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
    MIRROR_W_RANGE, INSTALL_HEIGHT_RANGE, RATED_POWER_MW,
)
from csp_heliostat.config.sampling import build_noon_schedule, build_sampling_schedule
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.simulation.annual import annual_simulation

# ============================================================
# 校准系数（从 calibration 结果读取）
# ============================================================
CALIBRATION_RATIO = 0.922955  # P_full ≈ k * P_noon
CALIBRATED_TARGET = RATED_POWER_MW / CALIBRATION_RATIO  # ≈ 65.01 MW
PENALTY_FACTOR = 1e4

print(f"CALIBRATION_RATIO = {CALIBRATION_RATIO:.6f}")
print(f"CALIBRATED_TARGET (noon) = {CALIBRATED_TARGET:.2f} MW")


def _simulate_zoned(params, n_rings, schedule=None, verbose=False):
    """分环参数 → 年度仿真结果。复用 heterogeneous.py 的逻辑。"""
    x_t, y_t = params[0], params[1]

    dr = (FIELD_RADIUS_M - EXCLUSION_RADIUS_M) / n_rings
    mirrors = []

    for k in range(n_rings):
        r_k = EXCLUSION_RADIUS_M + (k + 0.5) * dr
        W_k = float(np.clip(params[2 + 2*k], MIRROR_W_RANGE[0], MIRROR_W_RANGE[1]))
        h_k = float(np.clip(params[2 + 2*k + 1], INSTALL_HEIGHT_RANGE[0], INSTALL_HEIGHT_RANGE[1]))
        H_k = W_k

        n_k = max(1, int(np.floor(2.0 * np.pi * r_k / (W_k + 5.0))))
        dtheta = 2.0 * np.pi / n_k
        offset = dtheta / 2.0 if k % 2 == 1 else 0.0

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


def objective_phase1(params, n_rings, noon_schedule):
    """阶段1：noon-only 快速评估 + 校准约束"""
    x_t, y_t = params[0], params[1]
    if abs(x_t) > 120 or abs(y_t) > 120:
        return 1e10

    result = _simulate_zoned(params, n_rings, schedule=noon_schedule)
    P_noon = result['P_field_avg_MW']
    N = result['n_mirrors']
    total_area = result['total_area_m2']

    if N == 0 or total_area < 1.0:
        return 1e10

    P_per_area = P_noon / total_area
    P_full_estimated = P_noon * CALIBRATION_RATIO

    penalty = 0.0
    if P_full_estimated < RATED_POWER_MW:
        penalty = PENALTY_FACTOR * (RATED_POWER_MW - P_full_estimated) ** 2

    return -P_per_area + penalty


def objective_phase2(params, n_rings, full_schedule):
    """阶段2：full-60 精确评估 + 真实约束"""
    x_t, y_t = params[0], params[1]
    if abs(x_t) > 120 or abs(y_t) > 120:
        return 1e10

    result = _simulate_zoned(params, n_rings, schedule=full_schedule)
    P_full = result['P_field_avg_MW']
    N = result['n_mirrors']
    total_area = result['total_area_m2']

    if N == 0 or total_area < 1.0:
        return 1e10

    P_per_area = P_full / total_area

    penalty = 0.0
    if P_full < RATED_POWER_MW:
        penalty = PENALTY_FACTOR * (RATED_POWER_MW - P_full) ** 2

    return -P_per_area + penalty


def build_initial_population(warm_start, n_rings, popsize):
    """从问题二 warm-start 构建初始种群"""
    x_t, y_t, W_base, h_base = warm_start['best_params'][:4]

    init_params = np.zeros(2 + 2 * n_rings)
    init_params[0] = x_t
    init_params[1] = y_t

    np.random.seed(42)
    for k in range(n_rings):
        W_factor = 1.0 + 0.1 * (np.random.random() - 0.5)
        h_factor = 1.0 + 0.1 * (np.random.random() - 0.5)
        init_params[2 + 2*k] = np.clip(W_base * W_factor, *MIRROR_W_RANGE)
        init_params[2 + 2*k + 1] = np.clip(h_base * h_factor, *INSTALL_HEIGHT_RANGE)

    init_pop = [init_params]
    for _ in range(popsize - 1):
        perturbed = init_params.copy()
        perturbed[0] += np.random.uniform(-10, 10)
        perturbed[1] += np.random.uniform(-10, 10)
        for k in range(n_rings):
            perturbed[2 + 2*k] *= np.random.uniform(0.85, 1.15)
            perturbed[2 + 2*k + 1] *= np.random.uniform(0.85, 1.15)
        perturbed[0] = np.clip(perturbed[0], -100, 100)
        perturbed[1] = np.clip(perturbed[1], -100, 100)
        perturbed[2::2] = np.clip(perturbed[2::2], *MIRROR_W_RANGE)
        perturbed[3::2] = np.clip(perturbed[3::2], *INSTALL_HEIGHT_RANGE)
        init_pop.append(perturbed)

    return np.array(init_pop[:popsize])


def run_two_phase_de():
    """主函数：两阶段 DE"""
    print("=" * 60)
    print("Q1: 两阶段 DE 优化 问题三")
    print(f"  校准系数 k = {CALIBRATION_RATIO:.6f}")
    print(f"  校准阈值 (noon) = {CALIBRATED_TARGET:.2f} MW")
    print("=" * 60)

    # 加载 warm-start
    with open('outputs/data/problem2_result.json', 'r') as f:
        warm_start_data = json.load(f)
    warm_start = {
        'best_params': [
            warm_start_data['params']['x_t'],
            warm_start_data['params']['y_t'],
            warm_start_data['params']['W'],
            warm_start_data['params']['h_inst'],
            warm_start_data['params']['n_rings'],
        ],
    }
    n_rings = warm_start_data['params']['n_rings']
    print(f"\nWarm-start: n_rings={n_rings}, W={warm_start['best_params'][2]:.1f}m, "
          f"h={warm_start['best_params'][3]:.1f}m")

    noon_schedule = build_noon_schedule()
    full_schedule = build_sampling_schedule()

    # 变量边界
    bounds = [(-100.0, 100.0), (-100.0, 100.0)]
    for _ in range(n_rings):
        bounds.append(MIRROR_W_RANGE)
        bounds.append(INSTALL_HEIGHT_RANGE)

    # ============================================================
    # 阶段 1：noon-only 快搜
    # ============================================================
    print(f"\n{'='*40}")
    print(f"阶段 1：noon-only DE（40代, popsize=15）")
    print(f"{'='*40}")

    init_pop_p1 = build_initial_population(warm_start, n_rings, 15)

    history = []
    _gen_counter = [0]

    def callback_phase1(xk, convergence):
        gen = _gen_counter[0]
        _gen_counter[0] += 1

        params = np.clip(xk, [b[0] for b in bounds], [b[1] for b in bounds])

        # Noon 评估（每次都做）
        r_noon = _simulate_zoned(params, n_rings, schedule=noon_schedule)
        P_noon = r_noon['P_field_avg_MW']
        P_per_area_noon = P_noon / r_noon['total_area_m2'] if r_noon['total_area_m2'] > 0 else 0
        P_full_est = P_noon * CALIBRATION_RATIO

        # 每5代 + 最后一代跑 full-60 验证
        if gen % 5 == 0 or gen >= 39:
            r_full = _simulate_zoned(params, n_rings, schedule=full_schedule)
            P_full = r_full['P_field_avg_MW']
            P_per_area_full = P_full / r_full['total_area_m2'] if r_full['total_area_m2'] > 0 else 0
            constraint_actual = P_full >= RATED_POWER_MW
        else:
            P_full = None
            P_per_area_full = None
            constraint_actual = None

        # 评估当前 best 的 phase1 objective
        best_f = objective_phase1(params, n_rings, noon_schedule)

        entry = {
            'generation': gen,
            'phase': 1,
            'best_fitness': round(float(best_f), 10),
            'P_noon_MW': round(P_noon, 4),
            'P_per_area_noon_W_m2': round(P_per_area_noon * 1e6, 2),
            'P_full_estimated_MW': round(P_full_est, 4),
            'P_full_actual_MW': round(P_full, 4) if P_full is not None else None,
            'P_per_area_full_W_m2': round(P_per_area_full * 1e6, 2) if P_per_area_full is not None else None,
            'constraint_estimated': P_full_est >= RATED_POWER_MW,
            'constraint_actual': constraint_actual,
            'n_mirrors': r_noon['n_mirrors'],
        }
        history.append(entry)
        print(f"  [Ph1] gen {gen:2d}: P_noon={P_noon:.2f}, P_full_est={P_full_est:.2f}, "
              f"constraint_est={entry['constraint_estimated']}, "
              f"P_full_actual={P_full:.2f if P_full else 'N/A'}, "
              f"constraint_actual={constraint_actual}")

    result_p1 = differential_evolution(
        lambda p: objective_phase1(p, n_rings, noon_schedule),
        bounds=bounds,
        maxiter=40,
        popsize=15,
        tol=1e-8,
        seed=42,
        init=init_pop_p1,
        polish=False,  # 阶段2再做polish
        callback=callback_phase1,
    )

    print(f"\n阶段 1 完成: best_fitness={result_p1.fun:.10f}, nfev={result_p1.nfev}")

    # ============================================================
    # 阶段 2：full-60 精修
    # ============================================================
    print(f"\n{'='*40}")
    print(f"阶段 2：full-60 DE（10代, popsize=5, warm-start）")
    print(f"{'='*40}")

    # 用阶段1的最终种群做 warm-start
    # differential_evolution 内部会保存最终种群
    # 如果种群大小不同，从阶段1种群中采样
    if hasattr(result_p1, 'population') and result_p1.population is not None:
        pop_p1 = result_p1.population
        if len(pop_p1) >= 5:
            indices = np.random.RandomState(43).choice(len(pop_p1), 5, replace=False)
            init_pop_p2 = pop_p1[indices]
        else:
            init_pop_p2 = np.tile(pop_p1[0:1], (5, 1))
    else:
        # Fallback: 用阶段1最优解生成种群
        best_p1 = result_p1.x
        init_pop_p2 = [best_p1]
        np.random.seed(43)
        for _ in range(4):
            perturbed = best_p1.copy()
            perturbed *= np.random.uniform(0.95, 1.05, size=len(best_p1))
            perturbed[0] = np.clip(perturbed[0], -100, 100)
            perturbed[1] = np.clip(perturbed[1], -100, 100)
            perturbed[2::2] = np.clip(perturbed[2::2], *MIRROR_W_RANGE)
            perturbed[3::2] = np.clip(perturbed[3::2], *INSTALL_HEIGHT_RANGE)
            init_pop_p2.append(perturbed)
        init_pop_p2 = np.array(init_pop_p2)

    _gen_counter_p2 = [0]

    def callback_phase2(xk, convergence):
        gen = _gen_counter_p2[0]
        _gen_counter_p2[0] += 1
        global_gen = 40 + gen

        params = np.clip(xk, [b[0] for b in bounds], [b[1] for b in bounds])

        # Full-60 评估
        r_full = _simulate_zoned(params, n_rings, schedule=full_schedule)
        P_full = r_full['P_field_avg_MW']
        P_per_area_full = P_full / r_full['total_area_m2'] if r_full['total_area_m2'] > 0 else 0
        constraint_actual = P_full >= RATED_POWER_MW

        best_f = objective_phase2(params, n_rings, full_schedule)

        entry = {
            'generation': global_gen,
            'phase': 2,
            'best_fitness': round(float(best_f), 10),
            'P_noon_MW': None,
            'P_per_area_noon_W_m2': None,
            'P_full_estimated_MW': None,
            'P_full_actual_MW': round(P_full, 4),
            'P_per_area_full_W_m2': round(P_per_area_full * 1e6, 2),
            'constraint_estimated': None,
            'constraint_actual': constraint_actual,
            'n_mirrors': r_full['n_mirrors'],
        }
        history.append(entry)
        print(f"  [Ph2] gen {global_gen:2d}: P_full={P_full:.2f}, "
              f"P/A_full={P_per_area_full*1e6:.1f} W/m2, "
              f"constraint={'OK' if constraint_actual else 'FAIL'}")

    result_p2 = differential_evolution(
        lambda p: objective_phase2(p, n_rings, full_schedule),
        bounds=bounds,
        maxiter=10,
        popsize=5,
        tol=1e-10,
        seed=43,
        init=init_pop_p2,
        polish=True,
        callback=callback_phase2,
    )

    print(f"\n阶段 2 完成: best_fitness={result_p2.fun:.10f}, nfev={result_p2.nfev}")

    # ============================================================
    # 最终评估
    # ============================================================
    final_params = result_p2.x
    final_result = _simulate_zoned(final_params, n_rings, schedule=full_schedule)
    P_final = final_result['P_field_avg_MW']
    P_per_area_final = P_final / final_result['total_area_m2'] if final_result['total_area_m2'] > 0 else 0

    print(f"\n{'='*40}")
    print(f"最终结果")
    print(f"{'='*40}")
    print(f"  x_t={final_params[0]:.1f}, y_t={final_params[1]:.1f}")
    print(f"  P_avg = {P_final:.4f} MW")
    print(f"  P/A = {P_per_area_final*1e6:.2f} W/m²")
    print(f"  N = {final_result['n_mirrors']}")
    print(f"  η = {final_result['eta_field_avg']:.6f}")
    print(f"  约束满足: {P_final >= RATED_POWER_MW}")

    # 输出每环参数
    per_ring = []
    for k in range(n_rings):
        W_k = float(final_params[2 + 2*k])
        h_k = float(final_params[2 + 2*k + 1])
        per_ring.append({'ring': k, 'W_m': round(W_k, 3), 'h_m': round(h_k, 3)})
    print(f"  首环: W={per_ring[0]['W_m']:.2f}, h={per_ring[0]['h_m']:.2f}")
    print(f"  末环: W={per_ring[-1]['W_m']:.2f}, h={per_ring[-1]['h_m']:.2f}")

    # ============================================================
    # 保存结果
    # ============================================================
    os.makedirs('outputs/data', exist_ok=True)

    # CSV: convergence history
    df = pd.DataFrame(history)
    df.to_csv('outputs/data/problem3_convergence_v2.csv', index=False)
    print(f"\n收敛历史已保存: outputs/data/problem3_convergence_v2.csv")

    # JSON: result
    # 判断收敛（阶段2末段 best_fitness 变化 < 0.1%）
    phase2_entries = [h for h in history if h['phase'] == 2]
    converged = False
    if len(phase2_entries) >= 3:
        recent_fitness = [h['best_fitness'] for h in phase2_entries[-3:]]
        if max(recent_fitness) - min(recent_fitness) < abs(np.mean(recent_fitness)) * 0.001:
            converged = True

    # 判断结论
    baseline_P_per_area = 525.85  # W/m²
    if P_final >= RATED_POWER_MW:
        if P_per_area_final * 1e6 > baseline_P_per_area * 1.005:
            conclusion = ("分环异构可提升 P/A 至 {:.1f} W/m²（+{:.1f}%），"
                          "同时满足 60MW 约束。".format(
                P_per_area_final * 1e6,
                (P_per_area_final * 1e6 / baseline_P_per_area - 1) * 100))
        else:
            conclusion = ("在 P̄ ≥ 60MW 约束下，均一配置接近最优。"
                          "经两阶段 DE（40代 noon 快搜 + 10代 full-60 精修）验证，"
                          "异构无法显著突破均一基线 (525.85 W/m²)。")
    else:
        conclusion = ("异构可提升单位面积效率，但无法同时满足 60MW 约束。"
                      "在约束下均一配置最优。")

    result_json = {
        'problem': 3,
        'type': 'heterogeneous_zones',
        'method': 'two_phase_DE_calibrated',
        'calibration_ratio': CALIBRATION_RATIO,
        'phase1': {
            'sampling': 'noon_12',
            'maxiter': 40,
            'popsize': 15,
            'nfev': result_p1.nfev,
        },
        'phase2': {
            'sampling': 'full_60',
            'maxiter': 10,
            'popsize': 5,
            'warm_start_from': 'phase1_population',
            'nfev': result_p2.nfev,
            'converged': converged,
        },
        'conclusion': conclusion,
        'constraint_handling': 'power_penalty_1e4_calibrated',
        'best_params': {
            'x_t': round(float(final_params[0]), 2),
            'y_t': round(float(final_params[1]), 2),
        },
        'per_ring': per_ring,
        'best_P_avg_MW': round(P_final, 4),
        'best_P_per_area_W_m2': round(P_per_area_final * 1e6, 2),
        'best_eta_avg': round(final_result['eta_field_avg'], 6),
        'best_n_mirrors': final_result['n_mirrors'],
        'best_total_area_m2': final_result['total_area_m2'],
    }

    with open('outputs/data/problem3_result_v2.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: outputs/data/problem3_result_v2.json")

    return result_json


if __name__ == '__main__':
    run_two_phase_de()