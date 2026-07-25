"""
Q1 两阶段 DE（5组分组方案）— 解决 popsize×N 种群爆炸问题。

核心改动：
  原 25 环独立参数 → 25 环分 5 组共享 (W, h)
  变量数 N: 52 → 12
  实际种群: popsize=5 × N=12 = 60（原 5×52=260, 10×52=520）

性能预估（单次仿真 ~33s, 8核并行）:
  阶段1: 60×33/8 ×25代 ≈ 1.7h
  阶段2: 60×53/8 × 8代 ≈ 0.9h
  合计 ≈ 2.6h

输出:
  - outputs/data/problem3_convergence_v2.csv
  - outputs/data/problem3_result_v2.json
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from csp_heliostat.config.constants import (
    TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
    FIELD_RADIUS_M, EXCLUSION_RADIUS_M,
    MIRROR_W_RANGE, INSTALL_HEIGHT_RANGE, RATED_POWER_MW,
)
from csp_heliostat.config.sampling import build_noon_schedule, build_sampling_schedule
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.field.layout import exclude_zone, field_boundary_filter
from csp_heliostat.simulation.annual import annual_simulation

# ============================================================
# 全局配置
# ============================================================
CALIBRATION_RATIO = 0.922955        # P_full ≈ k × P_noon
PENALTY_FACTOR = 1e4
N_RINGS = 25
N_GROUPS = 5                        # 25 环分 5 组
N_VARS = 2 + 2 * N_GROUPS           # = 12（塔xy + 5组W/h）

NOON_SCHEDULE = None
FULL_SCHEDULE = None
HISTORY = []
GEN_COUNTER = 0
PHASE = 1

# 变量边界（12 个）
BOUNDS = [(-100.0, 100.0), (-100.0, 100.0)]   # 塔 x, y
for _ in range(N_GROUPS):
    BOUNDS.append(MIRROR_W_RANGE)             # 每组 W
    BOUNDS.append(INSTALL_HEIGHT_RANGE)       # 每组 h

PHASE1_MAXITER = 25
PHASE1_POPSIZE = 5
PHASE2_MAXITER = 8
PHASE2_POPSIZE = 5


# ============================================================
# 核心仿真：5 组分组 → 25 环展开 → annual_simulation
# ============================================================
def _simulate(params, schedule):
    """构建镜场 + 运行仿真。

    参数布局（12 维）:
        params[0], params[1]   = 塔 (x_t, y_t)
        params[2], params[3]    = 第 0 组 (W, h)
        params[4], params[5]    = 第 1 组 (W, h)
        params[6], params[7]    = 第 2 组 (W, h)
        params[8], params[9]    = 第 3 组 (W, h)
        params[10], params[11]  = 第 4 组 (W, h)

    分组映射：环 k ∈ [0, 24] → 组 g = floor(k × N_GROUPS / N_RINGS)
        环 0-4  → 组 0（内环）
        环 5-9  → 组 1
        环 10-14 → 组 2
        环 15-19 → 组 3
        环 20-24 → 组 4（外环）
    """
    x_t, y_t = float(params[0]), float(params[1])

    # 提取 5 组参数
    group_W = []
    group_h = []
    for g in range(N_GROUPS):
        W_g = float(np.clip(params[2 + 2*g], *MIRROR_W_RANGE))
        h_g = float(np.clip(params[2 + 2*g + 1], *INSTALL_HEIGHT_RANGE))
        group_W.append(W_g)
        group_h.append(h_g)

    dr = (FIELD_RADIUS_M - EXCLUSION_RADIUS_M) / N_RINGS
    mirrors = []
    for k in range(N_RINGS):
        r_k = EXCLUSION_RADIUS_M + (k + 0.5) * dr
        # 分组映射
        g = int(k * N_GROUPS / N_RINGS)
        W_k = group_W[g]
        h_k = group_h[g]
        H_k = W_k

        n_k = max(1, int(np.floor(2.0 * np.pi * r_k / (W_k + 5.0))))
        dtheta = 2.0 * np.pi / n_k
        offset = dtheta / 2.0 if k % 2 == 1 else 0.0

        for j in range(n_k):
            theta = offset + j * dtheta
            mirrors.append(Mirror(
                x=float(r_k * np.cos(theta)),
                y=float(r_k * np.sin(theta)),
                width=W_k, height=H_k,
                install_height=h_k,
            ))

    mirrors = exclude_zone(mirrors, (x_t, y_t), EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, (0.0, 0.0), FIELD_RADIUS_M)

    if len(mirrors) == 0:
        return {'P_field_avg_MW': 0.0, 'n_mirrors': 0, 'total_area_m2': 0.0,
                'eta_field_avg': 0.0}

    return annual_simulation(
        mirrors,
        tower_xy=(x_t, y_t),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        schedule=schedule,
        verbose=False,
    )


# ============================================================
# 目标函数
# ============================================================
def obj_phase1(params):
    """阶段1：noon-only + 校准约束"""
    if abs(params[0]) > 120 or abs(params[1]) > 120:
        return 1e10
    r = _simulate(params, NOON_SCHEDULE)
    if r['n_mirrors'] == 0 or r['total_area_m2'] < 1:
        return 1e10
    P_per_area = r['P_field_avg_MW'] / r['total_area_m2']   # MW/m²
    P_est = r['P_field_avg_MW'] * CALIBRATION_RATIO
    penalty = PENALTY_FACTOR * max(0, RATED_POWER_MW - P_est) ** 2
    return float(-P_per_area + penalty)


def obj_phase2(params):
    """阶段2：full-60 + 真实约束"""
    if abs(params[0]) > 120 or abs(params[1]) > 120:
        return 1e10
    r = _simulate(params, FULL_SCHEDULE)
    if r['n_mirrors'] == 0 or r['total_area_m2'] < 1:
        return 1e10
    P_per_area = r['P_field_avg_MW'] / r['total_area_m2']
    penalty = PENALTY_FACTOR * max(0, RATED_POWER_MW - r['P_field_avg_MW']) ** 2
    return float(-P_per_area + penalty)


# ============================================================
# 初始种群：从问题二 warm-start
# ============================================================
def build_init_pop(popsize):
    """从 problem2_result.json 构建 (popsize, 12) 初始种群"""
    with open('outputs/data/problem2_result.json') as f:
        ws = json.load(f)
    x_t = ws['params']['x_t']
    y_t = ws['params']['y_t']
    W0 = ws['params']['W']
    h0 = ws['params']['h_inst']

    # 基础解：所有组都用问题二的 W, h
    base = np.zeros(N_VARS)
    base[0], base[1] = x_t, y_t
    for g in range(N_GROUPS):
        base[2 + 2*g] = W0
        base[2 + 2*g + 1] = h0

    # 扰动构建种群
    np.random.seed(42)
    pop = [base.copy()]
    for _ in range(popsize - 1):
        p = base.copy()
        p[0] += np.random.uniform(-10, 10)
        p[1] += np.random.uniform(-10, 10)
        for g in range(N_GROUPS):
            # 各组扰动幅度 10%（与原方案一致）
            p[2 + 2*g] *= np.random.uniform(0.9, 1.1)
            p[2 + 2*g + 1] *= np.random.uniform(0.9, 1.1)
        p = np.clip(p, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        pop.append(p)
    return np.array(pop)


# ============================================================
# 回调函数
# ============================================================
def cb_phase1(xk, convergence):
    global GEN_COUNTER, HISTORY, PHASE
    params = np.clip(xk, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    r_noon = _simulate(params, NOON_SCHEDULE)
    P_noon = r_noon['P_field_avg_MW']
    PA_noon = P_noon / r_noon['total_area_m2'] if r_noon['total_area_m2'] > 0 else 0
    P_est = P_noon * CALIBRATION_RATIO

    P_full = PA_full = None
    constraint_actual = None
    if GEN_COUNTER % 5 == 0 or GEN_COUNTER >= PHASE1_MAXITER - 1:
        r_full = _simulate(params, FULL_SCHEDULE)
        P_full = r_full['P_field_avg_MW']
        PA_full = P_full / r_full['total_area_m2'] if r_full['total_area_m2'] > 0 else 0
        constraint_actual = P_full >= RATED_POWER_MW

    best_f = obj_phase1(params)
    entry = {
        'generation': GEN_COUNTER, 'phase': PHASE,
        'best_fitness': round(best_f, 10),
        'P_noon_MW': round(P_noon, 4),
        'P_per_area_noon_W_m2': round(PA_noon * 1e6, 2),
        'P_full_estimated_MW': round(P_est, 4),
        'P_full_actual_MW': round(P_full, 4) if P_full is not None else None,
        'P_per_area_full_W_m2': round(PA_full * 1e6, 2) if PA_full is not None else None,
        'constraint_estimated': P_est >= RATED_POWER_MW,
        'constraint_actual': constraint_actual,
        'n_mirrors': r_noon['n_mirrors'],
    }
    HISTORY.append(entry)
    pfs = f"{P_full:.2f}" if P_full is not None else "N/A"
    print(f"  [Ph1] gen {GEN_COUNTER:2d}: P_noon={P_noon:.2f}, P_est={P_est:.2f}, "
          f"P_full={pfs}, constraint={constraint_actual}", flush=True)
    GEN_COUNTER += 1


def cb_phase2(xk, convergence):
    global GEN_COUNTER, HISTORY, PHASE
    params = np.clip(xk, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    r_full = _simulate(params, FULL_SCHEDULE)
    P_full = r_full['P_field_avg_MW']
    PA_full = P_full / r_full['total_area_m2'] if r_full['total_area_m2'] > 0 else 0
    constraint_actual = P_full >= RATED_POWER_MW
    best_f = obj_phase2(params)
    entry = {
        'generation': GEN_COUNTER, 'phase': PHASE,
        'best_fitness': round(best_f, 10),
        'P_noon_MW': None, 'P_per_area_noon_W_m2': None,
        'P_full_estimated_MW': None,
        'P_full_actual_MW': round(P_full, 4),
        'P_per_area_full_W_m2': round(PA_full * 1e6, 2),
        'constraint_estimated': None, 'constraint_actual': constraint_actual,
        'n_mirrors': r_full['n_mirrors'],
    }
    HISTORY.append(entry)
    print(f"  [Ph2] gen {GEN_COUNTER:2d}: P_full={P_full:.2f}, PA={PA_full*1e6:.1f}, "
          f"OK={constraint_actual}", flush=True)
    GEN_COUNTER += 1


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    t_start = time.time()

    # 切换到开发文件目录（保证 outputs 路径正确）
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 如果 outputs/data 不存在，创建
    os.makedirs('outputs/data', exist_ok=True)

    NOON_SCHEDULE = build_noon_schedule()
    FULL_SCHEDULE = build_sampling_schedule()

    print("=" * 60, flush=True)
    print("Q1: 两阶段 DE（5组分组方案）", flush=True)
    print(f"  变量数 N = {N_VARS}（原 52）", flush=True)
    print(f"  分组数 = {N_GROUPS}（每 {N_RINGS//N_GROUPS} 环共享参数）", flush=True)
    print(f"  校准系数 k = {CALIBRATION_RATIO:.6f}", flush=True)
    print(f"  种群规模 = popsize×N = {PHASE1_POPSIZE}×{N_VARS} = {PHASE1_POPSIZE*N_VARS}", flush=True)
    print("=" * 60, flush=True)

    # === 阶段1 ===
    print(f"\n=== Phase 1: noon DE ({PHASE1_MAXITER} gen, pop {PHASE1_POPSIZE}) ===", flush=True)
    init_pop = build_init_pop(PHASE1_POPSIZE)
    print(f"Init pop shape: {init_pop.shape} (应为 ({PHASE1_POPSIZE}, {N_VARS}))", flush=True)

    # 快速测试：评估第一个成员
    print("Testing first eval...", flush=True)
    t0 = time.time()
    f0 = obj_phase1(init_pop[0])
    print(f"First eval: {f0:.8f} ({time.time()-t0:.1f}s)", flush=True)

    print("Starting DE Phase 1 (并行 workers=-1)...", flush=True)
    t0 = time.time()
    result_p1 = differential_evolution(
        obj_phase1, bounds=BOUNDS,
        maxiter=PHASE1_MAXITER, popsize=PHASE1_POPSIZE,
        tol=1e-8, seed=42,
        init=init_pop, polish=False,
        callback=cb_phase1,
        workers=-1,            # ← 多进程并行
        updating='deferred',   # ← 并行模式必须
    )
    print(f"Phase 1 done: {time.time()-t0:.0f}s, nfev={result_p1.nfev}", flush=True)

    # === 阶段2 ===
    PHASE = 2
    GEN_COUNTER = 0
    print(f"\n=== Phase 2: full-60 DE ({PHASE2_MAXITER} gen, pop {PHASE2_POPSIZE}) ===", flush=True)

    if hasattr(result_p1, 'population') and result_p1.population is not None:
        pop_p1 = result_p1.population
        if len(pop_p1) >= PHASE2_POPSIZE:
            idx = np.random.RandomState(43).choice(len(pop_p1), PHASE2_POPSIZE, replace=False)
            init_pop_p2 = pop_p1[idx]
        else:
            init_pop_p2 = np.tile(pop_p1[0:1], (PHASE2_POPSIZE, 1))
    else:
        init_pop_p2 = np.array([result_p1.x])
        np.random.seed(43)
        for _ in range(PHASE2_POPSIZE - 1):
            p = result_p1.x.copy()
            p *= np.random.uniform(0.95, 1.05, size=len(p))
            p = np.clip(p, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
            init_pop_p2 = np.vstack([init_pop_p2, p])

    t0 = time.time()
    result_p2 = differential_evolution(
        obj_phase2, bounds=BOUNDS,
        maxiter=PHASE2_MAXITER, popsize=PHASE2_POPSIZE,
        tol=1e-10, seed=43,
        init=init_pop_p2, polish=True,
        callback=cb_phase2,
        workers=-1,
        updating='deferred',
    )
    print(f"Phase 2 done: {time.time()-t0:.0f}s, nfev={result_p2.nfev}", flush=True)

    # === 最终评估 ===
    final = _simulate(result_p2.x, FULL_SCHEDULE)
    P_final = final['P_field_avg_MW']
    PA_final = P_final / final['total_area_m2'] if final['total_area_m2'] > 0 else 0
    print(f"\nFinal: P={P_final:.4f} MW, PA={PA_final*1e6:.2f} W/m2, "
          f"N={final['n_mirrors']}, "
          f"constraint={'OK' if P_final >= RATED_POWER_MW else 'FAIL'}", flush=True)

    # 输出 5 组参数
    print("\n5 组最优参数：", flush=True)
    per_group = []
    for g in range(N_GROUPS):
        W_g = float(result_p2.x[2 + 2*g])
        h_g = float(result_p2.x[2 + 2*g + 1])
        per_group.append({'group': g, 'W': round(W_g, 3), 'h': round(h_g, 3)})
        ring_start = int(g * N_RINGS / N_GROUPS)
        ring_end = int((g+1) * N_RINGS / N_GROUPS) - 1
        print(f"  组 {g} (环 {ring_start}-{ring_end}): W={W_g:.3f}m, h={h_g:.3f}m", flush=True)

    # === 保存 ===
    import pandas as pd
    pd.DataFrame(HISTORY).to_csv('outputs/data/problem3_convergence_v2.csv', index=False)
    print(f"\n收敛历史已保存: outputs/data/problem3_convergence_v2.csv", flush=True)

    # 判断收敛
    p2_entries = [h for h in HISTORY if h['phase'] == 2]
    conv = False
    if len(p2_entries) >= 3:
        recent = [h['best_fitness'] for h in p2_entries[-3:]]
        if max(recent) - min(recent) < abs(np.mean(recent)) * 0.001:
            conv = True

    # 判断结论
    baseline_PA = 525.85  # W/m²
    if P_final >= RATED_POWER_MW and PA_final * 1e6 > baseline_PA * 1.005:
        conclusion = (f"5组分组异构可提升 P/A 至 {PA_final*1e6:.1f} W/m2 "
                      f"(+{(PA_final*1e6/baseline_PA-1)*100:.1f}%)，满足60MW约束")
    elif P_final >= RATED_POWER_MW:
        conclusion = (f"在60MW约束下均一配置接近最优。经5组分组两阶段DE验证，"
                     f"异构无法显著突破均一基线 ({baseline_PA} W/m2)")
    else:
        conclusion = "异构提升单位面积效率但无法满足60MW约束，约束下均一最优"

    result_json = {
        'problem': 3,
        'type': 'heterogeneous_5groups',
        'method': 'two_phase_DE_calibrated',
        'calibration_ratio': CALIBRATION_RATIO,
        'n_groups': N_GROUPS,
        'n_rings': N_RINGS,
        'n_vars': N_VARS,
        'phase1': {
            'sampling': 'noon_12', 'maxiter': PHASE1_MAXITER,
            'popsize': PHASE1_POPSIZE, 'actual_pop': PHASE1_POPSIZE * N_VARS,
            'nfev': result_p1.nfev,
        },
        'phase2': {
            'sampling': 'full_60', 'maxiter': PHASE2_MAXITER,
            'popsize': PHASE2_POPSIZE, 'actual_pop': PHASE2_POPSIZE * N_VARS,
            'warm_start_from': 'phase1_population',
            'nfev': result_p2.nfev, 'converged': conv,
        },
        'conclusion': conclusion,
        'constraint_handling': 'power_penalty_1e4_calibrated',
        'best_params': {
            'x_t': round(float(result_p2.x[0]), 2),
            'y_t': round(float(result_p2.x[1]), 2),
        },
        'per_group': per_group,
        'best_P_avg_MW': round(P_final, 4),
        'best_P_per_area_W_m2': round(PA_final * 1e6, 2),
        'best_eta_avg': round(final['eta_field_avg'], 6),
        'best_n_mirrors': final['n_mirrors'],
        'best_total_area_m2': final['total_area_m2'],
    }
    with open('outputs/data/problem3_result_v2.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: outputs/data/problem3_result_v2.json", flush=True)

    print(f"\n总耗时: {(time.time()-t_start)/60:.1f} min", flush=True)
    print(f"结论: {conclusion}", flush=True)
