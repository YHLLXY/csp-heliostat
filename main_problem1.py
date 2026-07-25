"""
问题一：正向模拟（基准场景）

参数：
  - 塔在场地中心 (0, 0)
  - 镜面 6m x 6m 正方形
  - 安装高度 h_i = 4m
  - 径向同心布局（18 环）
  - 60 个采样时刻

目标：
  - 计算年平均光学效率 eta_field
  - 计算年平均输出热功率 P_field

验证：
  - 春分正午 alpha_s ~ 50.6 deg (solar_position 单元测试)
  - DNI 应在 1000-1100 W/m2 范围
  - P_field 应在 30-90 MW 范围（视布局）
"""

import sys
import os

# Fix Windows GBK encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

from csp_heliostat.config.constants import (LATITUDE_DEG, ALTITUDE_KM,
                                              TOWER_HEIGHT_M, RECEIVER_HEIGHT_M,
                                              RECEIVER_RADIUS_M, REFLECTIVITY,
                                              FIELD_RADIUS_M, EXCLUSION_RADIUS_M)
from csp_heliostat.field.layout import radial_layout, exclude_zone, field_boundary_filter
from csp_heliostat.field.constraints import (spacing_check, count_spacing_violations,
                                               total_mirrors_count, total_reflective_area)
from csp_heliostat.simulation.annual import annual_simulation, annual_results_to_dataframe
from csp_heliostat.simulation.snapshot import simulate_one
from csp_heliostat.core.solar_position import sun_state_batch, sun_position_single


def run_smoke_test():
    """冒烟测试：验证春分正午的太阳位置和 DNI"""
    print("=" * 60)
    print("冒烟测试：物理基础验证")
    print("=" * 60)

    # 春分正午
    alpha, gamma = sun_position_single(LATITUDE_DEG, D=0, t_hour=12,
                                        convention="from_spring_equinox")
    print(f"  春分正午 (D=0, t=12):")
    print(f"    太阳高度角 alpha_s = {alpha:.2f} deg (期望约 50.6 deg)")
    print(f"    太阳方位角 gamma_s = {gamma:.2f} deg (期望约 0 deg)")

    # DNI 检查
    from csp_heliostat.core.dni import direct_normal_irradiance
    dni = direct_normal_irradiance(np.array([alpha]), ALTITUDE_KM)
    print(f"    DNI = {dni[0]:.1f} W/m2 (期望 1000-1100)")

    # 夏至正午
    alpha_s, gamma_s = sun_position_single(LATITUDE_DEG, D=93, t_hour=12,
                                            convention="from_spring_equinox")
    print(f"  夏至正午 (D~93, t=12):")
    print(f"    alpha_s = {alpha_s:.2f} deg (期望约 73.6 deg)")

    # 秋分正午
    alpha_w, gamma_w = sun_position_single(LATITUDE_DEG, D=184, t_hour=12,
                                            convention="from_spring_equinox")
    print(f"  秋分正午 (D~184, t=12):")
    print(f"    alpha_s = {alpha_w:.2f} deg (期望约 50.6 deg)")

    print()


def run_snapshot_demo():
    """单时刻仿真演示"""
    print("=" * 60)
    print("单时刻仿真演示：春分正午")
    print("=" * 60)

    # 生成镜场（少一些环，快速演示）
    mirrors = radial_layout(
        n_rings=10, W=6.0, H=6.0, install_h=4.0,
        R_inner=EXCLUSION_RADIUS_M,
        R_outer=FIELD_RADIUS_M,
    )
    # 过滤禁区和边界
    mirrors = exclude_zone(mirrors, exclusion_r=EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, field_radius=FIELD_RADIUS_M)

    print(f"  镜面总数: {len(mirrors)}")
    print(f"  总反射面积: {total_reflective_area(mirrors):.0f} m²")

    # 间距检查
    n_violations = count_spacing_violations(mirrors)
    if n_violations > 0:
        print(f"  !! 间距违规: {n_violations} 对")
    else:
        print(f"  [OK] 间距约束满足")

    # 春分正午单时刻仿真
    sun = sun_state_batch(LATITUDE_DEG, D=0.0, t_hour=12.0,
                           convention="from_spring_equinox")

    t0 = time.time()
    result = simulate_one(mirrors, sun, tower_xy=(0, 0))
    elapsed = time.time() - t0

    print(f"  计算耗时: {elapsed:.2f} s")
    print(f"  DNI: {result['dni']:.1f} W/m2")
    print(f"  镜场平均光学效率: {result['eta_field_avg']:.4f}")
    print(f"  输出热功率: {result['power_mw']:.2f} MW")
    print(f"  效率子项 -- eta_cos mean: {result['eta_cos'].mean():.4f}")
    print(f"            eta_sb mean:  {result['eta_sb'].mean():.4f}")
    print(f"            eta_trunc mean: {result['eta_trunc'].mean():.4f}")
    print(f"            eta_at mean:   {result['eta_at'].mean():.4f}")
    print()


def main():
    """问题一主入口 — 完整年度仿真"""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     问题一：定日镜场基准场景正向模拟                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ---- 冒烟测试 ----
    run_smoke_test()

    # ---- 单时刻演示 ----
    run_snapshot_demo()

    # ---- 完整年度仿真 ----
    print("=" * 60)
    print("完整年度仿真：60 时刻加权平均")
    print("=" * 60)

    # 生成完整镜场（问题一基准：18 环）
    print("  生成镜场布局...")
    mirrors = radial_layout(
        n_rings=18, W=6.0, H=6.0, install_h=4.0,
        R_inner=EXCLUSION_RADIUS_M,
        R_outer=FIELD_RADIUS_M,
    )
    mirrors = exclude_zone(mirrors, exclusion_r=EXCLUSION_RADIUS_M)
    mirrors = field_boundary_filter(mirrors, field_radius=FIELD_RADIUS_M)

    print(f"  镜面总数: {total_mirrors_count(mirrors)}")
    print(f"  总反射面积: {total_reflective_area(mirrors):.0f} m2")

    # 间距检查
    n_v = count_spacing_violations(mirrors)
    if n_v > 0:
        print(f"  !! 间距违规: {n_v} 对（共 {len(mirrors)} 面镜）")
    else:
        print(f"  [OK] 间距约束全部满足")

    # 年度仿真
    print(f"\n  开始 60 时刻年度仿真...")
    t0 = time.time()

    result = annual_simulation(
        mirrors,
        tower_xy=(0.0, 0.0),
        tower_height=TOWER_HEIGHT_M,
        receiver_height=RECEIVER_HEIGHT_M,
        receiver_radius=RECEIVER_RADIUS_M,
        verbose=True,
    )

    elapsed = time.time() - t0

    # ---- 输出结果 ----
    print()
    print("=" * 60)
    print("问题一 最终结果")
    print("=" * 60)
    print(f"  镜面总数:           {result['n_mirrors']}")
    print(f"  总反射面积:         {result['total_area_m2']:.0f} m2")
    print(f"  年平均光学效率 eta: {result['eta_field_avg']:.6f}")
    print(f"  年平均输出功率 P:   {result['P_field_avg_MW']:.2f} MW")
    print(f"  年仿真耗时:          {elapsed:.1f} s")
    print()

    # 月度汇总
    df = annual_results_to_dataframe(result)
    monthly = df.groupby('month').agg({
        'eta_field_avg': 'mean',
        'power_mw': 'mean',
        'dni': 'mean',
    }).round(4)

    print("  月度平均（跨 5 个时刻）：")
    for month, row in monthly.iterrows():
        print(f"    {month:2d}月  eta={row['eta_field_avg']:.4f}  "
              f"P={row['power_mw']:.2f} MW  DNI={row['dni']:.0f} W/m2")

    print()

    # 合理性检查
    print("合理性检查：")
    if 30 <= result['P_field_avg_MW'] <= 90:
        print(f"  [OK] P={result['P_field_avg_MW']:.1f} MW 在 30-90 MW 预期范围内")
    else:
        print(f"  [!!] P={result['P_field_avg_MW']:.1f} MW 偏离预期范围 [30, 90] MW")

    if 0.3 <= result['eta_field_avg'] <= 0.7:
        print(f"  [OK] eta={result['eta_field_avg']:.4f} 在合理范围 [0.3, 0.7]")
    else:
        print(f"  [!!] eta={result['eta_field_avg']:.4f} 偏离预期范围 [0.3, 0.7]")

    print()
    print("问题一完成 ✅")
    return result


if __name__ == '__main__':
    main()