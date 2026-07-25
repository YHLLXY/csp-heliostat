"""
单时刻仿真 — 对所有镜面计算五项效率与输出功率。

simulate_one() 是整条仿真链路的"原子操作"。
"""

import numpy as np
from typing import List, Dict, Optional

from csp_heliostat.core.solar_position import SunState, sun_state_batch
from csp_heliostat.core.dni import direct_normal_irradiance
from csp_heliostat.config.constants import (LATITUDE_DEG, ALTITUDE_KM,
                                              REFLECTIVITY, TOWER_HEIGHT_M,
                                              RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M)
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.efficiency.composition import total_efficiency, field_efficiency, field_power


def simulate_one(mirrors: List[Mirror],
                 sun: SunState,
                 tower_xy: tuple = (0.0, 0.0),
                 tower_height: float = TOWER_HEIGHT_M,
                 receiver_height: float = RECEIVER_HEIGHT_M,
                 receiver_radius: float = RECEIVER_RADIUS_M,
                 H_km: float = ALTITUDE_KM,
                 reflectivity: float = REFLECTIVITY) -> Dict:
    """
    单时刻仿真 — 核心"原子操作"。

    对一个太阳状态，对所有镜面计算：
      - 五项效率子项
      - 总光学效率 η_i
      - DNI
      - 镜场总输出热功率

    Args:
        mirrors: 镜面列表
        sun: 太阳状态（单时刻，shape=(1,)）
        tower_xy: 塔 (x, y)
        tower_height: 塔高（m）
        receiver_height: 接收器高度（m）
        receiver_radius: 接收器半径（m）
        H_km: 海拔（km）
        reflectivity: 反射率 ρ

    Returns:
        dict: {
            'eta_cos': ndarray (N,),
            'eta_sb': ndarray (N,),
            'eta_trunc': ndarray (N,),
            'eta_at': ndarray (N,),
            'eta_ref': ndarray (N,),
            'eta_total': ndarray (N,),
            'dni': float,
            'power_w': float,
            'power_mw': float,
            'eta_field_avg': float,
            'altitude_deg': float,
            'azimuth_deg': float,
        }
    """
    N = len(mirrors)
    if N == 0:
        return {
            'eta_cos': np.array([]),
            'eta_sb': np.array([]),
            'eta_trunc': np.array([]),
            'eta_at': np.array([]),
            'eta_ref': np.array([]),
            'eta_total': np.array([]),
            'dni': 0.0,
            'power_w': 0.0,
            'power_mw': 0.0,
            'eta_field_avg': 0.0,
            'altitude_deg': float(sun.altitude_deg[0]),
            'azimuth_deg': float(sun.azimuth_deg[0]),
        }

    # 夜晚
    if not sun.is_daytime[0]:
        areas = np.array([m.area for m in mirrors])
        return {
            'eta_cos': np.zeros(N),
            'eta_sb': np.ones(N),
            'eta_trunc': np.zeros(N),
            'eta_at': np.ones(N),
            'eta_ref': np.full(N, reflectivity),
            'eta_total': np.zeros(N),
            'dni': 0.0,
            'power_w': 0.0,
            'power_mw': 0.0,
            'eta_field_avg': 0.0,
            'altitude_deg': float(sun.altitude_deg[0]),
            'azimuth_deg': float(sun.azimuth_deg[0]),
        }

    # 五项效率
    eff = total_efficiency(mirrors, sun, tower_xy, tower_height,
                           receiver_height, receiver_radius, reflectivity)

    # DNI
    dni = float(direct_normal_irradiance(sun.altitude_deg, H_km)[0])

    # 功率
    areas = np.array([m.area for m in mirrors])
    power_w = field_power(eff['eta_total'], areas, dni)
    power_mw = power_w / 1e6

    # 镜场平均效率
    eta_field = field_efficiency(eff['eta_total'], areas)

    # 各分项效率的面积加权镜场均值（供分项效率拆解用）
    def _field_mean(arr):
        return float(np.sum(arr * areas) / areas.sum()) if areas.sum() > 0 else 0.0

    return {
        'eta_cos': eff['eta_cos'],
        'eta_sb': eff['eta_sb'],
        'eta_trunc': eff['eta_trunc'],
        'eta_at': eff['eta_at'],
        'eta_ref': eff['eta_ref'],
        'eta_total': eff['eta_total'],
        'dni': dni,
        'power_w': power_w,
        'power_mw': power_mw,
        'eta_field_avg': eta_field,
        'eta_cos_avg': _field_mean(eff['eta_cos']),
        'eta_sb_avg': _field_mean(eff['eta_sb']),
        'eta_trunc_avg': _field_mean(eff['eta_trunc']),
        'eta_at_avg': _field_mean(eff['eta_at']),
        'altitude_deg': float(sun.altitude_deg[0]),
        'azimuth_deg': float(sun.azimuth_deg[0]),
    }