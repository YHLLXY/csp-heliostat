"""
年平均仿真 — 60 时刻加权积分。

对全年 60 个采样时刻（12个月 × 5个时间点）逐一仿真，
按 weight = max(0, sin α_s) 加权计算年平均光学效率与平均输出功率。
"""

import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

from csp_heliostat.config.constants import (LATITUDE_DEG, ALTITUDE_KM,
                                              TOWER_HEIGHT_M, RECEIVER_HEIGHT_M,
                                              RECEIVER_RADIUS_M, REFLECTIVITY)
from csp_heliostat.config.sampling import build_sampling_schedule, solar_weights, SamplePoint
from csp_heliostat.core.solar_position import sun_state_batch
from csp_heliostat.field.mirror import Mirror
from .snapshot import simulate_one


def annual_simulation(mirrors: List[Mirror],
                      tower_xy: tuple = (0.0, 0.0),
                      tower_height: float = TOWER_HEIGHT_M,
                      receiver_height: float = RECEIVER_HEIGHT_M,
                      receiver_radius: float = RECEIVER_RADIUS_M,
                      phi_deg: float = LATITUDE_DEG,
                      H_km: float = ALTITUDE_KM,
                      schedule: Optional[List[SamplePoint]] = None,
                      verbose: bool = True,
                      parallel: bool = False) -> Dict:
    """
    对 60 个采样时刻逐一仿真，按 sin α_s 权重积分。

    weight = max(0, sin α_s)，归一化后加权平均。
    物理意义：模拟实际日照辐射的相对贡献，避免早晚极端低角时刻主导平均。

    Args:
        mirrors: 镜面列表
        tower_xy: 塔 (x, y)
        tower_height: 塔高（m）
        receiver_height: 接收器高度（m）
        receiver_radius: 接收器半径（m）
        phi_deg: 纬度（°N）
        H_km: 海拔（km）
        schedule: 采样时间表。如果为 None，自动生成 60 时刻。
        verbose: 是否显示进度条
        parallel: 是否使用 joblib 并行（60 时刻可并行）

    Returns:
        dict: {
            'eta_field_avg': float,          # 年平均光学效率（加权）
            'P_field_avg_MW': float,         # 年平均输出热功率（MW）
            'total_area_m2': float,          # 总镜面面积
            'n_mirrors': int,                # 镜面数
            'per_sample': List[dict],        # 每时刻详情
            'schedule': List[SamplePoint],   # 采样表
        }
    """
    if schedule is None:
        schedule = build_sampling_schedule()

    # 预计算所有时刻的太阳状态
    D_array = np.array([s.D_spring for s in schedule], dtype=float)
    t_array = np.array([s.hour for s in schedule], dtype=float)

    sun_states = sun_state_batch(phi_deg, D_array, t_array,
                                  convention="from_spring_equinox")

    # 计算权重
    weights = solar_weights(sun_states.sin_altitude)

    # 更新 schedule 中的权重
    for i, s in enumerate(schedule):
        s.weight = float(weights[i])

    # 逐时刻仿真
    per_sample = []
    n_moments = len(schedule)
    iterator = list(enumerate(schedule))
    if verbose:
        iterator = tqdm(iterator, desc="年度仿真", total=n_moments)

    total_area = sum(m.area for m in mirrors)

    weighted_eta_sum = 0.0
    weighted_power_sum = 0.0
    weight_sum = 0.0

    for idx, sample in iterator:
        # 构建该时刻的 SunState（单时刻切片）
        sun_single = type(sun_states)(
            altitude_deg=sun_states.altitude_deg[idx:idx+1],
            azimuth_deg=sun_states.azimuth_deg[idx:idx+1],
            sin_altitude=sun_states.sin_altitude[idx:idx+1],
            cos_altitude=sun_states.cos_altitude[idx:idx+1],
            is_daytime=sun_states.is_daytime[idx:idx+1],
        )

        result = simulate_one(mirrors, sun_single, tower_xy,
                              tower_height, receiver_height, receiver_radius, H_km)

        w = float(weights[idx])

        weighted_eta_sum += result['eta_field_avg'] * w
        weighted_power_sum += result['power_mw'] * w
        weight_sum += w

        per_sample.append({
            'month': sample.month,
            'day': sample.day,
            'hour': sample.hour,
            'D_spring': sample.D_spring,
            'weight': w,
            **{k: result[k] for k in ['eta_field_avg', 'power_mw', 'dni',
                                        'altitude_deg', 'azimuth_deg']},
            'eta_cos_avg': result.get('eta_cos_avg', 0.0),
            'eta_sb_avg': result.get('eta_sb_avg', 0.0),
            'eta_trunc_avg': result.get('eta_trunc_avg', 0.0),
            'eta_at_avg': result.get('eta_at_avg', 0.0),
        })

    # 加权平均
    eta_field_avg = weighted_eta_sum / weight_sum if weight_sum > 0 else 0.0
    P_field_avg_MW = weighted_power_sum / weight_sum if weight_sum > 0 else 0.0

    return {
        'eta_field_avg': eta_field_avg,
        'P_field_avg_MW': P_field_avg_MW,
        'total_area_m2': total_area,
        'n_mirrors': len(mirrors),
        'per_sample': per_sample,
        'schedule': schedule,
    }


def annual_results_to_dataframe(result: Dict) -> pd.DataFrame:
    """
    将 annual_simulation 的结果转为 pandas DataFrame，便于分析。

    Args:
        result: annual_simulation 的返回值

    Returns:
        DataFrame: 每时刻一行，包含效率、功率、DNI 等
    """
    rows = []
    for s in result['per_sample']:
        rows.append({
            'month': s['month'],
            'hour': s['hour'],
            'eta_field_avg': s['eta_field_avg'],
            'power_mw': s['power_mw'],
            'dni': s['dni'],
            'altitude_deg': s['altitude_deg'],
            'azimuth_deg': s['azimuth_deg'],
            'weight': s['weight'],
        })
    return pd.DataFrame(rows)