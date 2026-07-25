"""
改进 #8：夏至/冬至全天曲线。

以 1 小时分辨率计算夏至日和冬至日的功率变化曲线。
"""

import numpy as np
from typing import Dict, List

from csp_heliostat.config.constants import (
    LATITUDE_DEG, ALTITUDE_KM, TOWER_HEIGHT_M,
    RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M,
)
from csp_heliostat.core.solar_position import sun_state_batch
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.simulation.snapshot import simulate_one

# 夏至：6月21日 ≈ doy=172（31+28+31+30+31+21）
SUMMER_SOLSTICE_DOY = 172
# 冬至：12月22日 ≈ doy=356（或 355，非闰年）
WINTER_SOLSTICE_DOY = 356


def _D_from_doy(doy: int) -> int:
    """年积日 → 从春分起算的积日 D（使用 from_spring_equinox 约定）。"""
    return doy - 80


def compute_diurnal_curve(
    mirrors: List[Mirror],
    tower_xy: tuple = (0.0, 0.0),
    doy: int = SUMMER_SOLSTICE_DOY,
    hours: List[float] = None,
) -> Dict:
    """
    对给定日期，以指定小时分辨率计算全天功率变化。

    Args:
        mirrors: 定日镜列表
        tower_xy: 塔位置
        doy: 年积日（夏至=172, 冬至=356）
        hours: 小时列表，默认 range(6, 19) 即 6:00-18:00

    Returns:
        dict: {
            'doy': int,
            'day_label': 'summer_solstice' | 'winter_solstice',
            'hours': [6.0, 7.0, ...],
            'P_MW': [float, ...],
            'eta': [float, ...],
            'dni_list': [float, ...],
            'altitude_deg': [float, ...],
            'azimuth_deg': [float, ...],
        }
    """
    if hours is None:
        hours = list(range(6, 19))  # 6:00 - 18:00

    D = _D_from_doy(doy)
    n = len(hours)

    # 批量计算太阳位置
    D_arr = np.full(n, float(D))
    t_arr = np.array(hours, dtype=float)
    sun_states = sun_state_batch(LATITUDE_DEG, D_arr, t_arr, convention="from_spring_equinox")

    P_list = []
    eta_list = []
    dni_list = []
    alt_list = []
    az_list = []

    for i, hour in enumerate(hours):
        sun_single = type(sun_states)(
            altitude_deg=sun_states.altitude_deg[i:i+1],
            azimuth_deg=sun_states.azimuth_deg[i:i+1],
            sin_altitude=sun_states.sin_altitude[i:i+1],
            cos_altitude=sun_states.cos_altitude[i:i+1],
            is_daytime=sun_states.is_daytime[i:i+1],
        )

        result = simulate_one(
            mirrors, sun_single, tower_xy,
            TOWER_HEIGHT_M, RECEIVER_HEIGHT_M, RECEIVER_RADIUS_M, ALTITUDE_KM,
        )

        P_list.append(float(result['power_mw']))
        eta_list.append(float(result['eta_field_avg']))
        dni_list.append(float(result['dni']))
        alt_list.append(float(sun_states.altitude_deg[i]))
        az_list.append(float(sun_states.azimuth_deg[i]))

    day_label = 'summer_solstice' if doy == SUMMER_SOLSTICE_DOY else 'winter_solstice'

    return {
        'doy': doy,
        'day_label': day_label,
        'hours': hours,
        'P_MW': P_list,
        'eta': eta_list,
        'dni_list': dni_list,
        'altitude_deg': alt_list,
        'azimuth_deg': az_list,
    }