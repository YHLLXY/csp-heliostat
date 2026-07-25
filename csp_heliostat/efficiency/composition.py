"""
总效率装配 — 将五项效率合并为综合光学效率。

η_i = ρ · η_cos_i · η_sb_i · η_trunc_i · η_at_i

其中 ρ 为镜面反射率（0.92）。
"""

import numpy as np
from typing import List, Dict

from csp_heliostat.config.constants import REFLECTIVITY
from csp_heliostat.field.mirror import Mirror
from csp_heliostat.core.solar_position import SunState
from csp_heliostat.core.dni import direct_normal_irradiance

from .cosine import cosine_for_field
from .shadow_blocking import shadow_blocking_efficiency
from .truncation import truncation_efficiency
from .atmospheric import atmospheric_for_field


def total_efficiency(mirrors: List[Mirror],
                     sun: SunState,
                     tower_xy: tuple = (0.0, 0.0),
                     tower_height: float = 80.0,
                     receiver_height: float = 8.0,
                     receiver_radius: float = 3.5,
                     reflectivity: float = REFLECTIVITY) -> Dict[str, np.ndarray]:
    """
    计算全部五项效率并装配总效率。

    η_i = ρ · η_cos_i · η_sb_i · η_trunc_i · η_at_i

    Args:
        mirrors: 镜面列表
        sun: 太阳状态（单时刻）
        tower_xy: 塔 (x, y)
        tower_height: 塔高（m）
        receiver_height: 接收器高度（m）
        receiver_radius: 接收器半径（m）
        reflectivity: 反射率 ρ

    Returns:
        dict: {
            'eta_cos': (N,) 余弦效率,
            'eta_sb': (N,) 阴影遮挡效率,
            'eta_trunc': (N,) 截断效率,
            'eta_at': (N,) 大气透过率,
            'eta_ref': (N,) 反射率（常数）,
            'eta_total': (N,) 总光学效率,
        }
    """
    eta_cos = cosine_for_field(mirrors, sun, tower_xy, tower_height, receiver_height)
    eta_sb = shadow_blocking_efficiency(mirrors, sun)
    eta_trunc = truncation_efficiency(mirrors, sun, tower_xy, tower_height,
                                       receiver_height, receiver_radius)
    eta_at = atmospheric_for_field(mirrors, tower_xy, tower_height, receiver_height)
    eta_ref = np.full(len(mirrors), reflectivity)

    eta_total = reflectivity * eta_cos * eta_sb * eta_trunc * eta_at

    return {
        'eta_cos': eta_cos,
        'eta_sb': eta_sb,
        'eta_trunc': eta_trunc,
        'eta_at': eta_at,
        'eta_ref': eta_ref,
        'eta_total': eta_total,
    }


def field_efficiency(eta_total: np.ndarray, areas: np.ndarray) -> float:
    """
    镜场面积加权平均效率。

    η_field = Σ(η_i · A_i) / Σ A_i

    Args:
        eta_total: 每镜总效率，shape (N,)
        areas: 每镜面积，shape (N,)

    Returns:
        镜场平均光学效率（标量）
    """
    total_area = areas.sum()
    if total_area == 0:
        return 0.0
    return float(np.sum(eta_total * areas) / total_area)


def field_power(eta_total: np.ndarray,
                areas: np.ndarray,
                dni: float) -> float:
    """
    单时刻镜场输出热功率。

    P = Σ_i (A_i · DNI · η_i)

    Args:
        eta_total: 每镜总效率，shape (N,)
        areas: 每镜面积（m²），shape (N,)
        dni: 该时刻 DNI（W/m²）

    Returns:
        热功率（W）
    """
    return float(np.sum(areas * dni * eta_total))