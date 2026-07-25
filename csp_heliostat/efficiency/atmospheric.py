"""
大气透过效率 η_at 计算（薄封装层）。

直接委托 core.atmosphere 模块。
"""

import numpy as np
from typing import List

from csp_heliostat.core.atmosphere import atmospheric_transmittance, slant_distance
from csp_heliostat.field.mirror import Mirror


def atmospheric_for_field(mirrors: List[Mirror],
                          tower_xy: tuple = (0.0, 0.0),
                          tower_height: float = 80.0,
                          receiver_height: float = 8.0) -> np.ndarray:
    """
    计算每面镜到接收器的大气透过率 η_at。

    Args:
        mirrors: 镜面列表
        tower_xy: 塔 (x, y)
        tower_height: 塔高（m）
        receiver_height: 接收器高度（m）

    Returns:
        η_at 数组，shape (N,)
    """
    N = len(mirrors)
    if N == 0:
        return np.array([])

    xy = np.array([[m.x, m.y] for m in mirrors])
    center_z = np.array([m.center_z for m in mirrors])

    receiver_center_z = tower_height  # 接收器中心高度 = 塔高

    distances = slant_distance(xy, center_z, tower_xy, receiver_center_z)

    return atmospheric_transmittance(distances)