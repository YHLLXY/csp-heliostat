"""
余弦效率 η_cos 计算。

η_cos = max(0, n̂ · ŝ) — 入射角余弦。
"""

import numpy as np
from typing import List

from csp_heliostat.core.geometry import (sun_unit_vector, receiver_vector_from_mirror,
                                          mirror_normal, cosine_efficiency)
from csp_heliostat.core.solar_position import SunState
from csp_heliostat.field.mirror import Mirror


def cosine_for_field(mirrors: List[Mirror],
                     sun: SunState,
                     tower_xy: tuple = (0.0, 0.0),
                     tower_height: float = 80.0,
                     receiver_height: float = 8.0) -> np.ndarray:
    """
    对每面镜在指定太阳位置下计算 η_cos。

    流程：
      1. 计算太阳方向 ŝ
      2. 计算接收器方向 r̂
      3. 计算法向 n̂ = normalize(ŝ + r̂)
      4. η_cos = max(0, n̂ · ŝ)

    Args:
        mirrors: 镜面列表
        sun: 太阳状态（单时刻）
        tower_xy: 塔/接收器顶部中心 (x, y)
        tower_height: 塔高（m）
        receiver_height: 接收器高度（m）

    Returns:
        η_cos 数组，shape (N,)
    """
    N = len(mirrors)
    if N == 0:
        return np.array([])

    if not sun.is_daytime[0]:
        return np.zeros(N)

    # 镜面坐标数组
    xy = np.array([[m.x, m.y] for m in mirrors])
    center_z = np.array([m.center_z for m in mirrors])

    # 接收器中心
    receiver_top_z = tower_height + receiver_height / 2.0  # 塔顶 + 半高 = 接收器中心

    # 太阳方向向量
    s_hat = sun_unit_vector(sun.altitude_deg, sun.azimuth_deg)  # (1, 3)
    s_hat = np.broadcast_to(s_hat, (N, 3))  # (N, 3)

    # 接收器方向向量
    r_hat = receiver_vector_from_mirror(xy, center_z, tower_xy, receiver_top_z)

    # 法向
    n_hat = mirror_normal(s_hat, r_hat)

    # 余弦效率
    eta_cos = cosine_efficiency(n_hat, s_hat)

    return eta_cos