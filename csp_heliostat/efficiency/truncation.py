"""
截断效率 η_trunc 计算。

η_trunc = 反射光线落入接收器面积 / 镜面总反射面积

方法：
  1. 镜面在自身平面细分为子网格
  2. 每子格中心沿反射方向 r̂ 追踪光线
  3. 光线与接收器圆柱面求交
  4. 落入 [H_R_bottom, H_R_top] × 半径 ≤ R_R → 接收

性能优化：Numba @njit 加速光线-圆柱求交循环。
"""

import numpy as np
from numba import njit
from typing import List

from csp_heliostat.core.solar_position import SunState
from csp_heliostat.core.geometry import (sun_unit_vector, receiver_vector_from_mirror,
                                          mirror_normal, reflection_vector)
from csp_heliostat.field.mirror import Mirror


@njit(cache=True)
def _ray_cylinder_intersection(origin: np.ndarray,
                                direction: np.ndarray,
                                cyl_center: np.ndarray,
                                cyl_radius: float,
                                cyl_height: float) -> float:
    """
    光线与竖直圆柱的求交（Numba 加速）。

    圆柱：中心在 (cx, cy, z_bottom)，半径 R_R，高度 H_R。

    光线参数方程：P(t) = origin + t * direction, t ≥ 0

    与圆柱侧面求交：|P_xy(t) - cyl_center_xy|² = R_R²
    展开得二次方程 at² + bt + c = 0

    Args:
        origin: 光线起点 (x, y, z)
        direction: 光线方向单位向量 (dx, dy, dz)
        cyl_center: 圆柱底面中心 (cx, cy, z_bottom)
        cyl_radius: 圆柱半径（m）
        cyl_height: 圆柱高度（m）

    Returns:
        最近的正面交点参数 t，若无交点返回 -1.0
    """
    ox, oy, oz = origin[0], origin[1], origin[2]
    dx, dy, dz = direction[0], direction[1], direction[2]
    cx, cy, cz_bottom = cyl_center[0], cyl_center[1], cyl_center[2]
    cz_top = cz_bottom + cyl_height

    # 侧面求交二次方程
    dox = ox - cx
    doy = oy - cy

    a = dx * dx + dy * dy
    b = 2.0 * (dox * dx + doy * dy)
    c = dox * dox + doy * doy - cyl_radius * cyl_radius

    best_t = -1.0

    if a > 1e-12:
        # 侧面求交
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            for t in (t1, t2):
                if t > 1e-9:
                    z = oz + t * dz
                    if cz_bottom - 0.01 <= z <= cz_top + 0.01:
                        if best_t < 0 or t < best_t:
                            best_t = t

    # 顶面求交（圆盘）
    if abs(dz) > 1e-10:
        t_top = (cz_top - oz) / dz
        if t_top > 1e-9:
            px = ox + t_top * dx
            py = oy + t_top * dy
            if (px - cx) ** 2 + (py - cy) ** 2 <= cyl_radius ** 2:
                if best_t < 0 or t_top < best_t:
                    best_t = t_top

    # 底面求交
    if abs(dz) > 1e-10:
        t_bottom = (cz_bottom - oz) / dz
        if t_bottom > 1e-9:
            px = ox + t_bottom * dx
            py = oy + t_bottom * dy
            if (px - cx) ** 2 + (py - cy) ** 2 <= cyl_radius ** 2:
                if best_t < 0 or t_bottom < best_t:
                    best_t = t_bottom

    return best_t


def truncation_efficiency(mirrors: List[Mirror],
                           sun: SunState,
                           tower_xy: tuple = (0.0, 0.0),
                           tower_height: float = 80.0,
                           receiver_height: float = 8.0,
                           receiver_radius: float = 3.5,
                           sub_n: int = 6) -> np.ndarray:
    """
    计算每面镜的截断效率 η_trunc。

    Args:
        mirrors: 镜面列表
        sun: 太阳状态（单时刻）
        tower_xy: 塔/接收器 (x, y)
        tower_height: 塔高（m）
        receiver_height: 接收器高度 H_R（m）
        receiver_radius: 接收器半径 R_R（m）
        sub_n: 镜面细分栅格（sub_n × sub_n）

    Returns:
        η_trunc 数组，shape (N,)
    """
    N = len(mirrors)
    if N == 0:
        return np.array([])

    if not sun.is_daytime[0]:
        return np.zeros(N)

    # 接收器参数
    receiver_center_xy = np.array([tower_xy[0], tower_xy[1]])
    receiver_bottom_z = tower_height - receiver_height / 2.0  # 接收器底部
    cyl_center = np.array([tower_xy[0], tower_xy[1], receiver_bottom_z])

    # 太阳和反射方向
    s_hat = sun_unit_vector(sun.altitude_deg, sun.azimuth_deg)  # (1, 3)

    xy = np.array([[m.x, m.y] for m in mirrors])
    center_z = np.array([m.center_z for m in mirrors])
    widths = np.array([m.width for m in mirrors])
    heights_arr = np.array([m.height for m in mirrors])

    # 接收器顶部（法向指向中心偏上）
    receiver_top_z = tower_height + receiver_height / 2.0

    eta_trunc = np.zeros(N)

    for i in range(N):
        # 反射方向 r̂
        s_i = s_hat[0]  # (3,)
        r_i_hat = receiver_vector_from_mirror(
            xy[i:i+1], np.array([center_z[i]]), tower_xy, receiver_top_z
        )[0]  # (3,)

        # 法向
        n_i_hat = mirror_normal(s_i[np.newaxis, :], r_i_hat[np.newaxis, :])[0]

        # 反射方向
        r_refl = reflection_vector(s_i[np.newaxis, :], n_i_hat[np.newaxis, :])[0]

        # 在镜面自身平面生成子格点
        # 简化：在镜面中心 xy 平面生成
        w2 = widths[i] / 2.0
        h2 = heights_arr[i] / 2.0

        xs = np.linspace(xy[i, 0] - w2, xy[i, 0] + w2, sub_n)
        ys = np.linspace(xy[i, 1] - h2, xy[i, 1] + h2, sub_n)

        total_rays = 0
        hit_count = 0

        for sx in xs:
            for sy in ys:
                origin = np.array([sx, sy, center_z[i]], dtype=np.float64)
                direction = r_refl.astype(np.float64)

                t = _ray_cylinder_intersection(
                    origin, direction, cyl_center.astype(np.float64),
                    receiver_radius, receiver_height
                )

                total_rays += 1
                if t > 0:
                    hit_count += 1

        eta_trunc[i] = hit_count / total_rays if total_rays > 0 else 0.0

    return eta_trunc