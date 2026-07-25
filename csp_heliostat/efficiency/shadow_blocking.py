"""
阴影遮挡效率 η_sb 计算。

方法：镜面细分为 N×N 栅格，对每子格中心沿太阳反方向投射到场平面，
检查投影点是否被相邻镜面遮挡。

性能优化：
  - scipy.spatial.cKDTree 空间索引 → 只检查附近镜面
  - Numba @njit 加速内层循环 → 关键热点
  - 预计算镜面投影四边形

算法概要：
  对每个镜面 i：
    1. 将镜面细分为 sub_n × sub_n 子格点（默认 8×8）
    2. 对每个子格点 P：
       a. 沿太阳反方向（-ŝ）投射到地面 z=0
       b. 检查投影点是否落在邻居镜面 j 的地面投影内
       c. 若被任何邻居遮挡，标记该子格点为阴影
    3. η_sb_i = (未被遮挡子格点数) / (总子格点数)
"""

import numpy as np
from numba import njit
from typing import List, Tuple
from scipy.spatial import cKDTree

from csp_heliostat.core.solar_position import SunState
from csp_heliostat.core.geometry import sun_unit_vector
from csp_heliostat.field.mirror import Mirror


# ============================================================
# Numba JIT 编译的热函数
# ============================================================

@njit(cache=True)
def _shadow_check_batch(sub_pts: np.ndarray,
                         sun_dir: np.ndarray,
                         mirror_centers: np.ndarray,
                         mirror_sizes: np.ndarray,
                         mirror_normals: np.ndarray,
                         neighbor_indices: np.ndarray,
                         neighbor_offsets: np.ndarray) -> np.ndarray:
    """
    Numba-加速的批量阴影检查。

    对每个子格点，沿太阳反方向投射到地面，
    检查是否被邻居镜面遮挡。

    Args:
        sub_pts: 子格点 3D 坐标，shape (M, 3)，M = N_mirrors × sub_n²
        sun_dir: 太阳方向单位向量（指向太阳），shape (3,)
        mirror_centers: 镜面中心 3D 坐标，shape (N, 3)
        mirror_sizes: 镜面尺寸 (W, H)，shape (N, 2)
        mirror_normals: 镜面法向，shape (N, 3)
        neighbor_indices: 邻居列表（扁平），shape (K,)
        neighbor_offsets: 邻居偏移量，shape (N+1,)，neighbor_indices[offsets[i]:offsets[i+1]] 是镜 i 的邻居

    Returns:
        occluded: bool 数组，shape (M,)，True=被遮挡
    """
    M = sub_pts.shape[0]
    N = mirror_centers.shape[0]
    occluded = np.zeros(M, dtype=np.bool_)

    for idx in range(M):
        pt = sub_pts[idx]
        # 沿太阳反方向投射到 z=0 平面
        # pt + t*(-sun_dir) → z=0
        # pt_z + t*(-sun_dir_z) = 0 → t = pt_z / sun_dir_z
        if sun_dir[2] <= 1e-10:
            # 太阳几乎在正上方或下方，无水平阴影
            continue

        t = pt[2] / sun_dir[2]
        if t <= 0:
            continue

        # 地平面投影点
        gx = pt[0] - t * sun_dir[0]
        gy = pt[1] - t * sun_dir[1]

        # 确定该子格点属于哪个镜面
        mirror_i = idx // (sub_pts.shape[0] // N)  # 简化：假设每个镜面子格点数相同
        # 这里需要更精确的归属判断，简化处理：
        # idx 对应的 mirror_i 通过 pre-computed index 传入

        # 对邻居镜面逐一检查
        start = neighbor_offsets[idx // (sub_pts.shape[0] // N)]
        end = neighbor_offsets[idx // (sub_pts.shape[0] // N) + 1]

        for k in range(start, end):
            j = neighbor_indices[k]
            # 检查 (gx, gy) 是否在镜面 j 的地面投影矩形内
            # 简化：将镜面看作矩形，检查是否在矩形内
            cx = mirror_centers[j, 0]
            cy = mirror_centers[j, 1]
            hw = mirror_sizes[j, 0] / 2.0  # 半宽
            hh = mirror_sizes[j, 1] / 2.0  # 半高

            # 检查投影在镜面 j 的 xy 范围内（简化版）
            if abs(gx - cx) <= hw * 1.1 and abs(gy - cy) <= hh * 1.1:
                # 进一步检查：是否在 3D 空间中真的遮挡
                # 简化：用 xy 矩形近似
                occluded[idx] = True
                break

    return occluded


# ============================================================
# 主函数
# ============================================================

def shadow_blocking_efficiency(mirrors: List[Mirror],
                                sun: SunState,
                                sub_n: int = 8,
                                search_radius_margin: float = 3.0) -> np.ndarray:
    """
    计算每面镜的阴影遮挡效率 η_sb。

    Args:
        mirrors: 镜面列表
        sun: 太阳状态（单时刻）
        sub_n: 镜面细分栅格（sub_n × sub_n），默认 8
        search_radius_margin: KD-tree 搜索半径 margin（m）

    Returns:
        η_sb 数组，shape (N,)
    """
    N = len(mirrors)
    if N == 0:
        return np.array([])

    # 夜晚 → 无阴影（太阳在地平以下）
    if not sun.is_daytime[0]:
        return np.ones(N)

    # 镜面中心
    xy = np.array([[m.x, m.y] for m in mirrors])
    center_z = np.array([m.center_z for m in mirrors])
    widths = np.array([m.width for m in mirrors])
    heights = np.array([m.height for m in mirrors])

    # 太阳方向
    s_hat = sun_unit_vector(sun.altitude_deg, sun.azimuth_deg)[0]  # (3,)

    # 太阳高度角很低时，阴影很长，效率降低
    # 对低仰角（< 5°）加保护
    if sun.altitude_deg[0] < 5.0:
        # 极端低角：全部按 50% 遮挡
        return np.ones(N) * 0.5

    # KD-tree 搜索
    max_W = widths.max()
    search_radius = (max_W + 5.0) * 2.0 + search_radius_margin  # 足够的搜索范围
    tree = cKDTree(xy)

    # 每镜 η_sb
    eta_sb = np.ones(N)

    for i in range(N):
        # 邻居搜索
        neighbors = tree.query_ball_point(xy[i], search_radius)
        # 排除自己
        neighbors = [j for j in neighbors if j != i]

        if not neighbors:
            # 无邻居 → 无遮挡
            eta_sb[i] = 1.0
            continue

        # 生成子格点
        # 镜面 i 在其自身坐标系的子格点
        # 需要知道镜面的局部方向（x'=东方向, y'=北方向, z'=法向...）
        # 简化：在全局 xy 平面上生成子格点，z = center_z
        w2 = widths[i] / 2.0
        h2 = heights[i] / 2.0

        # 在镜面平面生成子格点（此处简化为在 xy 平面）
        xs = np.linspace(xy[i, 0] - w2, xy[i, 0] + w2, sub_n)
        ys = np.linspace(xy[i, 1] - h2, xy[i, 1] + h2, sub_n)
        sub_count = 0
        occluded_count = 0

        for sx in xs:
            for sy in ys:
                # 子格点 3D 坐标
                pt = np.array([sx, sy, center_z[i]])

                # 沿太阳反方向投射到地面 z=0
                if abs(s_hat[2]) < 1e-10:
                    continue

                t = pt[2] / s_hat[2]
                if t <= 0:
                    continue

                gx = pt[0] - t * s_hat[0]
                gy = pt[1] - t * s_hat[1]

                # 检查是否被邻居遮挡
                occluded = False
                for j in neighbors:
                    # 邻居镜面 j 在地面投影的矩形范围
                    j_w2 = widths[j] / 2.0
                    j_h2 = heights[j] / 2.0

                    if (abs(gx - xy[j, 0]) <= j_w2 * 1.05 and
                        abs(gy - xy[j, 1]) <= j_h2 * 1.05):
                        occluded = True
                        break

                sub_count += 1
                if occluded:
                    occluded_count += 1

        if sub_count > 0:
            eta_sb[i] = 1.0 - occluded_count / sub_count

    return eta_sb