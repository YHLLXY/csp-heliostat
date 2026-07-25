"""
镜场约束检查 — 间距、禁区、边界等。

约束条件：
  1. 相邻距：d_ij > W_i + 5（或 W_j + 5，取大者）
  2. 禁区：镜面中心到塔距离 ≥ 100m
  3. 场地边界：镜面中心在半径 350m 的圆内
  4. 安装高度：h_i ∈ [2, 6]
  5. 镜面尺寸：W ∈ [2, 8], H ∈ [2, 8]
"""

import numpy as np
from typing import List, Tuple
from .mirror import Mirror


def spacing_check(mirrors: List[Mirror],
                  min_clearance: float = 5.0) -> List[Tuple[int, int]]:
    """
    检查相邻距约束：d_ij > W_i + min_clearance（以较大镜宽为准）。

    使用 KD-tree 加速邻域搜索（O(N log N) 而非 O(N²)）。

    Args:
        mirrors: 镜面列表
        min_clearance: 最小净距（m），默认 5.0

    Returns:
        违规对下标列表 [(i, j), ...]
    """
    from scipy.spatial import cKDTree

    N = len(mirrors)
    if N <= 1:
        return []

    xy = np.array([[m.x, m.y] for m in mirrors])

    # 搜索半径 = 最大镜宽 + clearance（超出此距离不可能违规）
    max_W = max(m.width for m in mirrors)
    search_radius = max_W + min_clearance + 1.0  # +1 安全margin

    tree = cKDTree(xy)
    violations = []

    for i in range(N):
        # 在搜索半径内找邻居
        neighbors = tree.query_ball_point(xy[i], search_radius)

        for j in neighbors:
            if j <= i:
                continue  # 每对只查一次

            dist = np.sqrt((xy[i, 0] - xy[j, 0])**2 +
                           (xy[i, 1] - xy[j, 1])**2)

            # 取较大镜宽
            W_max = max(mirrors[i].width, mirrors[j].width)
            min_dist = W_max + min_clearance

            if dist < min_dist:
                violations.append((i, j))

    return violations


def count_spacing_violations(mirrors: List[Mirror],
                             min_clearance: float = 5.0) -> int:
    """返回间距违规数量（便捷函数）"""
    return len(spacing_check(mirrors, min_clearance))


def check_exclusion_zone(mirrors: List[Mirror],
                         tower_xy: Tuple[float, float] = (0.0, 0.0),
                         exclusion_r: float = 100.0) -> List[int]:
    """
    检查禁区违规。

    Returns:
        禁区内的镜面下标列表
    """
    tx, ty = tower_xy
    violators = []
    for i, m in enumerate(mirrors):
        dist = np.sqrt((m.x - tx)**2 + (m.y - ty)**2)
        if dist < exclusion_r:
            violators.append(i)
    return violators


def check_field_boundary(mirrors: List[Mirror],
                         center_xy: Tuple[float, float] = (0.0, 0.0),
                         field_radius: float = 350.0) -> List[int]:
    """
    检查场地边界违规。

    Returns:
        超出边界的镜面下标列表
    """
    cx, cy = center_xy
    violators = []
    for i, m in enumerate(mirrors):
        dist = np.sqrt((m.x - cx)**2 + (m.y - cy)**2)
        if dist > field_radius:
            violators.append(i)
    return violators


def validate_mirror_params(mirrors: List[Mirror],
                           W_range: Tuple[float, float] = (2.0, 8.0),
                           H_range: Tuple[float, float] = (2.0, 8.0),
                           h_range: Tuple[float, float] = (2.0, 6.0)) -> List[str]:
    """
    参数合法性检查。

    Returns:
        错误消息列表（空列表 = 全部合法）
    """
    errors = []
    for i, m in enumerate(mirrors):
        if not (W_range[0] <= m.width <= W_range[1]):
            errors.append(f"镜面 {i}: W={m.width} 超出范围 {W_range}")
        if not (H_range[0] <= m.height <= H_range[1]):
            errors.append(f"镜面 {i}: H={m.height} 超出范围 {H_range}")
        if not (h_range[0] <= m.install_height <= h_range[1]):
            errors.append(f"镜面 {i}: h={m.install_height} 超出范围 {h_range}")
    return errors


def total_mirrors_count(mirrors: List[Mirror]) -> int:
    """镜面总数"""
    return len(mirrors)


def total_reflective_area(mirrors: List[Mirror]) -> float:
    """总反射面积（m²）"""
    return sum(m.area for m in mirrors)