"""
镜场布局生成器。

支持的布局方式：
  1. radial_layout — 径向同心圆（推荐基线）
  2. grid_layout — 矩形栅格（均匀优化初始解）
  3. spiral_layout — 螺旋布局（可选）

核心算法（径向布局密度公式）：
  2π · r_k = n_k · (W + clearance)
  n_k = ceil(2π · r_k / (W + clearance))
"""

import numpy as np
from typing import List, Tuple, Optional
from .mirror import Mirror


def radial_layout(n_rings: int,
                  W: float = 6.0,
                  H: float = None,
                  install_h: float = 4.0,
                  R_inner: float = 100.0,
                  R_outer: float = 350.0,
                  min_clearance: float = 5.0,
                  azimuth_offset: float = 0.0,
                  stagger: bool = True) -> List[Mirror]:
    """
    径向同心圆布局（推荐基线方案）。

    算法：
      1. 环半径 r_k = R_inner + (k + 0.5) · Δr,  k=0..n_rings-1
         Δr = (R_outer - R_inner) / n_rings
      2. 每环镜数 n_k = ceil(2π · r_k / (W + clearance))
      3. 奇数环旋转半格（角度交错），减少相邻环遮挡

    Args:
        n_rings: 环数
        W: 镜面宽度（m），默认 6.0
        H: 镜面高度（m），默认等于 W
        install_h: 安装高度（m），默认 4.0
        R_inner: 内径（禁建区半径），默认 100m
        R_outer: 外径（镜场半径），默认 350m
        min_clearance: 最小间距（m），默认 5.0
        azimuth_offset: 第一环整体旋转角（弧度），默认 0
        stagger: 是否奇偶环交错，默认 True

    Returns:
        Mirror 列表
    """
    if H is None:
        H = W

    dr = (R_outer - R_inner) / n_rings
    mirrors = []

    for k in range(n_rings):
        # 环半径：环在宽度方向的中点
        r_k = R_inner + (k + 0.5) * dr

        # 根据周长计算该环的镜面数量
        n_k = max(1, int(np.floor(2.0 * np.pi * r_k / (W + min_clearance))))

        # 角度步长
        dtheta = 2.0 * np.pi / n_k

        # 奇偶环交错
        offset = azimuth_offset
        if stagger and k % 2 == 1:
            offset += dtheta / 2.0

        for j in range(n_k):
            theta = offset + j * dtheta
            x = r_k * np.cos(theta)
            y = r_k * np.sin(theta)

            mirrors.append(Mirror(
                x=float(x),
                y=float(y),
                width=W,
                height=H,
                install_height=install_h,
            ))

    return mirrors


def grid_layout(n_rows: int,
                n_cols: int,
                dx: float = None,
                dy: float = None,
                x0: float = 0.0,
                y0: float = 0.0,
                W: float = 6.0,
                H: float = None,
                install_h: float = 4.0,
                min_clearance: float = 5.0) -> List[Mirror]:
    """
    矩形栅格布局（用于均匀优化初始解）。

    镜面按行列排列在矩形网格上。

    Args:
        n_rows: 行数（y 方向）
        n_cols: 列数（x 方向）
        dx: 列间距（m），默认 W + clearance
        dy: 行间距（m），默认 H + clearance
        x0: 左下角 x 坐标
        y0: 左下角 y 坐标
        W: 镜面宽度（m）
        H: 镜面高度（m）
        install_h: 安装高度（m）
        min_clearance: 最小间距

    Returns:
        Mirror 列表
    """
    if H is None:
        H = W
    if dx is None:
        dx = W + min_clearance
    if dy is None:
        dy = H + min_clearance

    mirrors = []
    for row in range(n_rows):
        for col in range(n_cols):
            # 交错偏移（偶数行偏移半格）
            offset_x = (dx / 2.0) if row % 2 == 1 else 0.0
            x = x0 + offset_x + col * dx
            y = y0 + row * dy

            mirrors.append(Mirror(
                x=float(x),
                y=float(y),
                width=W,
                height=H,
                install_height=install_h,
            ))

    return mirrors


def spiral_layout(n_mirrors: int,
                  R_inner: float = 100.0,
                  R_outer: float = 350.0,
                  W: float = 6.0,
                  H: float = None,
                  install_h: float = 4.0,
                  min_clearance: float = 5.0) -> List[Mirror]:
    """
    螺旋布局（Fermat 螺旋 / 向日葵型）。

    在极坐标下按黄金角分布，理论上风阻较小、填充均匀。
    常用于 CSP 工程的镜场优化参考。

    Args:
        n_mirrors: 镜面总数
        R_inner: 内径（m）
        R_outer: 外径（m）
        W: 镜面宽度（m）
        H: 镜面高度（m）
        install_h: 安装高度（m）
        min_clearance: 最小间距

    Returns:
        Mirror 列表
    """
    if H is None:
        H = W

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ≈ 137.508°

    mirrors = []
    for i in range(n_mirrors):
        # Fermat 螺旋：r ∝ sqrt(i)
        t = i / (n_mirrors - 1) if n_mirrors > 1 else 0.0
        r = R_inner + (R_outer - R_inner) * np.sqrt(t)
        theta = i * golden_angle

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        mirrors.append(Mirror(
            x=float(x),
            y=float(y),
            width=W,
            height=H,
            install_height=install_h,
        ))

    return mirrors


def exclude_zone(mirrors: List[Mirror],
                 tower_xy: Tuple[float, float] = (0.0, 0.0),
                 exclusion_r: float = 100.0) -> List[Mirror]:
    """
    剔除禁区（厂区建筑物）内的镜面。

    Args:
        mirrors: 镜面列表
        tower_xy: 塔/禁建区中心坐标 (x, y)
        exclusion_r: 禁建区半径（m）

    Returns:
        过滤后的镜面列表
    """
    tx, ty = tower_xy
    result = []
    for m in mirrors:
        dist = np.sqrt((m.x - tx)**2 + (m.y - ty)**2)
        if dist >= exclusion_r:
            result.append(m)
    return result


def field_boundary_filter(mirrors: List[Mirror],
                          center_xy: Tuple[float, float] = (0.0, 0.0),
                          field_radius: float = 350.0,
                          margin: float = 3.0) -> List[Mirror]:
    """
    剔除超出镜场边界的镜面（可保留 margin 边距）。

    Args:
        mirrors: 镜面列表
        center_xy: 场地中心 (x, y)
        field_radius: 场地半径（m）
        margin: 边距（m），镜面允许超出半径的距离

    Returns:
        过滤后的镜面列表
    """
    cx, cy = center_xy
    r_max = field_radius - margin
    result = []
    for m in mirrors:
        dist = np.sqrt((m.x - cx)**2 + (m.y - cy)**2)
        if dist <= r_max:
            result.append(m)
    return result