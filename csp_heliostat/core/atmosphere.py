"""
大气透过率 η_at 模型。

采用 Sandia/DLR 经验式（工程常用，无需题目提供具体大气公式）：

  η_at = 0.99321 - 0.0001176·d + 1.97e-8·d²

其中 d 为镜面中心到接收器的直线距离（米）。

适用范围：d ∈ [0, 2000] 米（基本覆盖 350m 镜场范围）。
超出范围时裁剪到合理边界。
"""

import numpy as np


def atmospheric_transmittance(distance_m: np.ndarray) -> np.ndarray:
    """
    计算大气透过率（Sandia/DLR 经验式）。

    Args:
        distance_m: 镜面到接收器的斜线距离（m），shape (N,)

    Returns:
        η_at，无量纲，shape (N,)
    """
    d = np.atleast_1d(np.asarray(distance_m, dtype=float))

    # Sandia/DLR 经验多项式
    eta = 0.99321 - 0.0001176 * d + 1.97e-8 * d**2

    # 物理合理性约束
    eta = np.clip(eta, 0.75, 0.995)

    return eta


def slant_distance(mirror_xy: np.ndarray,
                   mirror_center_z: np.ndarray,
                   receiver_xy: tuple,
                   receiver_center_z: float) -> np.ndarray:
    """
    计算镜面到接收器的斜线距离。

    Args:
        mirror_xy: 镜面 (x, y) 坐标，shape (N, 2)
        mirror_center_z: 镜面中心高度（m），shape (N,)
        receiver_xy: 接收器 (x, y) 坐标
        receiver_center_z: 接收器中心高度（m）

    Returns:
        斜线距离（m），shape (N,)
    """
    dx = receiver_xy[0] - mirror_xy[..., 0]
    dy = receiver_xy[1] - mirror_xy[..., 1]
    dz = receiver_center_z - mirror_center_z

    return np.sqrt(dx**2 + dy**2 + dz**2)