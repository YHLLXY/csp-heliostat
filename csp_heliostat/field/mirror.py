"""
镜面数据结构 — Mirror dataclass 及批量操作工具。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Mirror:
    """
    单面定日镜的完整描述。

    坐标系：[东(x), 北(y), 天(z)]

    Attributes:
        x: 镜面中心东向坐标（m）
        y: 镜面中心北向坐标（m）
        width: 镜面宽度 W（m），东-西方向
        height: 镜面高度 H（m），竖直方向
        install_height: 安装高度 h_i（m），镜面下边缘离地高度
        center_z: 镜面中心 z 坐标 = h_i + H/2（m）
        area: 镜面面积 = W × H（m²）
    """
    x: float
    y: float
    width: float = 6.0
    height: float = 6.0
    install_height: float = 4.0

    @property
    def center_z(self) -> float:
        return self.install_height + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xy(self) -> np.ndarray:
        """返回 (x, y) 坐标 numpy 数组"""
        return np.array([self.x, self.y])


def mirrors_to_arrays(mirrors: List[Mirror]) -> dict:
    """
    将 Mirror 列表转为批量 numpy 数组，用于向量化计算。

    Returns:
        dict: {
            'xy': (N, 2)  镜面 (x, y),
            'width': (N,) 镜面宽度,
            'height': (N,) 镜面高度,
            'install_h': (N,) 安装高度,
            'center_z': (N,) 中心 z,
            'area': (N,) 面积,
        }
    """
    N = len(mirrors)
    xy = np.zeros((N, 2))
    widths = np.zeros(N)
    heights = np.zeros(N)
    install_h = np.zeros(N)
    center_z = np.zeros(N)
    areas = np.zeros(N)

    for i, m in enumerate(mirrors):
        xy[i, 0] = m.x
        xy[i, 1] = m.y
        widths[i] = m.width
        heights[i] = m.height
        install_h[i] = m.install_height
        center_z[i] = m.center_z
        areas[i] = m.area

    return {
        'xy': xy,
        'width': widths,
        'height': heights,
        'install_h': install_h,
        'center_z': center_z,
        'area': areas,
    }


def arrays_to_mirrors(arrays: dict) -> List[Mirror]:
    """从批量数组恢复 Mirror 列表（用于优化后重建）"""
    xy = arrays['xy']
    widths = arrays['width']
    heights = arrays['height']
    install_h = arrays['install_h']

    mirrors = []
    for i in range(len(xy)):
        mirrors.append(Mirror(
            x=float(xy[i, 0]),
            y=float(xy[i, 1]),
            width=float(widths[i]),
            height=float(heights[i]),
            install_height=float(install_h[i]),
        ))
    return mirrors