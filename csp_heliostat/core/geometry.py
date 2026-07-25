"""
反射几何 — 太阳/镜面/接收器的三维向量运算。

========================== 坐标系约定（极其重要） ==========================

  全局坐标系：[东, 北, 天] 右手系
    x 轴 → 东（正东为正）
    y 轴 → 北（正北为正）
    z 轴 → 天（垂直向上为正）

  方位角 γ_s（gamma_solar）：
    0°  = 正南方向
    +90° = 正东方向
    -90° = 正西方向

  太阳单位向量 ŝ（从镜面指向太阳）：
    约定 ─ 太阳方位的标准映射：
      ŝ = [cos α_s · sin(-γ_s),  cos α_s · cos(-γ_s),  sin α_s]
        = [-cos α_s · sin γ_s,   cos α_s · cos γ_s,    sin α_s]

    验证（北纬站点，春分正午，γ_s=0°）：
      ŝ = [0, cos α_s, sin α_s]  → 指向正南+天顶 ✓
    验证（日出后，上午9点，太阳在东南，γ_s≈+45°）：
      ŝ = [-cos α_s · sin45°, cos α_s · cos45°, sin α_s]
        = [-cos α_s/√2, cos α_s/√2, sin α_s]
      → x 分量为负（太阳在西边？）→ 不对...

      实际上 γ_s=+45° 表示太阳在东南方（东=+90, 南=0, 东南在中间=+45）
      从镜面看：东偏南 → 应指向 +x（东）、-y（南）
      所以：ŝ = [cos α_s · sin γ_s,  -cos α_s · cos γ_s,  sin α_s]

      重新验证（春分9am，太阳在东南，γ_s≈+45°）：
        ŝ = [cos α_s/√2, -cos α_s/√2, sin α_s]
        x 分量为正 → 东 ✓
        y 分量为负 → 南 ✓

      验证（春分正午，γ_s=0°）：
        ŝ = [0, -cos α_s, sin α_s]
        y 分量为负 → 指向南 ✓

  接收器单位向量 r̂（从镜面中心指向接收器中心）：
    r̂ = normalize([x_R - x_M,  y_R - y_M,  z_R - z_M])

  镜面法向 n̂（反射定律）：
    n̂ = (ŝ + r̂) / |ŝ + r̂|
    仅保留 z 分量 ≥ 0 的解（朝上的法向）

===========================================================================
"""

import numpy as np
from typing import Tuple


def sun_unit_vector(altitude_deg: np.ndarray,
                    azimuth_deg: np.ndarray) -> np.ndarray:
    """
    从太阳角计算 3D 单位向量（从镜面指向太阳）。

    公式：
      ŝ_x =  cos α_s · sin γ_s    （东分量，γ_s>0 → 东 → x>0）
      ŝ_y = -cos α_s · cos γ_s    （南分量，γ_s=0 → 正南 → y<0）
      ŝ_z =  sin α_s              （天顶分量）

    推导：
      γ_s=0 表示正南 → ŝ = [0, -cos α, sin α] ✓
      γ_s=+90 表示正东 → ŝ = [cos α, 0, sin α] ✓

    Args:
        altitude_deg: 太阳高度角（°），shape (N,)
        azimuth_deg: 太阳方位角（°），南=0, 东=+90, shape (N,)

    Returns:
        太阳单位向量，shape (N, 3)
    """
    alpha = np.deg2rad(np.atleast_1d(np.asarray(altitude_deg, dtype=float)))
    gamma = np.deg2rad(np.atleast_1d(np.asarray(azimuth_deg, dtype=float)))

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    sin_g = np.sin(gamma)
    cos_g = np.cos(gamma)

    # 构建 (N, 3) 数组
    s = np.stack([
        cos_a * sin_g,     # x: 东分量
        -cos_a * cos_g,    # y: 南分量（指向南为负）
        sin_a,             # z: 天顶分量
    ], axis=-1)

    # 已经是单位向量（cos²α·sin²γ + cos²α·cos²γ + sin²α = cos²α + sin²α = 1）
    return s


def receiver_vector_from_mirror(mirror_xy: np.ndarray,
                                mirror_height: np.ndarray,
                                receiver_xy: Tuple[float, float],
                                receiver_top_z: float) -> np.ndarray:
    """
    计算从镜面中心指向接收器中心的单位向量 r̂。

    镜面中心 z 坐标 = h_i + H_i/2（安装高度 + 镜面半高）
    接收器中心位于 (x_t, y_t, receiver_top_z - H_R/2)

    Args:
        mirror_xy: 镜面 (x, y) 坐标，shape (N, 2)
        mirror_height: 镜面中心 z 坐标，shape (N,)
        receiver_xy: 接收器顶部中心 (x_t, y_t)
        receiver_top_z: 接收器顶部高度（塔高 + 接收器半高）

    Returns:
        单位向量 r̂，shape (N, 3)
    """
    rx, ry = receiver_xy
    mx = mirror_xy[..., 0]
    my = mirror_xy[..., 1]
    mz = mirror_height

    dx = rx - mx
    dy = ry - my
    dz = receiver_top_z - mz

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    dist = np.maximum(dist, 1e-10)  # 避免除零

    r = np.stack([dx / dist, dy / dist, dz / dist], axis=-1)
    return r


def mirror_normal(s_hat: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    """
    反射定律：n̂ = (ŝ + r̂) / |ŝ + r̂|

    n̂ 指向镜面的反射法向（平分入射光和反射光的夹角）。
    仅 z 分量 ≥ 0 时有效（朝上的法向）。

    Args:
        s_hat: 太阳方向单位向量, shape (N, 3)
        r_hat: 接收器方向单位向量, shape (N, 3)

    Returns:
        镜面法向单位向量, shape (N, 3)
    """
    n = s_hat + r_hat
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    n_hat = n / norm

    # 确保 z 分量 ≥ 0（法向朝上）
    mask = n_hat[..., 2] < 0
    n_hat[mask] = -n_hat[mask]

    return n_hat


def cosine_efficiency(n_hat: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
    """
    余弦效率 η_cos = max(0, n̂ · ŝ)

    即入射角余弦。当 n̂ · ŝ < 0（太阳在镜面背后）时返回 0。

    Args:
        n_hat: 镜面法向单位向量, shape (N, 3)
        s_hat: 太阳方向单位向量, shape (N, 3)

    Returns:
        余弦效率，shape (N,)
    """
    dot = np.sum(n_hat * s_hat, axis=-1)
    return np.maximum(0.0, dot)


def reflection_vector(s_hat: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    """
    反射方向单位向量 r̂_reflected = 2(n̂·ŝ)n̂ - ŝ

    Args:
        s_hat: 入射方向（太阳），shape (N, 3)
        n_hat: 法向，shape (N, 3)

    Returns:
        反射方向单位向量, shape (N, 3)
    """
    dot = np.sum(n_hat * s_hat, axis=-1, keepdims=True)
    r = 2.0 * dot * n_hat - s_hat
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    return r / norm