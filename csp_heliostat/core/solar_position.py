"""
太阳位置模型 — 计算任意时刻的太阳高度角与方位角。

输入：纬度 φ、积日 D、当地时间 t（小时）
输出：太阳高度角 α_s、太阳方位角 γ_s

方位角约定（关键）：
  0°  = 正南方向（太阳在正南，即正午前后）
  +90° = 正东方向（上午太阳在东南 ≈ +45° 附近）
  -90° = 正西方向（下午太阳在西南 ≈ -45° 附近）

  这是"从南起算、逆时针为正"的约定，等价于：
  南=0, 东=+90, 北=±180, 西=-90

积日 D 两种约定：
  "from_spring_equinox": D = doy - 80（春分=0，推荐）
  "from_jan1":           D = doy（1月1日=1）
"""

import numpy as np
from dataclasses import dataclass
from typing import Union


def _doy_from_D(D: np.ndarray, convention: str = "from_spring_equinox") -> np.ndarray:
    """
    将积日 D 统一转换为年积日（day of year, 1-365）。

    Args:
        D: 积日数组
        convention: "from_spring_equinox" 或 "from_jan1"

    Returns:
        年积日（1 = 1月1日）
    """
    if convention == "from_spring_equinox":
        return D + 80  # 春分=3月21日=第80天
    elif convention == "from_jan1":
        return D
    else:
        raise ValueError(f"Unknown D convention: {convention}")


def declination(D: Union[float, np.ndarray],
                convention: str = "from_spring_equinox") -> np.ndarray:
    """
    计算太阳赤纬角 δ。

    公式：δ = 23.45° · sin(2π(284 + doy) / 365)

    Args:
        D: 积日（含义由 convention 决定）
        convention: D 的约定方式

    Returns:
        赤纬角（度），-23.45° ~ +23.45°
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    doy = _doy_from_D(D, convention)
    delta_deg = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    return delta_deg


def hour_angle(t_hour: Union[float, np.ndarray]) -> np.ndarray:
    """
    计算时角 ω。

    公式：ω = 15° · (t - 12)

    Args:
        t_hour: 当地时间（小时，24h制）

    Returns:
        时角（度），上午为负、下午为正
    """
    t = np.atleast_1d(np.asarray(t_hour, dtype=float))
    omega_deg = 15.0 * (t - 12.0)
    return omega_deg


def solar_altitude(phi_deg: float,
                   delta_deg: Union[float, np.ndarray],
                   omega_deg: Union[float, np.ndarray]) -> np.ndarray:
    """
    计算太阳高度角 α_s。

    公式：sin α_s = sin φ sin δ + cos φ cos δ cos ω

    Args:
        phi_deg: 纬度（度，北纬为正）
        delta_deg: 赤纬角（度）
        omega_deg: 时角（度）

    Returns:
        高度角（度），-90° ~ +90°（正=地平以上）
    """
    phi = np.deg2rad(phi_deg)
    delta = np.deg2rad(np.atleast_1d(np.asarray(delta_deg, dtype=float)))
    omega = np.deg2rad(np.atleast_1d(np.asarray(omega_deg, dtype=float)))

    sin_alpha = (np.sin(phi) * np.sin(delta) +
                 np.cos(phi) * np.cos(delta) * np.cos(omega))
    # 数值稳定性：clamp 到 [-1, 1]
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_alpha))


def solar_azimuth(phi_deg: float,
                  delta_deg: Union[float, np.ndarray],
                  omega_deg: Union[float, np.ndarray]) -> np.ndarray:
    """
    计算太阳方位角 γ_s。

    公式：γ_s = atan2(sin ω, cos ω sin φ - tan δ cos φ)

    约定：南=0, 东=+90°, 西=-90°

    Args:
        phi_deg: 纬度（度，北纬为正）
        delta_deg: 赤纬角（度）
        omega_deg: 时角（度）

    Returns:
        方位角（度），-180° ~ +180°
    """
    phi = np.deg2rad(phi_deg)
    delta = np.deg2rad(np.atleast_1d(np.asarray(delta_deg, dtype=float)))
    omega = np.deg2rad(np.atleast_1d(np.asarray(omega_deg, dtype=float)))

    # 标准天文公式：atan2(sin ω, cos ω sin φ - tan δ cos φ)
    # 此公式在南=0、东为负的约定下输出
    # 为兼容"南=0、东=+90"约定，取负号
    y = np.sin(omega)
    x = np.cos(omega) * np.sin(phi) - np.tan(delta) * np.cos(phi)

    gamma = np.arctan2(y, x)
    # 转换为度
    gamma_deg = np.rad2deg(gamma)

    # 统一为南=0, 东=+90° 的约定
    # 对公式输出取负：使上午（东边）为正
    gamma_deg = -gamma_deg

    return gamma_deg


@dataclass
class SunState:
    """单时刻太阳状态"""
    altitude_deg: np.ndarray      # 太阳高度角（°）
    azimuth_deg: np.ndarray       # 太阳方位角（°），南=0, 东=+90
    sin_altitude: np.ndarray      # sin(α_s)，预计算避免重复
    cos_altitude: np.ndarray      # cos(α_s)
    is_daytime: np.ndarray        # α_s > 0（布尔数组）


def sun_state_batch(phi_deg: float,
                    D: Union[float, np.ndarray],
                    t_hour: Union[float, np.ndarray],
                    convention: str = "from_spring_equinox") -> SunState:
    """
    批量计算太阳状态。

    凡 α_s ≤ 0 标记为"夜晚"（is_daytime=False），下游模块需跳过。

    Args:
        phi_deg: 纬度（°N）
        D: 积日（含义由 convention 决定）
        t_hour: 当地时间（小时）
        convention: D 的约定

    Returns:
        SunState 包含高度角、方位角及预计算值
    """
    delta = declination(D, convention)
    omega = hour_angle(t_hour)
    alpha = solar_altitude(phi_deg, delta, omega)
    gamma = solar_azimuth(phi_deg, delta, omega)

    alpha_rad = np.deg2rad(alpha)
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)

    is_daytime = alpha > 0.0  # 地平以上

    return SunState(
        altitude_deg=alpha,
        azimuth_deg=gamma,
        sin_altitude=sin_alpha,
        cos_altitude=cos_alpha,
        is_daytime=is_daytime,
    )


# ============================================================
# 便捷函数：对 Python 标量的单次调用
# ============================================================

def sun_position_single(phi_deg: float, D: float, t_hour: float,
                        convention: str = "from_spring_equinox"):
    """
    单个时刻的太阳位置（非向量化版本）。

    Returns:
        (alpha_s_deg, gamma_s_deg)
    """
    state = sun_state_batch(phi_deg, D, t_hour, convention)
    return float(state.altitude_deg[0]), float(state.azimuth_deg[0])