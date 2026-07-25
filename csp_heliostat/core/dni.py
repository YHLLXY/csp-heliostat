"""
直接法向辐照度（DNI）模型。

DNI = G_0 · [a(H) + b(H) · exp(-c(H) / sin α_s)]

其中 a, b, c 是海拔 H 的三次多项式。
边界处理：当 sin α_s → 0 时线性退化为 0，避免数值爆炸。
"""

import numpy as np
from typing import Tuple


def dni_coefficients(H_km: float) -> Tuple[float, float, float]:
    """
    海拔三次多项式系数。

    来源：题目给出的标准大气辐射模型。

    Args:
        H_km: 海拔高度（km）

    Returns:
        (a, b, c) 三参数
    """
    H = H_km
    a = 0.4237 - 0.00821 * H + 0.000505 * H**2 - 0.000032 * H**3
    b = 0.5055 + 0.00595 * H - 0.000360 * H**2 + 0.000021 * H**3
    c = 0.2711 + 0.01858 * H - 0.001180 * H**2 + 0.000060 * H**3
    return a, b, c


def direct_normal_irradiance(alpha_s_deg: np.ndarray,
                             H_km: float = 3.0,
                             G0: float = 1366.0) -> np.ndarray:
    """
    计算直接法向辐照度。

    DNI = G_0 · [a + b · exp(-c / sin α_s)]

    边界处理：
    - sin α_s 过小时（< 1e-3），线性退化为 0
    - 结果裁剪到 [0, 1.5·G_0] 防止数值发散
    - α_s ≤ 0 时直接返回 0

    Args:
        alpha_s_deg: 太阳高度角（度），shape (N,)
        H_km: 海拔高度（km），默认 3.0
        G0: 太阳常数（W/m²），默认 1366.0

    Returns:
        DNI 值（W/m²），shape (N,)
    """
    a, b, c = dni_coefficients(H_km)

    alpha = np.atleast_1d(np.asarray(alpha_s_deg, dtype=float))
    sin_alpha = np.sin(np.deg2rad(alpha))

    # 极限处理：当 sin α_s < 1e-3 时，exp(-c/sin) 爆炸
    threshold = 1e-3
    mask = sin_alpha > threshold

    out = np.zeros_like(alpha)
    out[mask] = a + b * np.exp(-c / sin_alpha[mask])

    # 线性过渡：sin 在 (0, threshold] 区域
    transition = ~mask & (sin_alpha > 0)
    if np.any(transition):
        # 在 threshold 处的值
        val_at_threshold = a + b * np.exp(-c / threshold)
        # 线性退化至 0
        out[transition] = val_at_threshold * (sin_alpha[transition] / threshold)

    # 夜晚（sin ≤ 0）保持为 0

    # 乘以 G0 并裁剪
    out = out * G0
    out = np.clip(out, 0.0, 1.5 * G0)

    return out