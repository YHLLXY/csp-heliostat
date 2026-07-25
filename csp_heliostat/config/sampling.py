"""
时间采样表生成。

生成 12 个月 × 5 个时刻 = 60 个采样时刻，以及对应的权重。
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# 每月天数（非闰年）
DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_CUMULATIVE = [0] * 13
for m in range(1, 13):
    DAYS_CUMULATIVE[m] = DAYS_CUMULATIVE[m - 1] + DAYS_IN_MONTH[m]

# 春分（3月21日）的年积日
SPRING_EQUINOX_DOY = 80  # 3月21日 = 31(Jan) + 28(Feb) + 21(Mar) = 80


def day_of_year(month: int, day: int) -> int:
    """
    Gregorian 日期 → 年积日（1-365）。

    Args:
        month: 月份 (1-12)
        day: 日 (1-31)

    Returns:
        年积日 (1 = 1月1日, 365 = 12月31日)
    """
    return DAYS_CUMULATIVE[month - 1] + day


def day_from_spring_equinox(doy: int) -> int:
    """
    年积日 → 从春分起算的积日 D。

    D = doy - 80，即春分日（3月21日）对应 D = 0。

    Args:
        doy: 年积日 (1-365)

    Returns:
        积日 D（可为负，表示春分之前）
    """
    return doy - SPRING_EQUINOX_DOY


@dataclass
class SamplePoint:
    """单个采样时刻"""
    month: int
    day: int
    hour: float
    doy: int          # 年积日（1-365）
    D_spring: int     # 从春分起算的积日 D（春分=0）
    weight: float     # 年权重 ∝ sin(α_s)


def build_sampling_schedule(
    times: List[float] = None,
    day_of_month: int = 21
) -> List[SamplePoint]:
    """
    生成 12×5=60 个采样时刻。

    Args:
        times: 每天采样的当地时间（小时），默认 [9, 10.5, 12, 13.5, 15]
        day_of_month: 每月采样的日期，默认 21

    Returns:
        60 个 SamplePoint 列表
    """
    if times is None:
        from csp_heliostat.config.constants import SAMPLE_TIMES_HOUR
        times = list(SAMPLE_TIMES_HOUR)

    samples = []
    for month in range(1, 13):
        doy = day_of_year(month, day_of_month)
        D = day_from_spring_equinox(doy)
        for hour in times:
            samples.append(SamplePoint(
                month=month,
                day=day_of_month,
                hour=hour,
                doy=doy,
                D_spring=D,
                weight=1.0,  # 稍后由 solar_weights() 计算
            ))
    return samples


def build_noon_schedule(day_of_month: int = 21) -> List[SamplePoint]:
    """
    生成 12 x 1 = 12 个正午采样时刻（快速优化模式）。

    仅采样每月 21 日正午 12:00，速度约为完整 60 时刻的 5 倍。
    用于迭代优化中快速评估目标函数；最终结果仍须用 60 时刻验证。

    Args:
        day_of_month: 每月采样的日期，默认 21

    Returns:
        12 个 SamplePoint 列表
    """
    return build_sampling_schedule(times=[12.0], day_of_month=day_of_month)


def solar_weights(sin_altitude: np.ndarray) -> np.ndarray:
    """
    计算每时刻的年权重。

    使用 weight = max(0, sin α_s) 加权，物理意义为
    "日照辐射的相对贡献"，避免日出日落极端低角时刻主导年平均。

    归一化使得 Σ weight = 1（用于加权平均）。

    Args:
        sin_altitude: 每个时刻的 sin(α_s)，shape (N,)

    Returns:
        归一化权重，shape (N,)
    """
    w = np.maximum(0, sin_altitude)
    total = w.sum()
    if total > 0:
        return w / total
    else:
        return np.ones_like(w) / len(w)