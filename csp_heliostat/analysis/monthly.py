"""
改进 #6：月度均值统计。

按月聚合 60 时刻的 P(t)，计算每月均值 ± 标准差。
标注哪些月份 P_month < 60 MW，估算储能缺口。
"""

import numpy as np
from typing import Dict, List


def compute_monthly_stats(annual_result: Dict) -> Dict:
    """
    按月聚合仿真结果，计算每月的平均功率和效率。

    Args:
        annual_result: annual_simulation 的返回值

    Returns:
        dict: {
            'monthly': [
                {'month': 1, 'P_mean_MW': ..., 'P_std_MW': ..., 'eta_mean': ...,
                 'n_samples': 5, 'below_60': True/False},
                ...
            ],
            'months_below_60': [11, 12, 1, 2, ...],
            'min_monthly_P_MW': float,
            'storage_required_MWh': float,  # 估算储能需求
        }
    """
    per_sample = annual_result['per_sample']

    # 按月分组
    monthly_data = {m: [] for m in range(1, 13)}
    for s in per_sample:
        monthly_data[s['month']].append(s)

    monthly_stats = []
    months_below_60 = []

    for month in range(1, 13):
        samples = monthly_data[month]
        if not samples:
            continue

        P_vals = np.array([s['power_mw'] for s in samples])
        eta_vals = np.array([s['eta_field_avg'] for s in samples])

        P_mean = float(np.mean(P_vals))
        P_std = float(np.std(P_vals))
        eta_mean = float(np.mean(eta_vals))
        below = P_mean < 60.0

        if below:
            months_below_60.append(month)

        monthly_stats.append({
            'month': month,
            'P_mean_MW': P_mean,
            'P_std_MW': P_std,
            'eta_mean': eta_mean,
            'n_samples': len(samples),
            'below_60': below,
        })

    # 估算储能需求：对低于 60MW 的月份，计算需要补足的"缺口能量"
    # 假设每个时刻代表 1/5 天，每月 30 天
    storage_required_MWh = 0.0
    for month in months_below_60:
        samples = monthly_data[month]
        for s in samples:
            deficit = max(0, 60.0 - s['power_mw'])
            # 每时刻代表 30天/5 = 6 天的平均，6天×24h = 144h
            hours_per_sample = 24.0 * (30.0 / len(samples))
            storage_required_MWh += deficit * hours_per_sample

    min_monthly = min(s['P_mean_MW'] for s in monthly_stats) if monthly_stats else 0.0

    return {
        'monthly': monthly_stats,
        'months_below_60': months_below_60,
        'min_monthly_P_MW': min_monthly,
        'storage_required_MWh': storage_required_MWh,
    }