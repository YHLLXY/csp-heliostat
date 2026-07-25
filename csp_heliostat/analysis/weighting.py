"""
改进 #3：四种加权方法对照。

对同一镜场配置，用四种加权方式计算年平均功率 P_avg：
  1. 等权平均：w = 1/N
  2. sin α_s 加权（当前默认）：w ∝ max(0, sin α_s)
  3. DNI 加权：w ∝ DNI(t)
  4. 余弦日照加权：w ∝ max(0, cos ω) — 日照时长代理变量

输出对比表和柱状图。
"""

import numpy as np
from typing import Dict, List


def compare_weighting_methods(annual_result: Dict) -> Dict:
    """
    用四种加权方法对同一批 per_sample 数据重新加权。

    注意：这里复用已仿真好的 per_sample，仅改变权重聚合方式。
    不重新跑仿真。

    Args:
        annual_result: annual_simulation 的返回值

    Returns:
        dict: {
            'methods': ['equal', 'sin_alpha', 'dni', 'cos_omega'],
            'labels': ['等权平均', 'sin α_s 加权', 'DNI 加权', '日照加权'],
            'P_avg_MW': [P1, P2, P3, P4],
            'eta_avg': [η1, η2, η3, η4],
            'max_deviation_pct': float,  # 四种方法的最大相对偏差
        }
    """
    per_sample = annual_result['per_sample']
    n = len(per_sample)

    P_arr = np.array([s['power_mw'] for s in per_sample])
    eta_arr = np.array([s['eta_field_avg'] for s in per_sample])
    dni_arr = np.array([s['dni'] for s in per_sample])

    # 从 schedule 中提取时角信息（hour → ω = 15°·(t-12)）
    hours = np.array([s['hour'] for s in per_sample])
    omega_deg = 15.0 * (hours - 12.0)
    omega_rad = np.deg2rad(omega_deg)
    cos_omega = np.cos(omega_rad)

    # 方法 1：等权
    w1 = np.ones(n) / n
    P1 = float(np.sum(P_arr * w1))
    eta1 = float(np.sum(eta_arr * w1))

    # 方法 2：sin α_s 加权（从 altitude_deg 反推）
    alt_deg = np.array([s['altitude_deg'] for s in per_sample])
    sin_alpha = np.maximum(0, np.sin(np.deg2rad(alt_deg)))
    w2 = sin_alpha / sin_alpha.sum() if sin_alpha.sum() > 0 else np.ones(n) / n
    P2 = float(np.sum(P_arr * w2))
    eta2 = float(np.sum(eta_arr * w2))

    # 方法 3：DNI 加权
    dni_pos = np.maximum(0, dni_arr)
    w3 = dni_pos / dni_pos.sum() if dni_pos.sum() > 0 else np.ones(n) / n
    P3 = float(np.sum(P_arr * w3))
    eta3 = float(np.sum(eta_arr * w3))

    # 方法 4：cos ω 加权（余弦日照时长代理）
    cos_pos = np.maximum(0, cos_omega)
    w4 = cos_pos / cos_pos.sum() if cos_pos.sum() > 0 else np.ones(n) / n
    P4 = float(np.sum(P_arr * w4))
    eta4 = float(np.sum(eta_arr * w4))

    # 最大偏差
    P_all = np.array([P1, P2, P3, P4])
    P_mean = P_all.mean()
    max_dev = float(np.max(np.abs(P_all - P_mean)) / P_mean * 100) if P_mean > 0 else 0.0

    return {
        'methods': ['equal', 'sin_alpha', 'dni', 'cos_omega'],
        'labels': ['Equal weight', 'sin_a_s weighted', 'DNI weighted', 'cos_w weighted'],
        'P_avg_MW': [P1, P2, P3, P4],
        'eta_avg': [eta1, eta2, eta3, eta4],
        'max_deviation_pct': max_dev,
    }