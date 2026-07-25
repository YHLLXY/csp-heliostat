"""
改进 #2：分项效率拆解。

从年度仿真结果中提取五项效率的年加权均值，生成饼图和条形图。
"""

import numpy as np
from typing import Dict, List


def compute_efficiency_breakdown(annual_result: Dict) -> Dict:
    """
    从 annual_simulation 结果中计算各分项效率的年加权均值。

    η_total = ρ × η_cos × η_sb × η_trunc × η_at

    Args:
        annual_result: annual_simulation 的返回值，per_sample 中需含
                       eta_cos_avg, eta_sb_avg, eta_trunc_avg, eta_at_avg

    Returns:
        dict: {
            'eta_ref': float,         # 反射率（常数 0.92）
            'eta_cos': float,         # 余弦效率年加权均值
            'eta_sb': float,          # 阴影遮挡效率年加权均值
            'eta_trunc': float,       # 截断效率年加权均值
            'eta_at': float,          # 大气透过率年加权均值
            'eta_total': float,       # 总效率年加权均值
            'product_check': float,   # ρ × η_cos × η_sb × η_trunc × η_at（应与 eta_total 接近）
        }
    """
    per_sample = annual_result['per_sample']
    n = len(per_sample)

    # 加权累加
    w_sum = 0.0
    eta_cos_w = 0.0
    eta_sb_w = 0.0
    eta_trunc_w = 0.0
    eta_at_w = 0.0
    eta_total_w = 0.0

    for s in per_sample:
        w = s.get('weight', 1.0 / n)
        eta_cos_w += s.get('eta_cos_avg', 0.0) * w
        eta_sb_w += s.get('eta_sb_avg', 0.0) * w
        eta_trunc_w += s.get('eta_trunc_avg', 0.0) * w
        eta_at_w += s.get('eta_at_avg', 0.0) * w
        eta_total_w += s.get('eta_field_avg', 0.0) * w
        w_sum += w

    eta_cos = eta_cos_w / w_sum if w_sum > 0 else 0.0
    eta_sb = eta_sb_w / w_sum if w_sum > 0 else 0.0
    eta_trunc = eta_trunc_w / w_sum if w_sum > 0 else 0.0
    eta_at = eta_at_w / w_sum if w_sum > 0 else 0.0
    eta_total = eta_total_w / w_sum if w_sum > 0 else 0.0

    # 反射率是常数
    eta_ref = 0.92
    product_check = eta_ref * eta_cos * eta_sb * eta_trunc * eta_at

    return {
        'eta_ref': eta_ref,
        'eta_cos': eta_cos,
        'eta_sb': eta_sb,
        'eta_trunc': eta_trunc,
        'eta_at': eta_at,
        'eta_total': eta_total,
        'product_check': product_check,
    }


def efficiency_breakdown_table(breakdown: Dict) -> str:
    """生成分项效率的文本表格。"""
    lines = [
        "=" * 50,
        "分项效率拆解（年加权均值）",
        "=" * 50,
        f"  η_ref  (反射率):       {breakdown['eta_ref']:.4f}",
        f"  η_cos  (余弦效率):     {breakdown['eta_cos']:.4f}",
        f"  η_sb   (阴影遮挡):     {breakdown['eta_sb']:.4f}",
        f"  η_trunc(截断效率):     {breakdown['eta_trunc']:.4f}",
        f"  η_at   (大气透过率):   {breakdown['eta_at']:.4f}",
        "-" * 50,
        f"  η_total (实测年加权):  {breakdown['eta_total']:.4f}",
        f"  ρ·Πη_i (乘积校验):     {breakdown['product_check']:.4f}",
        f"  偏差:                   {abs(breakdown['eta_total'] - breakdown['product_check']):.6f}",
        "=" * 50,
    ]
    return "\n".join(lines)