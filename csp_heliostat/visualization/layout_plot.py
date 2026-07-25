"""
镜场布局可视化。

平面图 + 效率热力图 + 年度效率曲线。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
from typing import List, Optional

from csp_heliostat.field.mirror import Mirror


# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_field(mirrors: List[Mirror],
               tower_xy: tuple = (0.0, 0.0),
               exclusion_r: float = 100.0,
               field_r: float = 350.0,
               eta_values: Optional[np.ndarray] = None,
               title: str = "定日镜场布局",
               save_path: Optional[str] = None) -> None:
    """
    镜场平面布局图。

    - 蓝色方块 = 镜面（色深按 eta 编码）
    - 红色圆 = 禁区
    - 黑色虚线 = 镜场边界
    - 红色三角形 = 塔

    Args:
        mirrors: 镜面列表
        tower_xy: 塔位置 (x, y)
        exclusion_r: 禁区半径
        field_r: 镜场半径
        eta_values: 每镜效率值（可选，用于色编码）
        title: 图标题
        save_path: 保存路径。为 None 则显示。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 绘制镜场边界
    field_circle = Circle((0, 0), field_r, fill=False,
                           linestyle='--', color='gray', linewidth=1,
                           label=f'Field boundary (r={field_r}m)')
    ax.add_patch(field_circle)

    # 绘制禁区
    excl_circle = Circle((0, 0), exclusion_r, fill=True,
                          color='lightcoral', alpha=0.3,
                          label=f'Exclusion zone (r={exclusion_r}m)')
    ax.add_patch(excl_circle)

    # 绘制塔
    ax.plot(tower_xy[0], tower_xy[1], 'r^', markersize=12,
            label=f'Tower ({tower_xy[0]}, {tower_xy[1]})')

    # 绘制镜面
    if eta_values is not None:
        colors = eta_values
        vmin, vmax = 0.0, 1.0
    else:
        colors = 'steelblue'
        vmin, vmax = None, None

    xs = [m.x for m in mirrors]
    ys = [m.y for m in mirrors]
    sizes = [m.width * 0.8 for m in mirrors]  # 缩小以可见间距

    scatter = ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.7,
                          cmap='RdYlGn', vmin=vmin, vmax=vmax,
                          edgecolors='none')

    if eta_values is not None:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('eta_total', fontsize=11)

    # 标注
    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 范围
    margin = 20
    ax.set_xlim(-field_r - margin, field_r + margin)
    ax.set_ylim(-field_r - margin, field_r + margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_efficiency_heatmap(mirrors: List[Mirror],
                             eta_values: np.ndarray,
                             title: str = "效率热力图",
                             save_path: Optional[str] = None) -> None:
    """
    效率热力图（散点色编码）。

    Args:
        mirrors: 镜面列表
        eta_values: 每镜效率值
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    xs = [m.x for m in mirrors]
    ys = [m.y for m in mirrors]

    scatter = ax.scatter(xs, ys, c=eta_values, s=2, alpha=0.8,
                          cmap='RdYlGn', vmin=eta_values.min(), vmax=eta_values.max(),
                          edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Optical Efficiency', fontsize=11)

    # 塔
    ax.plot(0, 0, 'r^', markersize=10, label='Tower')

    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_annual_efficiency_curve(per_sample: List[dict],
                                  save_path: Optional[str] = None) -> None:
    """
    年度效率/功率曲线（60 时刻）。

    Args:
        per_sample: annual_simulation 的 per_sample 列表
        save_path: 保存路径
    """
    months = [s['month'] for s in per_sample]
    hours = [s['hour'] for s in per_sample]
    eta = [s['eta_field_avg'] for s in per_sample]
    power = [s['power_mw'] for s in per_sample]

    # 构造横轴：月份 + 时刻偏移
    x = []
    labels = []
    for i, (m, h) in enumerate(zip(months, hours)):
        x.append(m + (h - 12) / 12.0)
        labels.append(f"{m}/{h:.0f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 效率
    ax1.plot(x, eta, 'o-', color='steelblue', markersize=4, linewidth=1)
    ax1.set_ylabel('eta_field', fontsize=12)
    ax1.set_title('Annual Efficiency & Power Curve (60 moments)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 功率
    ax2.plot(x, power, 's-', color='darkorange', markersize=4, linewidth=1)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('P_field (MW)', fontsize=12)
    ax2.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='60 MW target')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # x 轴标注
    month_ticks = list(range(1, 13))
    ax2.set_xticks(month_ticks)
    ax2.set_xticklabels([f'{m}月' for m in month_ticks])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ================================================================
# 改进项专用图表函数
# ================================================================

def plot_efficiency_breakdown(components: dict,
                               save_path_pie: str = None,
                               save_path_bar: str = None) -> None:
    """
    分项效率饼图 + 条形图。

    Args:
        components: compute_efficiency_breakdown 的返回值
        save_path_pie: 饼图保存路径
        save_path_bar: 条形图保存路径
    """
    labels = ['eta_cos\n(Cosine)', 'eta_sb\n(Shadow/Block)',
              'eta_trunc\n(Truncation)', 'eta_at\n(Atmospheric)', 'eta_ref\n(Reflectivity)']
    values = [components['eta_cos'], components['eta_sb'],
              components['eta_trunc'], components['eta_at'], components['eta_ref']]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

    # --- 饼图 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.2f%%',
        colors=colors, startangle=90,
        explode=(0.02, 0.02, 0.02, 0.02, 0.02),
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f'Efficiency Breakdown (annual avg)\n'
                 f'Total eta = {components["eta_total"]:.4f}',
                 fontsize=13)
    plt.tight_layout()
    if save_path_pie:
        plt.savefig(save_path_pie, dpi=150, bbox_inches='tight')
        plt.close()

    # --- 条形图 ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bar_labels = ['cos', 'sb', 'trunc', 'at', 'ref', 'total']
    bar_values = values + [components['eta_total']]
    bar_colors = colors + ['#34495e']
    bars = ax.bar(bar_labels, bar_values, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=10)
    ax.set_ylabel('Efficiency', fontsize=12)
    ax.set_title('Efficiency Components (annual weighted avg)', fontsize=13)
    ax.set_ylim(0, max(bar_values) * 1.12)
    ax.axhline(y=components['eta_total'], color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path_bar:
        plt.savefig(save_path_bar, dpi=150, bbox_inches='tight')
        plt.close()


def plot_weighting_comparison(weighting_result: dict,
                               save_path: str = None) -> None:
    """
    四种加权方法对比柱状图。

    Args:
        weighting_result: compare_weighting_methods 的返回值
        save_path: 保存路径
    """
    labels = weighting_result['labels']
    P_vals = weighting_result['P_avg_MW']

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(labels, P_vals, color=colors_bar, edgecolor='white')

    for bar, val in zip(bars, P_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f} MW', ha='center', fontsize=11, fontweight='bold')

    ax.axhline(y=60.0, color='red', linestyle='--', alpha=0.6, label='60 MW target')
    ax.set_ylabel('Annual Avg Power (MW)', fontsize=12)
    ax.set_title(f'Weighting Method Comparison\n'
                 f'(Max deviation: {weighting_result["max_deviation_pct"]:.2f}%)',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_sensitivity_curves(sensitivity_results: dict,
                             save_dir: str = None) -> None:
    """
    敏感性分析曲线（多面板）。

    Args:
        sensitivity_results: run_sensitivity 的返回值
        save_dir: 保存目录
    """
    import os

    var_labels = {
        'x_t': 'Tower East Offset (m)',
        'y_t': 'Tower North Offset (m)',
        'W': 'Mirror Width (m)',
        'h': 'Install Height (m)',
        'n_rings': 'Number of Rings',
    }

    # x_t 和 y_t 合并为一张双面板图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, var_name in zip(axes, ['x_t', 'y_t']):
        data = sensitivity_results[var_name]
        values = data['values']
        P = [p['P_avg_MW'] for p in data['points']]
        P_pa = [p['P_per_area_W_m2'] for p in data['points']]

        color1 = '#3498db'
        ax2 = ax.twinx()
        line1, = ax.plot(values, P, 'o-', color=color1, markersize=6, linewidth=2)
        line2, = ax2.plot(values, P_pa, 's--', color='#e74c3c', markersize=6, linewidth=2)

        ax.axhline(y=60.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(var_labels[var_name], fontsize=11)
        ax.set_ylabel('P_avg (MW)', fontsize=11, color=color1)
        ax2.set_ylabel('P/A (W/m^2)', fontsize=11, color='#e74c3c')
        ax.tick_params(axis='y', labelcolor=color1)
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sensitivity: {var_name}')

    fig.suptitle('Tower Position Sensitivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sensitivity_xy.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # W, h 各自独立图
    for var_name in ['W', 'h']:
        data = sensitivity_results[var_name]
        values = data['values']
        P = [p['P_avg_MW'] for p in data['points']]
        P_pa = [p['P_per_area_W_m2'] for p in data['points']]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = ax.twinx()
        line1, = ax.plot(values, P, 'o-', color='#3498db', markersize=8, linewidth=2)
        line2, = ax2.plot(values, P_pa, 's--', color='#e74c3c', markersize=8, linewidth=2)
        ax.axhline(y=60.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(var_labels[var_name], fontsize=12)
        ax.set_ylabel('P_avg (MW)', fontsize=12, color='#3498db')
        ax2.set_ylabel('P/A (W/m^2)', fontsize=12, color='#e74c3c')
        ax.tick_params(axis='y', labelcolor='#3498db')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sensitivity: {var_labels[var_name]}', fontsize=13)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'sensitivity_{var_name}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

    # n_rings 独立图
    data = sensitivity_results['n_rings']
    values = data['values']
    P = [p['P_avg_MW'] for p in data['points']]
    P_pa = [p['P_per_area_W_m2'] for p in data['points']]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax2 = ax.twinx()
    line1, = ax.plot(values, P, 'D-', color='#3498db', markersize=8, linewidth=2)
    line2, = ax2.plot(values, P_pa, 's--', color='#e74c3c', markersize=8, linewidth=2)
    ax.axhline(y=60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(var_labels['n_rings'], fontsize=12)
    ax.set_ylabel('P_avg (MW)', fontsize=12, color='#3498db')
    ax2.set_ylabel('P/A (W/m^2)', fontsize=12, color='#e74c3c')
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Sensitivity: {var_labels["n_rings"]}', fontsize=13)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sensitivity_n_rings.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()


def plot_field_with_normals(mirrors: List[Mirror],
                             sun_alt_deg: float,
                             sun_az_deg: float,
                             tower_xy: tuple = (0.0, 0.0),
                             tower_height: float = 80.0,
                             receiver_height: float = 8.0,
                             receiver_radius: float = 3.5,
                             title: str = "Mirror Normals",
                             save_path: str = None,
                             max_arrows: int = 500) -> None:
    """
    镜场法向量俯视图（箭头叠加）。

    每面镜画一个箭头，方向 = 镜面法向量的水平投影。
    箭头颜色 = 余弦效率，长度 = eta_cos 缩放。

    Args:
        mirrors: 镜面列表
        sun_alt_deg: 太阳高度角 (°)
        sun_az_deg: 太阳方位角 (°)
        tower_xy: 塔位置
        tower_height: 塔高
        receiver_height: 接收器高度
        receiver_radius: 接收器半径
        title: 图标题
        save_path: 保存路径
        max_arrows: 最多画多少箭头（过多会糊成一团）
    """
    from csp_heliostat.core.solar_position import SunState
    from csp_heliostat.core.geometry import sun_unit_vector, mirror_normal
    from csp_heliostat.efficiency.cosine import cosine_for_field
    import numpy as np

    # 构造 SunState
    alt = np.array([sun_alt_deg])
    az = np.array([sun_az_deg])
    sin_alt = np.sin(np.deg2rad(alt))
    cos_alt = np.cos(np.deg2rad(alt))
    is_day = np.array([sin_alt[0] > 0])

    sun = SunState(
        altitude_deg=alt,
        azimuth_deg=az,
        sin_altitude=sin_alt,
        cos_altitude=cos_alt,
        is_daytime=is_day,
    )

    # 计算余弦效率（用于着色）
    eta_cos = cosine_for_field(mirrors, sun, tower_xy, tower_height, receiver_height)

    # 降采样（避免箭头过于密集）
    if len(mirrors) > max_arrows:
        step = max(1, len(mirrors) // max_arrows)
        idx = list(range(0, len(mirrors), step))
    else:
        idx = list(range(len(mirrors)))

    # 太阳单位向量（squeeze 到 (3,) 便于逐镜计算）
    s_hat = sun_unit_vector(sun_alt_deg, sun_az_deg)
    if s_hat.ndim == 2 and s_hat.shape[0] == 1:
        s_hat = s_hat[0]  # (1,3) -> (3,)

    # 接收器中心坐标
    rx, ry = tower_xy
    rz = tower_height + receiver_height / 2.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 背景：全部镜面位置（灰点）
    all_xs = [m.x for m in mirrors]
    all_ys = [m.y for m in mirrors]
    ax.scatter(all_xs, all_ys, s=1, color='lightgray', alpha=0.5, zorder=0)

    # 法向量箭头
    for i in idx:
        m = mirrors[i]

        # 镜面中心坐标
        mz = m.install_height + m.height / 2.0

        # 接收器方向向量 (从镜面指向接收器)
        dx = rx - m.x
        dy = ry - m.y
        dz = rz - mz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 1e-10:
            continue
        r_hat = np.array([dx / dist, dy / dist, dz / dist])

        # 镜面法向量
        n_hat = mirror_normal(s_hat, r_hat)

        # n_hat 的水平投影
        nx, ny = n_hat[0], n_hat[1]
        length = np.sqrt(nx**2 + ny**2)
        if length < 1e-6:
            continue
        nx /= length
        ny /= length

        arrow_len = eta_cos[i] * 8.0  # 缩放箭头长度
        color = plt.cm.RdYlGn(eta_cos[i])

        ax.arrow(m.x, m.y, nx * arrow_len, ny * arrow_len,
                 head_width=1.2, head_length=1.5, fc=color, ec=color,
                 alpha=0.7, linewidth=0.3)

    # 塔
    ax.plot(tower_xy[0], tower_xy[1], 'r^', markersize=12, label='Tower')

    # 禁区
    excl = Circle((0, 0), 100, fill=False, linestyle='--', color='red', alpha=0.5)
    ax.add_patch(excl)

    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-360, 360)
    ax.set_ylim(-360, 360)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_convergence_curve(history: List[dict],
                            uniform_baseline: float = 525.85,
                            save_path: str = None) -> None:
    """
    问题三 DE 收敛曲线。

    Args:
        history: [{'generation': int, 'best_fitness': float}, ...]
        uniform_baseline: 均一配置的 P/A (W/m^2) 水平线
        save_path: 保存路径
    """
    gens = [h['generation'] for h in history]
    # fitness 值是 MW/m² 单位（objective_zoned 返回 -P_per_area，P_per_area = MW/m²）
    # 乘以 1e6 转换为 W/m²，与 uniform_baseline 的单位一致
    fitness = [h['best_fitness'] * 1e6 for h in history]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(gens, fitness, 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter(gens, fitness, c='steelblue', s=20, alpha=0.6)

    # 均一基线（注意：DE 最小化 -P/A，所以这里用负值）
    ax.axhline(y=-uniform_baseline, color='red', linestyle='--', alpha=0.7,
               label=f'Uniform baseline (-{uniform_baseline:.1f} W/m2)')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Objective (-P/A, W/m2)', fontsize=12)
    ax.set_title('Problem 3 DE Convergence', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_monthly_power(monthly_stats: List[dict],
                        save_path: str = None) -> None:
    """
    月度均值柱状图。

    Args:
        monthly_stats: compute_monthly_stats 的 monthly 列表
        save_path: 保存路径
    """
    months = [s['month'] for s in monthly_stats]
    P_mean = [s['P_mean_MW'] for s in monthly_stats]
    P_std = [s['P_std_MW'] for s in monthly_stats]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    colors_bar = ['#e74c3c' if p < 60 else '#2ecc71' for p in P_mean]
    bars = ax.bar(months, P_mean, yerr=P_std, color=colors_bar, edgecolor='white',
                  capsize=4, error_kw={'linewidth': 1})

    ax.axhline(y=60.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
               label='60 MW target')

    for bar, val in zip(bars, P_mean):
        idx = int(bar.get_x() + bar.get_width() / 2) - 1
        if 0 <= idx < len(P_std):
            y_pos = bar.get_height() + P_std[idx] + 0.5
        else:
            y_pos = bar.get_height() + 0.5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.1f}', ha='center', fontsize=8)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Monthly Avg Power (MW)', fontsize=12)
    ax.set_title('Monthly Average Power (with standard deviation)', fontsize=13)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_diurnal_curves(summer_result: dict,
                         winter_result: dict,
                         save_path_single: str = None,
                         save_path_comparison: str = None) -> None:
    """
    夏至/冬至全天功率曲线。

    Args:
        summer_result: compute_diurnal_curve 的夏至结果
        winter_result: compute_diurnal_curve 的冬至结果
        save_path_single: 单日图保存路径（可用 {label} 占位）
        save_path_comparison: 叠加对比图保存路径
    """
    # --- 夏至图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    hours_s = summer_result['hours']
    ax1.plot(hours_s, summer_result['P_MW'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title('Summer Solstice Diurnal Curve', fontsize=13)
    ax1.axhline(y=60.0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours_s, summer_result['eta'], 's-', color='#3498db', linewidth=2, markersize=8)
    ax2.set_xlabel('Local Time (hour)', fontsize=12)
    ax2.set_ylabel('eta_field', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path_single:
        sp = save_path_single.replace('{label}', 'summer')
        plt.savefig(sp, dpi=150, bbox_inches='tight')
        plt.close()

    # --- 冬至图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    hours_w = winter_result['hours']
    ax1.plot(hours_w, winter_result['P_MW'], 'o-', color='#3498db', linewidth=2, markersize=8)
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title('Winter Solstice Diurnal Curve', fontsize=13)
    ax1.axhline(y=60.0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours_w, winter_result['eta'], 's-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xlabel('Local Time (hour)', fontsize=12)
    ax2.set_ylabel('eta_field', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path_single:
        sp = save_path_single.replace('{label}', 'winter')
        plt.savefig(sp, dpi=150, bbox_inches='tight')
        plt.close()

    # --- 叠加对比图 ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(summer_result['hours'], summer_result['P_MW'], 'o-',
            color='#e74c3c', linewidth=2, markersize=6, label='Summer Solstice')
    ax.plot(winter_result['hours'], winter_result['P_MW'], 's-',
            color='#3498db', linewidth=2, markersize=6, label='Winter Solstice')
    ax.axhline(y=60.0, color='gray', linestyle='--', alpha=0.5, label='60 MW target')
    ax.set_xlabel('Local Time (hour)', fontsize=12)
    ax.set_ylabel('Power (MW)', fontsize=12)
    ax.set_title('Summer vs Winter Solstice — Diurnal Power Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path_comparison:
        plt.savefig(save_path_comparison, dpi=150, bbox_inches='tight')
        plt.close()