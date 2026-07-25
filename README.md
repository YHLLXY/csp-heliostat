# 光热发电定日镜场优化设计 (CSP Heliostat Field Optimization)

> 数学建模竞赛 — 《定日镜场的优化设计》
> 开发日期：2026-07-24

---

## 快速开始

```bash
# 安装依赖（推荐国内镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行问题一（基准正向模拟，~1 分钟）
python main_problem1.py

# 运行问题二（均匀优化，耗时较长）
python main_problem2.py

# 运行问题三（异构优化，需先跑完问题二）
python main_problem3.py
```

---

## 坐标系约定（极其重要）

```
全局坐标系：[东(x), 北(y), 天(z)] 右手系
  +x → 东
  +y → 北
  +z → 天顶（垂直向上）

方位角 γ_s（太阳方位角）：
  0°  = 正南方向（正午太阳在正南）
  +90° = 正东方向
  -90° = 正西方向
  注意：这是"南=0，逆时针为正"的约定

塔位置：(0, 0) 为场地中心
镜面位置 (x_i, y_i)：在高斯平面投影中，东向和北向坐标
安装高度 h_i：镜面下边缘离地高度（m），镜面中心 z = h_i + H/2
```

## 积日 D 的两种约定

| 约定 | 公式 | 春分对应 | 使用场景 |
|------|------|----------|----------|
| `from_spring_equinox` | D = doy - 80 | D = 0 | **默认推荐** |
| `from_jan1` | D = doy | D = 80 | 备选 |

可通过 `solar_position.py` 的函数参数 `convention` 切换。
两种约定的结果差异极小（< 0.5° 高度角偏差），选用哪种对最终优化结果影响可忽略。

## 项目结构

```
csp_heliostat/
├── config/                 # 参数配置
│   ├── constants.py        #   全局常量 + 坐标系文档
│   └── sampling.py         #   60时刻采样表 + 权重
├── core/                   # 基础物理模型（纯函数）
│   ├── solar_position.py   #   太阳位置（δ, ω, α_s, γ_s）
│   ├── dni.py              #   直接法向辐照度
│   ├── geometry.py         #   反射几何（ŝ, r̂, n̂）
│   └── atmosphere.py       #   大气透过率（Sandia经验式）
├── field/                  # 镜场建模
│   ├── mirror.py           #   Mirror dataclass
│   ├── layout.py           #   径向同心/栅格/螺旋布局
│   └── constraints.py      #   间距/禁区/边界检查
├── efficiency/             # 五项效率
│   ├── cosine.py           #   余弦效率 η_cos
│   ├── shadow_blocking.py  #   阴影遮挡 η_sb (Numba)
│   ├── truncation.py       #   截断效率 η_trunc (Numba)
│   ├── atmospheric.py      #   大气透过率 η_at
│   └── composition.py      #   总效率装配
├── simulation/             # 仿真器
│   ├── snapshot.py         #   单时刻仿真
│   └── annual.py           #   60时刻加权年平均
├── optimization/           # 优化求解
│   ├── uniform.py          #   问题二：DE均匀优化
│   └── heterogeneous.py    #   问题三：分环异构 + warm-start + 收敛回调
├── analysis/               # 后处理分析（改进项）
│   ├── sensitivity.py      #   改进#1：敏感性分析
│   ├── efficiency_breakdown.py  #   改进#2：分项效率拆解
│   ├── weighting.py        #   改进#3：加权方法对照
│   ├── monthly.py          #   改进#6：月度均值统计
│   ├── monte_carlo.py      #   改进#7：蒙特卡洛置信带
│   └── diurnal.py          #   改进#8：全天曲线
├── visualization/          # 可视化
│   └── layout_plot.py      #   布局图 + 热力图 + 年曲线 + 改进项图表
└── tests/                  # 单元测试
main_problem1.py            # 问题一入口
main_problem2.py            # 问题二入口
main_problem3.py            # 问题三入口
main_improvements.py        # 改进清单统一入口（8项分析+可视化）
```

## 改进清单

| # | 优先级 | 改进项 | 输出 |
|---|--------|--------|------|
| 1 | 🔴 高 | P/A 敏感性分析 | sensitivity_*.png (4) |
| 2 | 🔴 高 | 分项效率拆解 | efficiency_breakdown_pie/bar.png |
| 3 | 🔴 高 | 加权方法对照 | weighting_comparison.png |
| 4 | 🟡 中 | 法向量彩图 | layout_with_normals_*.png (2) |
| 5 | 🟡 中 | 问题三收敛曲线 | problem3_convergence.png |
| 6 | 🟡 中 | 月度均值表 | monthly_power_bar.png |
| 7 | 🟢 低 | 蒙特卡洛置信带 | monte_carlo_confidence.csv |
| 8 | 🟢 低 | 夏至/冬至全天曲线 | diurnal_*.png (3) |

运行方式：
```bash
python main_improvements.py              # 执行全部 8 项改进
python main_improvements.py --fast       # 快速模式（跳过蒙特卡洛+问题三重跑）
python main_improvements.py --skip-mc    # 跳过蒙特卡洛
```

## 核心公式

| 符号 | 含义 | 公式 |
|------|------|------|
| δ | 赤纬角 | 23.45°·sin(2π(284+doy)/365) |
| ω | 时角 | 15°·(t-12) |
| α_s | 太阳高度角 | sin⁻¹(sin φ sin δ + cos φ cos δ cos ω) |
| γ_s | 太阳方位角 | -atan2(sin ω, cos ω sin φ − tan δ cos φ) |
| DNI | 法向辐照度 | G_0·[a(H)+b(H)·exp(−c(H)/sin α_s)] |
| n̂ | 镜面法向 | (ŝ+r̂)/|ŝ+r̂| |
| η_cos | 余弦效率 | max(0, n̂·ŝ) |
| η | 总光学效率 | ρ·η_sb·η_cos·η_at·η_trunc |
| P | 输出热功率 | Σᵢ Aᵢ·DNI·ηᵢ |

## 年平均加权

使用 `weight = max(0, sin α_s)` 进行加权平均：
- 物理意义：日照辐射的相对贡献
- 避免早晚极端低角时刻主导平均值
- 归一化使得 Σ weight = 1

## 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 编程语言 | Python + NumPy | 向量化、生态成熟 |
| 优化算法 | 差分进化 (DE) | 解析梯度缺失、混合变量 |
| 阴影遮挡 | 栅格投影 + cKDTree | 速度与精度平衡 |
| 截断判断 | 光线-圆柱求交 | 几何精确 |
| Numba 加速 | shadow_blocking, truncation | 热点函数，否则超时 |
| 问题三 warm-start | 问题二最优解 | 显著提升收敛 |

## 性能预算

| 步骤 | 目标耗时 | 实际 |
|------|----------|------|
| 单时刻仿真 (N=2300) | < 2 s | ~0.9 s |
| 全年 60 时刻 | < 2 min | ~53 s |
| 问题一完整 | < 5 min | ~1 min |

## 问题一验证结果（基准场景）

- 镜面数：2305（18环径向布局，6m×6m）
- 年均光学效率：0.558
- 年均输出功率：46.12 MW
- 功率范围验证：✅ 在 [30, 90] MW 预期范围内

## 许可与参考文献

- DNI 模型：标准大气辐射模型
- 大气透过率：Sandia/DLR 经验式
- 优化算法：scipy.optimize.differential_evolution