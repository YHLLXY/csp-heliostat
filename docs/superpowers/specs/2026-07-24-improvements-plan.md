# 改进清单实施方案

> 基于 `outputs/改进清单与优先级.md` 的 8 项改进
> 基线：P̄ = 60.58 MW，η̄ = 0.5256，P/A = 525.9 W/m²，N = 3200，n_rings=25，W=6m，h=3m

---

## 架构总览

```
新增模块:
  csp_heliostat/analysis/          # 后处理分析包（全新）
  ├── __init__.py
  ├── sensitivity.py               # 改进 #1：单变量敏感性扫描
  ├── efficiency_breakdown.py      # 改进 #2：分项效率拆解
  ├── weighting.py                 # 改进 #3：四种加权方法对照
  ├── monthly.py                   # 改进 #6：月度均值统计
  ├── monte_carlo.py               # 改进 #7：蒙特卡洛置信带
  └── diurnal.py                   # 改进 #8：夏至/冬至全天曲线

  main_improvements.py             # 统一入口，逐一执行 8 项改进

修改模块:
  csp_heliostat/simulation/annual.py        # 改动：per_sample 透传五项效率分量
  csp_heliostat/optimization/heterogeneous.py # 改动：DE 增加 callback 记录收敛历史
  csp_heliostat/visualization/layout_plot.py  # 改动：新增 5 类图表函数
  README.md                                  # 改动：更新项目结构 + 更新日志
```

### 设计原则

1. **不改最优解**：所有改进只做分析+可视化，不动核心物理模型
2. **模块独立**：每个改进项一个文件，可单独运行也可统一入口调用
3. **最小侵入**：对现有代码的修改仅限于"透传数据"和"加回调"，不影响已有功能
4. **输出规范**：每个改进项输出独立 CSV + PNG 到 `outputs/data/` 和 `outputs/figure/`

---

## 改进项逐一设计

### 改进 #1：P/A 敏感性分析

**文件**：`csp_heliostat/analysis/sensitivity.py`

**函数签名**：
```python
def run_sensitivity(best_params, n_rings, schedule=None) -> dict
def sensitivity_single_var(var_name, values, fixed_params, n_rings) -> list
```

**逻辑**：
1. 加载最优解：x_t=0, y_t=0, W=6, h=3, n_rings=25
2. 对 5 个变量分别做单变量扫描，其余固定
3. 每个扫描点运行一次年度仿真（快速noon-only模式加速，最后用60时刻验证关键点）
4. 扫描范围：
   - x_t/y_t: linspace(-50, 50, 11)
   - W: linspace(5.0, 7.0, 11)
   - h: linspace(2.0, 4.0, 11)
   - n_rings: 22,23,24,25,26,27,28
5. 输出 n_rings 用快速模式，其余用完整60时刻

**输出**：
- `outputs/figure/sensitivity_x_t.png` + `sensitivity_y_t.png`（合并为一张双面板）
- `outputs/figure/sensitivity_W.png`
- `outputs/figure/sensitivity_h.png`
- `outputs/figure/sensitivity_n_rings.png`
- `outputs/data/sensitivity_summary.csv`

**新增图表函数**（`visualization/layout_plot.py`）：
```python
def plot_sensitivity_curve(var_name, values, P_values, P_per_area_values, save_path)
```

---

### 改进 #2：分项效率拆解

**前置改动**：`annual.py` 的 `per_sample` 字典中增加五项效率的年均值

**文件**：`csp_heliostat/analysis/efficiency_breakdown.py`

**函数签名**：
```python
def compute_efficiency_breakdown(annual_result) -> dict
```

**逻辑**：
1. 修改 `simulation/snapshot.py` 的 `simulate_one()`，返回中增加面积加权后的各分量均值
2. 修改 `simulation/annual.py`，在 per_sample 中存储各效率分量的镜场均值
3. 对各分量按 sin α_s 做加权年平
4. 验证 η_total ≈ ρ × η_cos × η_sb × η_trunc × η_at

**输出**：
- `outputs/figure/efficiency_breakdown_pie.png`
- `outputs/figure/efficiency_breakdown_bar.png`
- `outputs/data/efficiency_components.csv`

**新增图表函数**：
```python
def plot_efficiency_breakdown(components, save_path_pie, save_path_bar)
```

---

### 改进 #3：加权方法对照表

**文件**：`csp_heliostat/analysis/weighting.py`

**函数签名**：
```python
def compare_weighting_methods(mirrors, tower_xy, schedule) -> dict
```

**四种方法**：
| # | 方法 | 权重公式 |
|---|------|---------|
| 1 | 等权平均 | w = 1/60 |
| 2 | sin α_s 加权 | w ∝ max(0, sin α_s)（当前默认） |
| 3 | DNI 加权 | w ∝ DNI(t)（直接用 DNI 值归一化） |
| 4 | 余弦加权 | w ∝ max(0, cos(ω))（日照时长代理变量） |

**逻辑**：
1. 固定镜场配置（最优解），运行一次60时刻仿真
2. 拿到每时刻的 P(t)，用四种权重分别加权，算出四个 P̄
3. 计算最大偏差 = max(|P_i - P_mean|) / P_mean

**输出**：
- `outputs/data/weighting_comparison.csv`
- `outputs/figure/weighting_comparison.png`（分组柱状图）

---

### 改进 #4：法向量彩图

**文件**：直接在 `visualization/layout_plot.py` 中新增函数

**函数签名**：
```python
def plot_field_with_normals(mirrors, sun, tower_xy, tower_height, receiver_height, receiver_radius, save_path)
```

**逻辑**：
1. 取春分正午（D=0, t=12）和夏至正午（D=172-80=92, t=12）
2. 对每面镜计算 ŝ、r̂、n̂
3. 在布局图上叠加箭头（箭头方向=n̂的水平投影，颜色=η_cos，长度=η_cos缩放）
4. 已有 `geometry.py` 的 `mirror_normal()` 可直接复用

**输出**：
- `outputs/figure/layout_with_normals_spring_noon.png`
- `outputs/figure/layout_with_normals_summer_noon.png`

---

### 改进 #5：问题三收敛曲线

**前置改动**：`optimization/heterogeneous.py` 的 `differential_evolution` 调用增加 `callback` 参数

**文件**：改动在 `heterogeneous.py`，图表在 `visualization/layout_plot.py`

**逻辑**：
1. 在 `solve_problem3()` 中增加 callback 函数，每代记录 (gen, best_fitness, best_x)
2. callback 把数据写入列表 `convergence_history`
3. 结果存入 JSON，新增绘图函数

**新增图表函数**：
```python
def plot_convergence_curve(history, uniform_baseline_P_per_area, save_path)
```

**输出**：
- `outputs/figure/problem3_convergence.png`
- `outputs/data/problem3_convergence.csv`

---

### 改进 #6：月度均值表

**文件**：`csp_heliostat/analysis/monthly.py`

**函数签名**：
```python
def compute_monthly_stats(annual_result) -> dict
```

**逻辑**：
1. 从 `per_sample` 按 `month` 分组
2. 每月5个时刻求均值和标准差
3. 标注哪些月份 P̄_month < 60 MW
4. 估算储能需求：对每个低于60MW的月份，计算"需要补足的缺口能量"

**输出**：
- `outputs/figure/monthly_power_bar.png`
- `outputs/data/monthly_power.csv`

---

### 改进 #7：蒙特卡洛置信带

**文件**：`csp_heliostat/analysis/monte_carlo.py`

**函数签名**：
```python
def run_monte_carlo(mirrors, tower_xy, n_runs=10, seed_start=0) -> dict
```

**逻辑**：
1. 固定最优镜场配置
2. 对阴影遮挡算法的随机采样做 10 次独立重复
3. 每次改 random seed，记录 P̄
4. 计算 mean ± std，如果 std/mean < 1% 则通过

**输出**：
- `outputs/data/monte_carlo_confidence.csv`
- 在年度曲线上叠加 ±1σ 阴影带 → `outputs/figure/annual_curve_with_confidence.png`

---

### 改进 #8：夏至/冬至全天曲线

**文件**：`csp_heliostat/analysis/diurnal.py`

**函数签名**：
```python
def compute_diurnal_curve(mirrors, tower_xy, doy, hours_range) -> dict
```

**逻辑**：
1. 夏至（doy=172, D=92）和冬至（doy=355, D=275）
2. 从 6:00 到 18:00，每小时一个点（共 13 点）
3. 计算每个点的 ŝ、DNI、五项效率、P
4. 验证曲线关于正午对称

**输出**：
- `outputs/figure/diurnal_curve_summer.png`
- `outputs/figure/diurnal_curve_winter.png`
- `outputs/figure/diurnal_comparison.png`（叠加对比）

---

## 执行顺序与时间估算

| 批次 | 改进项 | 预估耗时 | 依赖 |
|------|--------|---------|------|
| **第一批** | #2 分项效率 + #3 加权对照 | 30 min | 需先改 annual.py |
| **第二批** | #1 敏感性分析 | 40 min | 无依赖，计算最密集 |
| **第三批** | #4 法向量彩图 | 20 min | 复用 geometry.py |
| **第四批** | #5 收敛曲线 + #6 月度均值 | 20 min | #5 需改 heterogeneous.py |
| **第五批** | #7 蒙特卡洛 + #8 全天曲线 | 30 min | 无依赖 |

**总预估**：约 2.5 小时（不含仿真计算等待时间）

---

## 风险分析

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 敏感性扫描耗时过长（5×11×60次仿真） | 高 | 等待时间长 | x_t/y_t/W/h 用 noon-only 加速，n_rings 只需 7 个点 |
| annual.py 改动影响现有结果 | 低 | 结果不一致 | 只增加透传字段，不改计算逻辑 |
| 问题三 DE 未保存历史，需重跑 | 中 | 需重新运行优化 | 增加 callback 后重跑一次问题三 |
| 蒙特卡洛随机种子不影响阴影结果 | 低 | 置信带为零宽度 | 检查 shadow_blocking 是否有随机性；若无则在报告中说明 |

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `csp_heliostat/analysis/__init__.py` | 包初始化 |
| 新建 | `csp_heliostat/analysis/sensitivity.py` | 改进 #1 |
| 新建 | `csp_heliostat/analysis/efficiency_breakdown.py` | 改进 #2 |
| 新建 | `csp_heliostat/analysis/weighting.py` | 改进 #3 |
| 新建 | `csp_heliostat/analysis/monthly.py` | 改进 #6 |
| 新建 | `csp_heliostat/analysis/monte_carlo.py` | 改进 #7 |
| 新建 | `csp_heliostat/analysis/diurnal.py` | 改进 #8 |
| 新建 | `main_improvements.py` | 统一入口 |
| 修改 | `csp_heliostat/simulation/annual.py` | per_sample 增加效率分量 |
| 修改 | `csp_heliostat/simulation/snapshot.py` | 返回值增加分量镜场均值 |
| 修改 | `csp_heliostat/optimization/heterogeneous.py` | 增加 callback 记录收敛 |
| 修改 | `csp_heliostat/visualization/layout_plot.py` | 新增 5 个图表函数 |
| 修改 | `csp_heliostat/visualization/__init__.py` | 导出新函数 |
| 修改 | `README.md` | 更新结构和日志 |

---

*计划结束。等待用户审批后开始执行。*