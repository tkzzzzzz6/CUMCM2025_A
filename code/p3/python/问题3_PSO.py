# -*- coding: utf-8 -*-
"""
Problem 3: 三枚烟幕弹对单导弹的协同遮蔽优化
依赖: numpy
- 控制台输出简洁报告（不导出 Excel）。
"""
import math
import os
import numpy as np
import time
import random
import copy
 

# -----------------------------
# 1) 物理参数
# -----------------------------
g = 9.80665  # 重力加速度 (m/s^2)
SMOKE_EFFECTIVE_RADIUS = 10.0  # 烟幕云团有效遮蔽半径 (m)，指云团中心到目标连线的最大判定距离
SINK_SPEED = 3.0  # 云团起爆后下沉速度 (m/s)
EFFECTIVE_TIME = 20.0  # 单枚烟幕弹起爆后对目标的有效遮蔽时长 (s)
MISSILE_SPEED = 300.0  # 来袭导弹速度 (m/s)
UAV_START_FY1 = np.array([17800.0, 0.0, 1800.0], dtype=float)  # FY1 无人机初始位置 (x, y, z)
MISSILE_START_M1 = np.array([20000.0, 0.0, 2000.0], dtype=float)  # M1 导弹被发现时刻的位置 (x, y, z)
TARGET_CENTER = np.array([0.0, 200.0, 5.0], dtype=float)  # 目标中心坐标 (x, y, z)，用于视线遮蔽判定

# 将以上物理参数绑定到原有变量名（尽量少改动后续代码）
R = SMOKE_EFFECTIVE_RADIUS
cloud_sink = SINK_SPEED
M0 = MISSILE_START_M1
to_origin = -M0 / np.linalg.norm(M0)
vM = MISSILE_SPEED * to_origin
U0 = UAV_START_FY1
T_center = TARGET_CENTER

def M_pos(t: float) -> np.ndarray:
    return M0 + vM * t

# -------------------------------------
# 2) 核心计算与适应度函数
# -------------------------------------
def point_to_segment_distance(c: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return np.linalg.norm(c - a), 0.0
    lam = float(np.dot(c - a, ab) / denom)
    lam_clamped = max(0.0, min(1.0, lam))
    d = float(np.linalg.norm(c - (a + lam_clamped * ab)))
    return d, lam_clamped

def calculate_total_coverage_union(params, N_grid=1001, is_final_calc=False):
    # 全局强制参数钳制，确保任何来源的参数都满足题设边界
    clamped = [check_bounds(v, PARAM_BOUNDS[i]) for i, v in enumerate(params)]
    angle, speed, t_r1, t_f1, dt_r2, t_f2, dt_r3, t_f3 = clamped
    t_releases = [t_r1, t_r1 + dt_r2, t_r1 + dt_r2 + dt_r3]
    t_fuzes = [t_f1, t_f2, t_f3]
    angle_rad = np.deg2rad(angle)
    uav_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

    bombs_data = []
    for t_r, t_f in zip(t_releases, t_fuzes):
        p_release = U0 + uav_dir * speed * t_r
        explosion_point = np.array([
            p_release[0] + uav_dir[0] * speed * t_f,
            p_release[1] + uav_dir[1] * speed * t_f,
            U0[2] - 0.5 * g * (t_f ** 2)
        ])
        t_explode = t_r + t_f
        bombs_data.append({'E': explosion_point, 't_exp': t_explode, 'P_rel': p_release})

    min_exp_time = min(b['t_exp'] for b in bombs_data)
    # 遮蔽只在起爆后 EFFECTIVE_TIME 内有效，将评估窗上限限制为各弹 t_exp+EFFECTIVE_TIME 的最大值
    t0 = min_exp_time
    t1 = max(b['t_exp'] + EFFECTIVE_TIME for b in bombs_data)
    t_grid = np.linspace(t0, t1, N_grid)
    dt = t_grid[1] - t_grid[0]
    
    union_mask = np.zeros_like(t_grid, dtype=bool)
    for i, t in enumerate(t_grid):
        m_pos = M_pos(t)
        is_obscured_this_step = False
        for bomb in bombs_data:
            if t >= bomb['t_exp'] and t <= bomb['t_exp'] + EFFECTIVE_TIME:
                c_pos = bomb['E'] - np.array([0, 0, cloud_sink * (t - bomb['t_exp'])])
                d, _ = point_to_segment_distance(c_pos, m_pos, T_center)
                if d <= R:
                    is_obscured_this_step = True
                    break
        union_mask[i] = is_obscured_this_step
        
    total_duration = np.sum(union_mask) * dt
    
    if is_final_calc:
        return total_duration, bombs_data
    else:
        return total_duration

# 低/高保真适应度缓存以加速计算
FITNESS_CACHE = {}

def get_fitness(params, fidelity='low'):
    key = (tuple([round(check_bounds(v, PARAM_BOUNDS[i]), 3) for i, v in enumerate(params)]), fidelity)
    if key in FITNESS_CACHE:
        return FITNESS_CACHE[key]
    N = 301 if fidelity == 'low' else 1501
    val = calculate_total_coverage_union(params, N_grid=N, is_final_calc=False)
    FITNESS_CACHE[key] = val
    return val

def calculate_individual_durations(bombs_data, N_grid=5001):
    """
    计算每枚烟幕弹单独作用下（忽略相互重叠）的有效遮蔽时长。
    使用与联合计算相同的时间窗与判定标准，便于对比。
    """
    min_exp_time = min(b['t_exp'] for b in bombs_data)
    t1_cap = max(b['t_exp'] + EFFECTIVE_TIME for b in bombs_data)
    t0, t1 = min_exp_time, t1_cap
    t_grid = np.linspace(t0, t1, N_grid)
    dt = t_grid[1] - t_grid[0]

    durations = []
    for bomb in bombs_data:
        mask = np.zeros_like(t_grid, dtype=bool)
        for i, t in enumerate(t_grid):
            if t >= bomb['t_exp']:
                c_pos = bomb['E'] - np.array([0, 0, cloud_sink * (t - bomb['t_exp'])])
                d, _ = point_to_segment_distance(c_pos, M_pos(t), T_center)
                if d <= R:
                    mask[i] = True
        durations.append(float(np.sum(mask) * dt))
    return durations

#############################################
# 3) 粒子群算法 (PSO) 模块
#############################################

PARAM_BOUNDS = [
    (0, 360),      # 角度
    (70, 140),     # 速度 (题设约束)
    (0.0, 12.0),   # t_r1
    (0.005, 12.0), # t_f1
    (1.0, 12.0),   # dt_r2 >= 1s
    (0.005, 12.0), # t_f2
    (1.0, 12.0),   # dt_r3 >= 1s
    (0.005, 12.0), # t_f3
]

def check_bounds(val, bounds):
    return max(bounds[0], min(bounds[1], val))

def project_params(params):
    q = [check_bounds(v, b) for v, b in zip(params, PARAM_BOUNDS)]
    q[0] = q[0] % 360.0
    q[4] = max(q[4], 1.0)
    q[6] = max(q[6], 1.0)
    return q

def run_pso_optimization():
    print("=== 开始执行三枚烟幕弹协同策略优化 (PSO) ===")
    random.seed(42); np.random.seed(42)

    # PSO 超参数
    SWARM_SIZE = 80
    ITERATIONS = 150
    W = 0.6           # 惯性权重
    C1 = 1.6          # 个体学习因子
    C2 = 1.6          # 群体学习因子

    dim = len(PARAM_BOUNDS)
    # 初始化粒子位置/速度
    positions = np.array([[random.uniform(b[0], b[1]) for b in PARAM_BOUNDS] for _ in range(SWARM_SIZE)], dtype=float)
    velocities = np.zeros((SWARM_SIZE, dim), dtype=float)

    # 个体最优与全局最优
    pbest = positions.copy()
    pbest_fit = np.array([get_fitness(project_params(list(p)), fidelity='low') for p in pbest])
    g_idx = int(np.argmax(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    start_time = time.time()
    best_at_10 = None; best_at_10_fit = -1.0

    for it in range(ITERATIONS):
        # 速度与位置更新
        r1 = np.random.rand(SWARM_SIZE, dim)
        r2 = np.random.rand(SWARM_SIZE, dim)
        velocities = W*velocities + C1*r1*(pbest - positions) + C2*r2*(gbest - positions)

        positions = positions + velocities
        # 约束投影
        positions = np.array([project_params(list(p)) for p in positions], dtype=float)

        # 评估 (低保真)
        fits = np.array([get_fitness(list(p), fidelity='low') for p in positions])

        # 个体最优更新
        improved = fits > pbest_fit
        pbest[improved] = positions[improved]
        pbest_fit[improved] = fits[improved]

        # 全局最优更新 (高保真对顶)
        g_idx = int(np.argmax(pbest_fit))
        gbest_candidate = pbest[g_idx].copy()
        gbest_candidate_fit = get_fitness(list(gbest_candidate), fidelity='high')
        if gbest_candidate_fit > gbest_fit:
            gbest, gbest_fit = gbest_candidate, gbest_candidate_fit
            angle, speed, t_r1, t_f1, dt_r2, t_f2, dt_r3, t_f3 = gbest
            print("[刷新历史遮挡最长]")
            print("无人机飞行速度(V_FY1)：{:.2f} 米/秒".format(speed))
            print("无人机飞行方向(theta_FY1)：{:.2f} 度".format(angle))
            print("投放前飞行时间(t_fly)：{:.2f} 秒".format(t_r1))
            print("烟幕弹引信时间(t_fuse)：{:.2f} 秒".format(t_f1))

        # 10s 快照
        if best_at_10 is None and (time.time() - start_time) >= 10.0:
            best_at_10 = gbest.copy(); best_at_10_fit = gbest_fit

        print(f"迭代 {it+1:3d}/{ITERATIONS} | 本轮最佳(低保真): {np.max(fits):.3f} s | 历史最佳(高保真): {gbest_fit:.3f} s")

    total_time = time.time() - start_time
    if best_at_10 is None:
        best_at_10 = gbest.copy(); best_at_10_fit = gbest_fit

    # 输出摘要
    try:
        out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'summary_p3_PSO.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("算法总体运行时间: {:.2f} s\n".format(total_time))
            f.write("\n[10秒时刻的最优策略]\n")
            angle, speed, t_r1, t_f1, dt_r2, t_f2, dt_r3, t_f3 = best_at_10
            f.write("最优无人机飞行方向(theta_FY1)：{:.3f} 度\n".format(angle))
            f.write("最优无人机飞行速度(V_FY1)：{:.2f} 米/秒\n".format(speed))
            t_releases = [t_r1, t_r1 + dt_r2, t_r1 + dt_r2 + dt_r3]
            t_fuzes = [t_f1, t_f2, t_f3]
            for i in range(3):
                f.write("第{}枚：飞行时间{:.3f}s，引信时间{:.3f}s\n".format(
                    i+1,
                    t_releases[i] - (0 if i==0 else t_releases[i-1]),
                    t_fuzes[i]
                ))
            f.write("[10秒时最优的高保真遮蔽时长：{:.3f}s]\n".format(best_at_10_fit))

            # 写入总体最优策略（全程高保真最佳）
            f.write("\n[总体最优策略]\n")
            angle_g, speed_g, t_r1_g, t_f1_g, dt_r2_g, t_f2_g, dt_r3_g, t_f3_g = gbest
            f.write("最优无人机飞行方向(theta_FY1)：{:.3f} 度\n".format(angle_g))
            f.write("最优无人机飞行速度(V_FY1)：{:.2f} 米/秒\n".format(speed_g))
            f.write("\n[总体时最优的高保真遮蔽时长：{:.3f}s]\n".format(gbest_fit))
    except Exception as e:
        print("写入输出摘要失败:", e)

    return list(gbest)

# -----------------------------
# 4) 主流程与格式化输出
# -----------------------------
def format_and_output_results(best_params):
    """
    对最优策略进行高精度计算，并按指定格式输出到控制台。
    """
    print("\n=== 对最优策略进行高精度计算并生成报告... ===")
    
    # 1. 高精度计算
    total_duration_precise, bombs_data_precise = calculate_total_coverage_union(
        best_params, N_grid=5001, is_final_calc=True
    )
    
    # 2. 准备数据
    angle, speed, t_r1, t_f1, dt_r2, t_f2, dt_r3, t_f3 = best_params
    t_releases = [t_r1, t_r1 + dt_r2, t_r1 + dt_r2 + dt_r3]
    t_fuzes = [t_f1, t_f2, t_f3]

    # 3. 打印到控制台（简洁标签格式 + 最终总结）
    print("无人机运动方向 (度): {:.2f}".format(angle))
    print("无人机运动速度 (m/s): {:.2f}".format(speed))
    for i in range(3):
        p = bombs_data_precise[i]['P_rel']
        e = bombs_data_precise[i]['E']
        print("烟幕干扰弹编号: {}".format(i + 1))
        print("  投放时间 (s): {:.2f}".format(t_releases[i]))
        print("  引信时间 (s): {:.2f}".format(t_fuzes[i]))
        print("  烟幕干扰弹投放点的x坐标 (m): {:.3f}".format(p[0]))
        print("  烟幕干扰弹投放点的y坐标 (m): {:.3f}".format(p[1]))
        print("  烟幕干扰弹投放点的z坐标 (m): {:.3f}".format(p[2]))
        print("  烟幕干扰弹起爆点的x坐标 (m): {:.3f}".format(e[0]))
        print("  烟幕干扰弹起爆点的y坐标 (m): {:.3f}".format(e[1]))
        print("  烟幕干扰弹起爆点的z坐标 (m): {:.3f}".format(e[2]))
    print("有效干扰时长 (s): {:.4f}".format(total_duration_precise))

    # 4. 计算并打印每枚烟幕弹的独立有效时长与最终最优信息
    indiv_durations = calculate_individual_durations(bombs_data_precise, N_grid=5001)
    print("\n—— 最终策略总结 ——")
    print("最优无人机飞行方向(theta_FY1)：{:.3f} 度".format(angle))
    print("最优无人机飞行速度(V_FY1)：{:.2f} 米/秒".format(speed))
    print("最优策略下的参数及各烟雾弹独立贡献：")
    for i in range(3):
        print("第{}枚：飞行时间{:.3f}s，引信时间{:.3f}s.".format(i+1, [t_r1, t_r1+dt_r2, t_r1+dt_r2+dt_r3][i] - (0 if i==0 else [t_r1, t_r1+dt_r2, t_r1+dt_r2+dt_r3][i-1]), [t_f1, t_f2, t_f3][i]))
        print("[独立有效时长：{:.3f}s]".format(indiv_durations[i]))

if __name__ == "__main__":
    best_params = run_pso_optimization()
    if best_params:
        format_and_output_results(best_params)