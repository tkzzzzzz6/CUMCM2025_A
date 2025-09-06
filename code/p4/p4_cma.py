# -*- coding: utf-8 -*-
"""
Problem 3: 三架无人机协同对单导弹的遮蔽优化
依赖: numpy, cma
- [最终策略版 - 全局视野优化] 采用终极版适应度函数，通过计算并求和所有无人机的“潜力得分”，
  为高维搜索提供丰富、准确的梯度，引导算法寻找真正的协同最优解。
"""
import math
import numpy as np
import time
import cma

# -----------------------------
# 1) 题设常量与几何/运动模型
# -----------------------------
g = 9.8
R = 10.0
cloud_sink = 3.0
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
to_origin = -M0 / np.linalg.norm(M0)
vM = 300.0 * to_origin
T_center = np.array([0.0, 200.0, 5.0], dtype=float)

UAV_INITIAL_POS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
}
UAV_NAMES = ['FY1', 'FY2', 'FY3']

def M_pos(t: float) -> np.ndarray:
    return M0 + vM * t

# -------------------------------------
# 2) 核心计算与适应度函数 (最终版)
# -------------------------------------
def point_to_segment_distance(c: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a; denom = float(np.dot(ab, ab))
    if denom < 1e-9: return np.linalg.norm(c - a), 0.0
    lam = float(np.dot(c - a, ab) / denom)
    d = float(np.linalg.norm(c - (a + max(0.0, min(1.0, lam)) * ab)))
    return d, lam

def calculate_fitness_and_details_multi_uav(params, uav_positions, N_grid=501, is_final_calc=False):
    num_uavs = len(uav_positions)
    bombs_data = []
    param_chunks = [params[i:i + 4] for i in range(0, len(params), 4)]

    for i in range(num_uavs):
        uav_name = list(uav_positions.keys())[i]
        uav_pos0 = uav_positions[uav_name]
        angle, speed, t_r, t_f = param_chunks[i]
        angle_rad = np.deg2rad(angle); uav_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        p_release = uav_pos0 + uav_dir * speed * t_r
        explosion_point = np.array([p_release[0]+uav_dir[0]*speed*t_f, p_release[1]+uav_dir[1]*speed*t_f, uav_pos0[2]-0.5*g*(t_f**2)])
        t_explode = t_r + t_f
        bombs_data.append({'E': explosion_point, 't_exp': t_explode, 'P_rel': p_release})

    min_exp_time = min(b['t_exp'] for b in bombs_data) if bombs_data else 0
    max_exp_time = max(b['t_exp'] for b in bombs_data) if bombs_data else 0
    t0, t1 = min_exp_time, max_exp_time + 20.0
    t_grid = np.linspace(t0, t1, N_grid)
    dt = t_grid[1] - t_grid[0] if N_grid > 1 else 0
    
    union_mask = np.zeros_like(t_grid, dtype=bool)
    for i_t, t in enumerate(t_grid):
        m_pos = M_pos(t); is_obscured = False
        for bomb in bombs_data:
            if t >= bomb['t_exp']:
                c_pos = bomb['E'] - np.array([0, 0, cloud_sink * (t - bomb['t_exp'])])
                d, lam = point_to_segment_distance(c_pos, m_pos, T_center)
                if d <= R and 0.0 <= lam <= 1.0: is_obscured = True; break
        union_mask[i_t] = is_obscured
        
    duration = np.sum(union_mask) * dt

    if is_final_calc: return duration, bombs_data
    
    # --- [最终改良] 全局视野地形塑造逻辑 ---
    if duration > 0:
        fitness = 1000 + duration
    else:
        total_potential_fitness = 0
        for bomb in bombs_data:
            min_dist_bomb, lam_at_min_dist_bomb = float('inf'), -1.0
            # 在每个弹的有效窗口内，找到其最小距离
            bomb_window = np.linspace(bomb['t_exp'], bomb['t_exp'] + 20.0, 101) # 用较低分辨率快速评估
            for t_bomb in bomb_window:
                c_pos = bomb['E'] - np.array([0, 0, cloud_sink * (t_bomb - bomb['t_exp'])])
                d, lam = point_to_segment_distance(c_pos, M_pos(t_bomb), T_center)
                if d < min_dist_bomb: min_dist_bomb, lam_at_min_dist_bomb = d, lam
            
            # 如果该弹位置合理，则累加其潜力得分
            if 0.0 <= lam_at_min_dist_bomb <= 1.0:
                potential = 1.0 / (min_dist_bomb - R + 1e-6) if min_dist_bomb > R else 1e6
                total_potential_fitness += potential
        fitness = total_potential_fitness
        
    return fitness, duration

# ---------------------------------------------
# 3) CMA-ES 优化模块
# ---------------------------------------------
PARAM_BOUNDS = []
param_template = [(0, 360), (70, 140), (0.01, 15.0), (0.5, 10.0)]
for _ in range(len(UAV_NAMES)): PARAM_BOUNDS.extend(param_template)

def objective_function_cma(params):
    fitness, _ = calculate_fitness_and_details_multi_uav(params, UAV_INITIAL_POS)
    return -fitness

def run_cma_es_optimization():
    print("=== 开始执行 CMA-ES 协同策略优化 (全局视野版) ===")
    start_time = time.time()
    
    x0 = [(b[0] + b[1]) / 2 for b in PARAM_BOUNDS]
    sigma0 = 8.0 
    stds_template = [90, 18, 4.0, 2.5]
    stds = []
    for _ in range(len(UAV_NAMES)): stds.extend(stds_template)
    bounds_low = [b[0] for b in PARAM_BOUNDS]; bounds_high = [b[1] for b in PARAM_BOUNDS]
    
    options = {'bounds': [bounds_low, bounds_high], 'CMA_stds': stds, 'maxiter': 500, 'popsize': 40, 'tolfun': 1e-5}
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    best_duration_overall = -1.0
    while not es.stop():
        solutions = es.ask()
        eval_results = [calculate_fitness_and_details_multi_uav(s, UAV_INITIAL_POS) for s in solutions]
        fitnesses = [-res[0] for res in eval_results]
        es.tell(solutions, fitnesses)
        
        current_best_duration = max(res[1] for res in eval_results)
        if current_best_duration > best_duration_overall: best_duration_overall = current_best_duration
        
        print(f"代数 {es.countiter:3d}/{options['maxiter']} | "
              f"本代最长: {current_best_duration:.3f} s | "
              f"历史最长: {best_duration_overall:.3f} s | "
              f"最优函数值: {es.result.fbest:.2f}")

    best_params = es.result.xbest
    print(f"\n优化完成，耗时: {time.time() - start_time:.2f} s")
    print(f"找到的历史最长时长: {best_duration_overall:.4f} s")
    
    return best_params

# -----------------------------
# 4) 主流程与格式化输出
# -----------------------------
def format_and_output_results(best_params):
    print("\n=== 对最优协同策略进行高精度计算并生成报告... ===")
    total_duration_precise, bombs_data_precise = calculate_fitness_and_details_multi_uav(
        best_params, UAV_INITIAL_POS, N_grid=5001, is_final_calc=True
    )
    
    print("\n" + "="*80)
    print(" " * 28 + "多无人机协同烟幕干扰弹投放策略")
    print("="*80)
    for i in range(len(UAV_NAMES)):
        angle, speed, t_r, t_f = best_params[i*4:(i+1)*4]
        p, e = bombs_data_precise[i]['P_rel'], bombs_data_precise[i]['E']
        print(f" 无人机 {UAV_NAMES[i]} 策略 ".center(50, '-'))
        print(f"  飞行速度: {speed:.2f} m/s".ljust(25) + f"飞行方向: {angle:.2f} 度")
        print(f"  投放时间: {t_r:.2f} s".ljust(25) + f"引信时间: {t_f:.2f} s")
        print(f"  投放点 (x,y,z): ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
        print(f"  起爆点 (x,y,z): ({e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f})")
    print("="*80)
    print(f"总有效遮蔽时长 (协同): {total_duration_precise:.4f} s")
    print("="*80)

if __name__ == "__main__":
    best_params = run_cma_es_optimization()
    if best_params is not None:
        format_and_output_results(best_params)