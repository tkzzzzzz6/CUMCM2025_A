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
import matplotlib.pyplot as plt

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

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
    # 以各弹 t_exp+EFFECTIVE_TIME 的最大值作为右端
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

# ---------------------------------------------
# 3) 遗传算法 (Genetic Algorithm) 模块
# ---------------------------------------------
POP_SIZE = 120
GENERATIONS = 100 #迭代次数
CXPB, MUTPB = 0.9, 0.2
ELITE_COUNT = 2
ETA = 20
PARAM_BOUNDS = [
    (0, 360),      # 角度
    (70, 140),     # 速度 (题设约束)
    (0.0, 12.0),   # t_r1
    (0.005, 12.0), # t_f1
    (1.0, 12.0),   # dt_r2 >= 1s (题设约束)
    (0.005, 12.0), # t_f2
    (1.0, 12.0),   # dt_r3 >= 1s (题设约束)
    (0.005, 12.0), # t_f3
]

def check_bounds(val, bounds): return max(bounds[0], min(bounds[1], val))
def _sample_biased_angle():
    """优先在[0°,20°]∪[340°,360°]采样角度，提升小偏角探索概率。"""
    if random.random() < 0.7:
        # 70% 概率采样小偏角区
        if random.random() < 0.5:
            return random.uniform(0.0, 20.0)
        else:
            return random.uniform(340.0, 360.0)
    # 30% 保留全局探索
    return random.uniform(0.0, 360.0)

def create_individual():
    ind = []
    for i, b in enumerate(PARAM_BOUNDS):
        if i == 0:  # 角度
            ind.append(_sample_biased_angle())
        else:
            ind.append(random.uniform(b[0], b[1]))
    return ind
def crossover_sbx(p1,p2): c1,c2=copy.deepcopy(p1),copy.deepcopy(p2); [crossover_sbx_gene(c1,c2,i) for i in range(len(p1))]; return c1,c2
def crossover_sbx_gene(c1,c2,i):
    if random.random()>0.5: return
    y1,y2,(yl,yu)=min(c1[i],c2[i]),max(c1[i],c2[i]),PARAM_BOUNDS[i]; u=random.random()
    if y2-y1>1e-6:
        beta=1.0+(2.0*(y1-yl)/(y2-y1)); alpha=2.0-beta**(-(ETA+1.0)); beta_q=(u*alpha)**(1.0/(ETA+1.0)) if u<=1.0/alpha else (1.0/(2.0-u*alpha))**(1.0/(ETA+1.0)); c1[i]=0.5*((y1+y2)-beta_q*(y2-y1))
        beta=1.0+(2.0*(yu-y2)/(y2-y1)); alpha=2.0-beta**(-(ETA+1.0)); beta_q=(u*alpha)**(1.0/(ETA+1.0)) if u<=1.0/alpha else (1.0/(2.0-u*alpha))**(1.0/(ETA+1.0)); c2[i]=0.5*((y1+y2)+beta_q*(y2-y1))
    c1[i],c2[i]=check_bounds(c1[i],PARAM_BOUNDS[i]),check_bounds(c2[i],PARAM_BOUNDS[i])
def mutate_polynomial(ind): [mutate_polynomial_gene(ind,i) for i in range(len(ind))]; return ind
def mutate_polynomial_gene(ind,i):
    if random.random()<MUTPB:
        yl,yu=PARAM_BOUNDS[i]; delta1,delta2=(ind[i]-yl)/(yu-yl),(yu-ind[i])/(yu-yl); u,mut_pow=random.random(),1.0/(ETA+1.0)
        if u<0.5: val=2.0*u+(1.0-2.0*u)*(1.0-delta1)**(ETA+1.0); delta_q=val**mut_pow-1.0
        else: val=2.0*(1.0-u)+2.0*(u-0.5)*(1.0-delta2)**(ETA+1.0); delta_q=1.0-val**mut_pow
        ind[i]+=delta_q*(yu-yl); ind[i]=check_bounds(ind[i],PARAM_BOUNDS[i])
    # 变异后再次校验“投放间隔≥1s”的耦合约束
    # i 发生改变时，确保 dt_r2、dt_r3 不小于 1.0
    if i in (4,6):
        ind[i] = max(ind[i], 1.0)
    # 角度方向的引导性探索：向小偏角区轻微吸引
    if i == 0:
        if random.random() < 0.3:  # 30% 概率朝 0° 或 360° 拉近一点
            target = 0.0 if random.random() < 0.5 else 360.0
            ind[i] = check_bounds( ind[i] + 0.1*(target - ind[i]), PARAM_BOUNDS[i] )

def run_ga_optimization():
    print("=== 开始执行三枚烟幕弹协同策略优化 ===")
    random.seed(42); np.random.seed(42)
    population = [create_individual() for _ in range(POP_SIZE)]
    best_ind_overall, best_fitness_overall = None, -1.0
    start_time = time.time()
    best_at_10, best_at_10_fit = None, -1.0
    gen_best_list, hist_best_list = [], []

    for gen in range(GENERATIONS):
        # 先用低保真评估整个种群
        fitnesses = [get_fitness(ind, fidelity='low') for ind in population]
        elite_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k], reverse=True)[:ELITE_COUNT]
        elites = [population[i] for i in elite_indices]

        # 对前若干名做高保真校正
        top_k = 10 if len(population) >= 50 else max(2, len(population)//5)
        top_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k], reverse=True)[:top_k]
        for idx in top_indices:
            fitnesses[idx] = get_fitness(population[idx], fidelity='high')
        elite_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k], reverse=True)[:ELITE_COUNT]
        elites = [population[i] for i in elite_indices]

        if fitnesses[elite_indices[0]] > best_fitness_overall:
            best_fitness_overall = fitnesses[elite_indices[0]]
            best_ind_overall = copy.deepcopy(elites[0])
            angle, speed, t_r1, t_f1, dt_r2, t_f2, dt_r3, t_f3 = best_ind_overall
            print("[刷新历史遮挡最长]")
            print("无人机飞行速度(V_FY1)：{:.2f} 米/秒".format(speed))
            print("无人机飞行方向(theta_FY1)：{:.2f} 度".format(angle))
            print("投放前飞行时间(t_fly)：{:.2f} 秒".format(t_r1))
            print("烟幕弹引信时间(t_fuse)：{:.2f} 秒".format(t_f1))

        # 到达10秒快照（仅记录一次），用高保真评估对齐
        if best_at_10 is None and (time.time() - start_time) >= 10.0:
            best_at_10 = copy.deepcopy(best_ind_overall) if best_ind_overall is not None else None
            if best_at_10 is not None:
                best_at_10_fit = get_fitness(best_at_10, fidelity='high')

        gen_best = max(fitnesses)
        gen_best_list.append(gen_best)
        if len(hist_best_list) == 0:
            hist_best_list.append(gen_best)
        else:
            hist_best_list.append(max(hist_best_list[-1], gen_best))
        print(f"代数 {gen+1:3d}/{GENERATIONS} | "f"本代最长遮蔽时间: {gen_best:.3f} s | "f"历史最长: {best_fitness_overall:.3f} s")

        offspring = [copy.deepcopy(e) for e in elites]
        while len(offspring) < POP_SIZE:
            # tournament selection (k=3)
            cand = random.sample(range(POP_SIZE), 3); p1 = max(cand, key=lambda idx: fitnesses[idx])
            cand = random.sample(range(POP_SIZE), 3); p2 = max(cand, key=lambda idx: fitnesses[idx])
            p1, p2 = population[p1], population[p2]
            if random.random()<CXPB: c1,c2=crossover_sbx(p1,p2)
            else: c1,c2=copy.deepcopy(p1),copy.deepcopy(p2)
            offspring.append(mutate_polynomial(c1));
            if len(offspring)<POP_SIZE: offspring.append(mutate_polynomial(c2))
        population = offspring

    print(f"\n优化完成，耗时: {time.time() - start_time:.2f} s")
    # 局部爬山细化（高保真评价）
    def refine(ind, steps=120, sigma=[2, 3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]):
        best = copy.deepcopy(ind); best_fit = get_fitness(best, fidelity='high')
        for _ in range(steps):
            trial = [check_bounds(v + np.random.normal(0, s), b) for v, s, b in zip(best, sigma, PARAM_BOUNDS)]
            f = get_fitness(trial, fidelity='high')
            if f > best_fit:
                best, best_fit = trial, f
        return best

    best_ind_overall = refine(best_ind_overall)
    total_time = time.time() - start_time

    # 如果算法总时长<10s，使用最终最优作为10s快照
    if best_at_10 is None:
        best_at_10 = copy.deepcopy(best_ind_overall)
        best_at_10_fit = get_fitness(best_at_10, fidelity='high')

    # 计算总体最优的高保真时长
    best_final_fit = get_fitness(best_ind_overall, fidelity='high')

    # 写入输出文本
    try:
        out_dir = os.path.join(os.path.dirname(__file__), 'output_GA')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'summary_p3_GA.txt')
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

            f.write("\n[总体最优策略]\n")
            angle_f, speed_f, t_r1f, t_f1f, dt_r2f, t_f2f, dt_r3f, t_f3f = best_ind_overall
            f.write("最优无人机飞行方向(theta_FY1)：{:.3f} 度\n".format(angle_f))
            f.write("最优无人机飞行速度(V_FY1)：{:.2f} 米/秒\n".format(speed_f))
            t_releases_f = [t_r1f, t_r1f + dt_r2f, t_r1f + dt_r2f + dt_r3f]
            t_fuzes_f = [t_f1f, t_f2f, t_f3f]
            for i in range(3):
                f.write("第{}枚：飞行时间{:.3f}s，引信时间{:.3f}s\n".format(
                    i+1,
                    t_releases_f[i] - (0 if i==0 else t_releases_f[i-1]),
                    t_fuzes_f[i]
                ))
            f.write("[总体时最优的高保真遮蔽时长：{:.3f}s]\n".format(best_final_fit))

        # 绘制并保存进度图（仅展示本代最长）
        plt.figure(figsize=(10,4.5))
        x = list(range(1, len(gen_best_list)+1))
        plt.plot(x, gen_best_list, 'o-', linewidth=1.8, markersize=4, color='#1f77b4', alpha=0.9, label='本代最长', zorder=2)
        plt.xlabel('代数')
        plt.ylabel('遮蔽时间 (s)')
        plt.title('GA 优化进度')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, 'ga_progress.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
    except Exception as e:
        print("写入输出摘要失败:", e)

    return best_ind_overall

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
    best_params = run_ga_optimization()
    if best_params:
        format_and_output_results(best_params)