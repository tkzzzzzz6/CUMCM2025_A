# -*- coding: utf-8 -*-
"""
Problem 1: 单无人机对单导弹的烟幕遮蔽计算（可复现数值 & 可视化）
依赖: numpy, matplotlib
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# -----------------------------
# 1) 题设常量与几何/运动模型
# -----------------------------
g = 9.8  # m/s^2
R = 10.0  # 有效遮蔽半径（云团中心10 m范围）
cloud_sink = 3.0  # 云团下沉速度 m/s

# 导弹 M1 初态与速度（300 m/s，指向原点）
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
to_origin = -M0
to_origin /= np.linalg.norm(to_origin)  # 单位方向
vM = 300.0 * to_origin  # 导弹速度向量

def M_pos(t: float) -> np.ndarray:
    """导弹在时刻 t 的位置"""
    return M0 + vM * t

# 无人机 FY1 初态与航迹（等高度、朝 -x，速 120 m/s）
U0 = np.array([17800.0, 0.0, 1800.0], dtype=float)
uav_speed = 120.0
uav_dir = np.array([-1.0, 0.0, 0.0], dtype=float)  # 朝原点方向水平方向（-x）

# 投放/起爆
t_release = 1.5   # s
t_fuze    = 3.6   # s
t_explode = t_release + t_fuze  # = 5.1 s

# 投放点
P_release = U0 + uav_dir * uav_speed * t_release  # (17620, 0, 1800)

# 起爆点（水平匀速、竖直自由落体）
x_e = P_release[0] + uav_dir[0] * uav_speed * t_fuze
y_e = P_release[1] + uav_dir[1] * uav_speed * t_fuze
z_e = U0[2] - 0.5 * g * (t_fuze ** 2)
E = np.array([x_e, y_e, z_e], dtype=float)  # (17188, 0, 1736.496)

def C_pos(t: float) -> np.ndarray:
    """烟幕球心在时刻 t 的位置（仅竖直下沉，水平不动）"""
    return np.array([x_e, y_e, z_e - cloud_sink * (t - t_explode)], dtype=float)

# 目标：质点近似 & 离散化
T_center = np.array([0.0, 200.0, 5.0], dtype=float)  # 质点近似

def target_points_33() -> np.ndarray:
    """
    生成你建议的 33 个离散点：
    - 中心 1 个: (0,200,5)
    - 顶面 8 个: z=10, 半径7，等角分布
    - 侧面 16 个: z={2.5,7.5} 各 8 个，半径7，等角分布
    - 底面 8 个: z=0, 半径7，等角分布
    """
    pts = [T_center.copy()]
    rad = 7.0
    cx, cy = 0.0, 200.0

    def ring(z, n=8):
        th = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.vstack([cx + rad*np.cos(th), cy + rad*np.sin(th), np.full_like(th, z)]).T

    # 顶圈 z=10  (8)
    pts.append(ring(10.0, 8))
    # 侧面两圈 z=2.5, 7.5 (16)
    pts.append(ring(2.5, 8))
    pts.append(ring(7.5, 8))
    # 底圈 z=0   (8)
    pts.append(ring(0.0, 8))

    return np.vstack([p if p.ndim==2 else p[None,:] for p in pts])

# -----------------------------
# 2) 几何判据：点到线段的最近距离
# -----------------------------
def point_to_segment_distance(c: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    c 到 线段 ab 的最近距离 d 及 段参数 lambda*.
    返回: (d, lambda*, 最近点)
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0:
        # 退化：a==b
        return np.linalg.norm(c - a), 0.0, a.copy()
    lam = float(np.dot(c - a, ab) / denom)
    lam_clamped = max(0.0, min(1.0, lam))
    closest = a + lam_clamped * ab
    d = float(np.linalg.norm(c - closest))
    return d, lam, closest

# -----------------------------
# 3) 遮蔽判据与时长/区间求解
# -----------------------------
def coverage_mask(T: np.ndarray, t_grid: np.ndarray, R: float):
    """
    对给定目标点 T，在时间网格上返回：是否满足 (d(t)<=R 且 lam∈[0,1]) 的布尔掩码；
    同时返回 d(t) 与 lam(t) 数组用于可视化。
    """
    d_list, lam_list, inside = [], [], []
    for t in t_grid:
        c = C_pos(t)
        m = M_pos(t)
        d, lam, _ = point_to_segment_distance(c, m, T)
        ok = (d <= R) and (0.0 <= lam <= 1.0)
        d_list.append(d); lam_list.append(lam); inside.append(ok)
    return np.array(inside, dtype=bool), np.array(d_list), np.array(lam_list)

def find_intervals(mask: np.ndarray, t_grid: np.ndarray):
    """
    在布尔掩码上找 True 的连续区间 [t_start, t_end]（闭开近似）。
    """
    intervals = []
    in_block = False
    start_t = None
    for i, val in enumerate(mask):
        if val and not in_block:
            in_block = True
            start_t = t_grid[i]
        if (not val and in_block):
            in_block = False
            end_t = t_grid[i]
            intervals.append((start_t, end_t))
    if in_block:
        intervals.append((start_t, t_grid[-1]))
    return intervals

def refine_edge(T: np.ndarray, tL: float, tR: float, R: float, tol=1e-4, maxit=50, want_enter=True):
    """
    对边界进行二分法细化，使 d(t)=R 且 lam∈[0,1]（进入/离开边界）。
    want_enter=True 找进入点（外->内），False 找离开点（内->外）。
    假设区间内确有一次单调穿越。
    """
    def f(t):
        c = C_pos(t); m = M_pos(t)
        d, lam, _ = point_to_segment_distance(c, m, T)
        # 非法 lam 直接返回正值（表示在外面）
        if not (0.0 <= lam <= 1.0):
            return +1.0
        return d - R

    a, b = tL, tR
    fa, fb = f(a), f(b)
    # 如果极端情况下同号，直接返回端点（容错）
    if fa == 0: return a
    if fb == 0: return b
    # 若没有严格异号，尝试把 want_enter 作为指导修正一侧
    if fa * fb > 0:
        return (a + b) / 2.0

    for _ in range(maxit):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or (b - a) < tol:
            return mid
        # 保证二分
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)

def intervals_with_refine(T: np.ndarray, t0: float, t1: float, R: float, N=20001):
    """
    先网格判断，再对每个边界做二分细化，得到更精确的 [t_in, t_out] 列表及总时长。
    """
    t_grid = np.linspace(t0, t1, N)
    mask, d_arr, lam_arr = coverage_mask(T, t_grid, R)
    coarse = find_intervals(mask, t_grid)

    refined = []
    for (a, b) in coarse:
        # 在原掩码里找到 a, b 的邻近索引，确定二分搜索的小区间
        iL = np.searchsorted(t_grid, a, side='left')
        iR = np.searchsorted(t_grid, b, side='left')
        # 向左扩/向右扩一格用于包住过零点
        tL1 = t_grid[max(0, iL-1)]
        tL2 = t_grid[min(len(t_grid)-1, iL+1)]
        tR1 = t_grid[max(0, iR-1)]
        tR2 = t_grid[min(len(t_grid)-1, iR+1)]
        # 细化进入点（外->内）
        t_in = refine_edge(T, tL1, tL2, R, want_enter=True)
        # 细化离开点（内->外）
        t_out = refine_edge(T, tR1, tR2, R, want_enter=False)
        refined.append((t_in, t_out))
    total = sum(b-a for a, b in refined)
    return refined, total, (t_grid, d_arr, lam_arr, mask)

# -----------------------------
# 4) 主流程：质点近似 + 离散点对比
# -----------------------------
if __name__ == "__main__":
    print("=== 固定参数 ===")
    print(f"导弹速度 300 m/s，方向 {to_origin}")
    print(f"FY1: U0={U0}, 速度={uav_speed} m/s, 航向={uav_dir}")
    print(f"投放点 P_release={P_release}")
    print(f"起爆点 E={E}, t_explode={t_explode:.3f} s")
    print()

    # 有效窗口（20 s）
    t0, t1 = t_explode, t_explode + 20.0

    # ---- (A) 质点近似：T_center
    I_center, dur_center, aux = intervals_with_refine(T_center, t0, t1, R, N=20001)
    t_grid, d_arr, lam_arr, mask = aux

    print("=== 质点近似（目标中心 T_c） ===")
    if I_center:
        for k, (a, b) in enumerate(I_center, 1):
            print(f"区间{k}: [{a:.3f}, {b:.3f}] s, 时长={b-a:.3f} s")
    else:
        print("无有效遮蔽。")
    print(f"总时长 Δt = {dur_center:.3f} s")
    print()

    # 额外参考：轴上下端点的时长（z=0, z=10）
    I_low, d_low, _  = intervals_with_refine(np.array([0.0,200.0,0.0]),  t0, t1, R, N=20001)
    I_top, d_top, _  = intervals_with_refine(np.array([0.0,200.0,10.0]), t0, t1, R, N=20001)
    print("轴端点参考：")
    print(f"z=0   总时长 ≈ {d_low:.3f} s；z=10  总时长 ≈ {d_top:.3f} s")
    print()

    # ---- (B) 33点离散计算与分析
    do_discrete = True
    if do_discrete:
        Tset = target_points_33()
        durations = []
        print("\n=== 计算33个离散点的精确遮蔽时长 ===")
        # 为了节省时间，这里的网格可以粗一些，因为最终依赖二分法细化
        for i, Ti in enumerate(Tset):
            _, dur, _ = intervals_with_refine(Ti, t0, t1, R, N=2001)
            durations.append((Ti, dur))
            # 打印前5个和最后一个点的时长作为参考
            if i < 5 or i == len(Tset) - 1:
                print(f"点 {i+1:>2d} {np.round(Ti, 2)} -> 时长: {dur:.3f} s")

        # 保存到文件
        output_filename = "p1_discrete_durations.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("点索引, X, Y, Z, 遮蔽时长(s)\n")
            # 写入所有33个点
            for i, (point, dur) in enumerate(durations):
                f.write(f"{i+1}, {point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}, {dur:.4f}\n")
        print(f"\n所有33个点的遮蔽时长已保存到: {output_filename}")
        print()

        # ---- (C) 离散点分析（沿用之前逻辑）
        # 逐点的掩码
        masks = []
        for Ti in Tset:
            mi, _, _ = coverage_mask(Ti, t_grid, R)
            masks.append(mi)
        masks = np.vstack(masks)  # [33, Nt]
        frac = masks.mean(axis=0)  # 每个时刻被遮蔽点比例

        # 交集（全部遮蔽）：
        all_mask = masks.all(axis=0)
        I_all = find_intervals(all_mask, t_grid)
        dur_all = sum(b-a for a,b in I_all)

        # 覆盖率阈值（例如 80%）：
        p = 0.8
        thr_mask = (frac >= p)
        I_thr = find_intervals(thr_mask, t_grid)
        dur_thr = sum(b-a for a,b in I_thr)

        print("=== 33点离散对比分析 ===")
        print(f"全部遮蔽(交集) 总时长 ≈ {dur_all:.3f} s，区间：{I_all}")
        print(f"覆盖率≥{int(p*100)}% 总时长 ≈ {dur_thr:.3f} s，区间：{I_thr}")
        print("（一般会看到与质点近似的结果非常接近，差在几十毫秒量级）")
        print()

    # -----------------------------
    # 5) 可视化
    # -----------------------------

    # (V1) 时间域判据图：d(t) vs R，叠加 lam*(t) 与有效遮蔽阴影
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(t_grid, d_arr, label="云团中心到视线段距离 d(t)")
    ax1.axhline(R, linestyle="--", color="r", label="有效遮蔽半径 R = 10 m")
    # 有效遮蔽阴影
    for (a, b) in I_center:
        ax1.axvspan(a, b, color="orange", alpha=0.2, label="有效遮蔽" if a==I_center[0][0] else None)
    ax1.set_xlabel("时间 t (s)")
    ax1.set_ylabel("距离 d(t) (m)")
    ax1.set_title("遮蔽条件随时间变化（质点目标 T_c）")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 第二纵轴画 lambda*(t)，并高亮 [0,1]
    ax2 = ax1.twinx()
    ax2.plot(t_grid, np.clip(lam_arr, -0.1, 1.1), "g-.", alpha=0.6, label="视线段参数 λ*(t)")
    ax2.axhspan(0,1, color="grey", alpha=0.1, label="0 ≤ λ* ≤ 1")
    ax2.set_ylabel("λ*(t)")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=9)
    fig1.tight_layout()

    # (V2) 目标三维表面与离散点
    if do_discrete:
        fig2 = plt.figure(figsize=(8, 8))
        ax3d = fig2.add_subplot(111, projection='3d')

        # 绘制圆柱体表面
        rad = 7.0
        cx, cy = 0.0, 200.0
        z_bottom, z_top = 0.0, 10.0
        theta = np.linspace(0, 2 * np.pi, 100)
        z_wall = np.linspace(z_bottom, z_top, 2)
        theta_grid, z_grid = np.meshgrid(theta, z_wall)
        x_grid = cx + rad * np.cos(theta_grid)
        y_grid = cy + rad * np.sin(theta_grid)
        ax3d.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='c', rstride=5, cstride=5, linewidth=0.1, edgecolors='k')

        # 绘制顶盖和底盖
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        x_circle = cx + rad * np.cos(theta_circle)
        y_circle = cy + rad * np.sin(theta_circle)
        verts_top = [list(zip(x_circle, y_circle, np.full_like(x_circle, z_top)))]
        verts_bottom = [list(zip(x_circle, y_circle, np.full_like(x_circle, z_bottom)))]
        ax3d.add_collection3d(Poly3DCollection(verts_top, facecolors='cyan', alpha=0.3))
        ax3d.add_collection3d(Poly3DCollection(verts_bottom, facecolors='cyan', alpha=0.3))

        # 绘制离散点
        Tset = target_points_33()
        ax3d.scatter(Tset[:,0], Tset[:,1], Tset[:,2], c='r', s=25, label='离散采样点', depthshade=True)

        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('目标圆柱体表面及离散采样点')
        ax3d.legend()
        
        # 设置一个好看的视角和坐标轴比例
        ax3d.view_init(elev=20, azim=-120)
        ax3d.set_box_aspect((np.ptp(Tset[:,0]), np.ptp(Tset[:,1]), np.ptp(Tset[:,2]))) # 视觉比例
        fig2.tight_layout()

    # 打印关键结论
    print("=== 结论（问题 1） ===")
    if I_center:
        a, b = I_center[0]
        print(f"有效遮蔽区间 ≈ [{a:.3f}, {b:.3f}] s，时长 Δt ≈ {b-a:.3f} s")
    else:
        print("无有效遮蔽。")

    plt.show()
