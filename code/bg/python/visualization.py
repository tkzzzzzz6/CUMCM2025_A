import os
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 数据
missiles = {
    "M1": np.array([20000.0,    0.0, 2000.0]),
    "M2": np.array([19000.0,  600.0, 2100.0]),
    "M3": np.array([18000.0, -600.0, 1900.0]),
}
uavs = {
    "FY1": np.array([17800.0,     0.0, 1800.0]),
    "FY2": np.array([12000.0,  1400.0, 1400.0]),
    "FY3": np.array([ 6000.0, -3000.0,  700.0]),
    "FY4": np.array([11000.0,  2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}
decoy_xy = np.array([0.0, 0.0])
decoy_z = 0.0
real_center_xy = np.array([0.0, 200.0])
real_r = 7.0
real_h = 10.0

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# 分类散点，便于图例简洁
miss_xy = np.vstack([v[:2] for v in missiles.values()])
uav_xy  = np.vstack([v[:2] for v in uavs.values()])

# 导弹：红色三角
ax.scatter(miss_xy[:, 0], miss_xy[:, 1], marker='^', s=70, label='导弹', color='red')
# 无人机：蓝色圆点
ax.scatter(uav_xy[:, 0],  uav_xy[:, 1],  marker='o', s=50, label='无人机', color='blue')

# 每个点旁边标注名称和高度z，适当偏移避免重叠
for name, pos in missiles.items():
    ax.annotate(f"{name} 高度={pos[2]:.0f}米", (pos[0], pos[1]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, alpha=0.9, color='red')
for name, pos in uavs.items():
    ax.annotate(f"{name} 高度={pos[2]:.0f}米", (pos[0], pos[1]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, alpha=0.9, color='blue')

# 方向箭头（导弹指向诱饵/原点）
def arrow_to_origin(xy, L=1000.0):
    v = -np.array(xy, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return (xy[0], xy[1]), (xy[0], xy[1])
    v = v / n
    L_eff = min(L, 0.9 * n)
    end = (xy[0] + v[0] * L_eff, xy[1] + v[1] * L_eff)
    return (xy[0], xy[1]), end

for pos in missiles.values():
    (x0, y0), (x1, y1) = arrow_to_origin(pos[:2], L=1000.0)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2), zorder=2)

# 诱饵
ax.scatter(decoy_xy[0], decoy_xy[1], marker='x', s=100, label='诱饵 (0,0)', color='green')
ax.annotate("诱饵 (0,0,0)", (decoy_xy[0], decoy_xy[1]),
            textcoords="offset points", xytext=(10, 10), color="green")

# 真目标投影圆（俯视）
theta = np.linspace(0, 2*np.pi, 256)
xc = real_center_xy[0] + real_r * np.cos(theta)
yc = real_center_xy[1] + real_r * np.sin(theta)
# 真目标：橙色圆圈
ax.plot(xc, yc, linestyle='-', label='真目标投影', color='orange', lw=1.5)
# 真目标中心点：橙色点
ax.scatter(real_center_xy[0], real_center_xy[1], marker='o', s=80, color='orange', label='真目标中心')
ax.annotate(f"真目标中心 (0,200)\n半径={real_r}米, 高度={real_h}米",
            (real_center_xy[0], real_center_xy[1]),
            textcoords="offset points", xytext=(12, 12), color="orange")

# 坐标轴与比例
ax.set_xlabel('X (米)')
ax.set_ylabel('Y (米)')
ax.set_title('XY俯视图：导弹、无人机、诱饵与真目标')
ax.set_aspect('equal', adjustable='box')
ax.ticklabel_format(style='plain')
ax.grid(True, color='0.85')

# 边界留白
xy_all = np.vstack([
    miss_xy, uav_xy, decoy_xy.reshape(1, -1), real_center_xy.reshape(1, -1)
])
xmin, ymin = xy_all.min(axis=0) - 1200.0
xmax, ymax = xy_all.max(axis=0) + 1200.0
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# 图例放在下方，横向排列
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)
plt.subplots_adjust(bottom=0.25)  # 留出图例空间

# 保存
save_path = './output/visualization/visualization.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close(fig)