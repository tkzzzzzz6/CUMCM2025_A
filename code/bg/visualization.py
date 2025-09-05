import os
import numpy as np
import matplotlib.pyplot as plt

# Data
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

# Group scatter by category so the legend stays short and meaningful
miss_xy = np.vstack([v[:2] for v in missiles.values()])
uav_xy  = np.vstack([v[:2] for v in uavs.values()])

ax.scatter(miss_xy[:, 0], miss_xy[:, 1], marker='^', s=70, label='Missiles', color='C0')
ax.scatter(uav_xy[:, 0],  uav_xy[:, 1],  marker='o', s=50, label='UAVs',     color='C1')

# Name + z near each point; slight offset to reduce overlap
for name, pos in missiles.items():
    ax.annotate(f"{name} z={pos[2]:.0f} m", (pos[0], pos[1]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, alpha=0.9)
for name, pos in uavs.items():
    ax.annotate(f"{name} z={pos[2]:.0f} m", (pos[0], pos[1]),
                textcoords="offset points", xytext=(6, 6), fontsize=9, alpha=0.9)

# Direction arrows (missiles -> decoy/origin)
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

# Decoy
ax.scatter(decoy_xy[0], decoy_xy[1], marker='x', s=100, label='Decoy (0,0)', color='C2')
ax.annotate("Decoy (0,0,0)", (decoy_xy[0], decoy_xy[1]),
            textcoords="offset points", xytext=(10, 10), color="#5e83b3")

# Real target footprint circle (top-down)
theta = np.linspace(0, 2*np.pi, 256)
xc = real_center_xy[0] + real_r * np.cos(theta)
yc = real_center_xy[1] + real_r * np.sin(theta)
ax.plot(xc, yc, linestyle='-', label='Real Target footprint', color='C3', lw=1.5)
ax.annotate(f"Real Target: center (0,200)\nr={real_r} m, h={real_h} m",
            (real_center_xy[0], real_center_xy[1]),
            textcoords="offset points", xytext=(12, 12), color="#5e83b3")

# Axes & aspect
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Top-Down XY View: Missiles, UAVs, Decoy and Real Target Footprint')
ax.set_aspect('equal', adjustable='box')
ax.ticklabel_format(style='plain')
ax.grid(True, color='0.85')

# Limits with margins
xy_all = np.vstack([
    miss_xy, uav_xy, decoy_xy.reshape(1, -1), real_center_xy.reshape(1, -1)
])
xmin, ymin = xy_all.min(axis=0) - 1200.0
xmax, ymax = xy_all.max(axis=0) + 1200.0
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Legend below X-axis, horizontally
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)
plt.subplots_adjust(bottom=0.25)  # leave room for the legend

# Save
save_path = './output/visualization/visualization.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close(fig)