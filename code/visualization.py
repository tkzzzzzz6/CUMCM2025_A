# Generate ONLY the top-down XY visualization with specific requirements:
# - Legend placed horizontally below the X-axis
# - Annotate z-values near missile and UAV points
# - Annotate decoy and real target parameters using color "#5e83b3"
# - Use matplotlib only; one figure; no seaborn; default colors except where the user requested

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

# Figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Plot missiles
for name, pos in missiles.items():
    ax.scatter(pos[0], pos[1], marker='^', s=70, label=name)
    ax.annotate(f"z={pos[2]:.0f} m", (pos[0], pos[1]), textcoords="offset points", xytext=(6, 6))

# Plot UAVs
for name, pos in uavs.items():
    ax.scatter(pos[0], pos[1], marker='o', s=50, label=name)
    ax.annotate(f"z={pos[2]:.0f} m", (pos[0], pos[1]), textcoords="offset points", xytext=(6, 6))

# Decoy
ax.scatter(decoy_xy[0], decoy_xy[1], marker='x', s=100, label='Decoy (0,0)')
ax.annotate("Decoy (0,0,0)", (decoy_xy[0], decoy_xy[1]), textcoords="offset points",
            xytext=(10, 10), color="#5e83b3")

# Real target footprint circle
theta = np.linspace(0, 2*np.pi, 256)
xc = real_center_xy[0] + real_r * np.cos(theta)
yc = real_center_xy[1] + real_r * np.sin(theta)
(line_real,) = ax.plot(xc, yc, linestyle='-', label='Real Target footprint')
ax.annotate(f"Real Target: center (0,200)\nr={real_r} m, h={real_h} m",
            (real_center_xy[0], real_center_xy[1]), textcoords="offset points",
            xytext=(12, 12), color="#5e83b3")

# Axes & aspect
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Top-Down XY View: Missiles, UAVs, Decoy and Real Target Footprint')
ax.set_aspect('equal', adjustable='box')

# Limits with margins
xy_all = np.array([[*p[:2]] for p in missiles.values()] + [[*p[:2]] for p in uavs.values()] + [decoy_xy, real_center_xy])
xmin, ymin = xy_all.min(axis=0) - 1200.0
xmax, ymax = xy_all.max(axis=0) + 1200.0
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Legend below X-axis, horizontally expanded
handles, labels = ax.get_legend_handles_labels()
# ensure unique
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(),
          loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=len(uniq), frameon=False)

fig.tight_layout()
save_path = './output/visualization/visualization.png'
fig.savefig(save_path, dpi=200)


