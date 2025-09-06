#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Problem 5 quick feasibility check: Is there any UAV that cannot interfere with any missile?

Assumptions from the problem statement:
- Missiles fly straight toward the origin at 300 m/s.
- UAVs instantly set heading, then fly straight at constant altitude with speed in [70, 140] m/s.
- A smoke munition is released, free-falls under gravity; after a chosen fuse time it detonates.
- At detonation, a spherical cloud forms instantly at the detonation point and then descends at 3 m/s.
- The cloud provides effective obscuration within 10 m radius of its center for 20 s after detonation.

Feasibility criterion used here (sufficient and enforces horizontal reachability):
- We set the line-of-sight (LOS) block at the exact detonation time (within the 20 s window), t_LOS = t_det = T.
- The detonation point must lie on the LOS segment between the origin and the missile at time T.
- The UAV flies at constant speed v ∈ [70, 140] m/s along a straight heading from t=0. The detonation horizontal
  location equals the UAV's horizontal position at time T (release timing cancels out in horizontal motion).
- Therefore, horizontal reachability requires that the distance r between the UAV's initial xy and the chosen LOS
  xy point at time T satisfies r = v T for some v ∈ [70, 140].
- Vertical/time feasibility requires the detonation altitude z_det to be achievable by free-fall with zero initial
  vertical speed in fuse time f, i.e., z_det = z_uav - 0.5 g f^2, with 0 ≤ f ≤ T and 0 ≤ z_det ≤ z_missile(T).

If there exists T ∈ (0, t_impact] and a point on LOS at altitude z_det that satisfy both horizontal and vertical
constraints, the UAV can interfere with that missile. If a UAV can interfere with any missile, it is considered
feasible.

The script prints per UAV whether it can interfere at least one missile and summarizes if any UAV cannot.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


Vector3 = Tuple[float, float, float]


def norm3(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def scale3(v: Vector3, s: float) -> Vector3:
    return (v[0] * s, v[1] * s, v[2] * s)


def add3(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def missile_position_over_time(m0: Vector3, speed: float, t: float) -> Vector3:
    """Position of a missile flying toward the origin at time t."""
    direction_to_origin = scale3(m0, -1.0 / max(norm3(m0), 1e-9))
    return add3(m0, scale3(direction_to_origin, speed * t))


def missile_time_to_impact(m0: Vector3, speed: float) -> float:
    return norm3(m0) / speed


def can_uav_interfere_missile(uav_pos: Vector3, missile_pos0: Vector3, missile_speed: float, g: float,
                              v_min: float = 70.0, v_max: float = 140.0) -> bool:
    """Check if a single UAV can interfere with a single missile with horizontal and vertical constraints."""
    z_uav = uav_pos[2]
    if z_uav <= 0.0:
        return False

    t_impact = missile_time_to_impact(missile_pos0, missile_speed)

    # Coarse sampling over time and LOS height fraction
    time_samples = 200
    height_samples = 120

    ux, uy = uav_pos[0], uav_pos[1]

    for i in range(1, time_samples + 1):
        T = t_impact * i / time_samples  # in (0, t_impact]
        m_t = missile_position_over_time(missile_pos0, missile_speed, T)
        mx, my, mz = m_t
        if mz <= 0.0:
            continue

        # Upper bound for detonation altitude due to missile and UAV altitude
        z_upper = min(mz, z_uav)
        if z_upper <= 0.0:
            continue

        # Sample along LOS by altitude fraction lambda in [0, z_upper/mz]
        lam_upper = z_upper / mz
        for k in range(1, height_samples + 1):
            lam = lam_upper * k / height_samples
            z_det = lam * mz

            # Vertical feasibility (free-fall)
            f = math.sqrt(max(0.0, 2.0 * (z_uav - z_det) / g))
            if f > T:
                continue

            # Horizontal feasibility at time T: distance equals v * T for some v in [v_min, v_max]
            x_los = lam * mx
            y_los = lam * my
            r = math.hypot(x_los - ux, y_los - uy)
            if r <= 0.0:
                required_speed = 0.0
            else:
                required_speed = r / T

            if v_min - 1e-6 <= required_speed <= v_max + 1e-6:
                return True

    return False


def main() -> None:
    g = 9.8
    missile_speed = 300.0

    # Initial positions (x, y, z)
    missiles: Dict[str, Vector3] = {
        "M1": (20000.0, 0.0, 2000.0),
        "M2": (19000.0, 600.0, 2100.0),
        "M3": (18000.0, -600.0, 1900.0),
    }

    uavs: Dict[str, Vector3] = {
        "FY1": (17800.0, 0.0, 1800.0),
        "FY2": (12000.0, 1400.0, 1400.0),
        "FY3": (6000.0, -3000.0, 700.0),
        "FY4": (11000.0, 2000.0, 1800.0),
        "FY5": (13000.0, -2000.0, 1300.0),
    }

    results: Dict[str, bool] = {}
    for uav_name, uav_pos in uavs.items():
        feasible_any = False
        for m_name, m_pos in missiles.items():
            if can_uav_interfere_missile(uav_pos, m_pos, missile_speed, g):
                feasible_any = True
                break
        results[uav_name] = feasible_any

    print("Feasibility of UAVs interfering at least one missile (True = can interfere):")
    for uav_name in sorted(results.keys()):
        print(f"  {uav_name}: {results[uav_name]}")

    cannot_list: List[str] = [name for name, ok in results.items() if not ok]
    if cannot_list:
        print("\nUAVs that cannot interfere with any missile:")
        for name in cannot_list:
            print(f"  - {name}")
    else:
        print("\nAll UAVs can interfere with at least one missile under the modeled constraints.")


if __name__ == "__main__":
    main()


