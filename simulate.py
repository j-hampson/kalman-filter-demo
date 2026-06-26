"""
simulate.py

Generates ground-truth trajectory and noisy GPS measurements.
"""

import numpy as np


def generate_trajectory(
    dt: float = 0.1,
    total_time: float = 40.0,
    measurement_noise_std: float = 8.0,
    seed: int = 42,
) -> dict:
    """
    Simulate a ground-truth path and add Gaussian GPS noise.

    Parameters:
    dt (float) - Time step in seconds.
    total_time (float) - Total simulation duration in seconds.
    measurement_noise_std (float) - Standard deviation of GPS noise in metres.
    seed (int) - Random seed.

    Returns:
    dict {
        times        : 1-D array of time stamps
        true_pos     : (N, 2) true positions [px, py]
        true_vel     : (N, 2) true velocities [vx, vy]
        measurements : (N, 2) noisy GPS positions
        }
    """
    rng = np.random.default_rng(seed)
    times = np.arange(0, total_time, dt)
    N = len(times)

    # Parametric curve trajectory

    speed = 12.0          # m/s 
    heading = np.zeros(N) # radians, 0 is East

    # Segment schedule (start_time, end_time, turn_rate_deg_per_s)
    segments = [
        (0,   8,   0),      # straight NE
        (8,   18,  -6),     # sweeping left turn
        (18,  24,  0),      # straight south
        (24,  34,  6),      # sweeping right turn
        (34,  40,  0),      # straight west
    ]

    for start, end, turn_rate_dps in segments:
        mask = (times >= start) & (times < end)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        h0 = heading[idxs[0]]
        for i, idx in enumerate(idxs):
            heading[idx] = np.radians(np.degrees(h0) + turn_rate_dps * i * dt)
        # Carry forward heading to the next segment
        if idxs[-1] + 1 < N:
            heading[idxs[-1] + 1 :] = heading[idxs[-1]]

    # Rotate initial heading to north-east (45°)
    heading += np.radians(45)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    px = np.zeros(N)
    py = np.zeros(N)
    for i in range(1, N):
        px[i] = px[i - 1] + vx[i - 1] * dt
        py[i] = py[i - 1] + vy[i - 1] * dt

    # Add Gaussian noise
    noise = rng.normal(0, measurement_noise_std, size=(N, 2))
    measurements = np.column_stack([px, py]) + noise

    return {
        "times":        times,
        "true_pos":     np.column_stack([px, py]),
        "true_vel":     np.column_stack([vx, vy]),
        "measurements": measurements,
    }
