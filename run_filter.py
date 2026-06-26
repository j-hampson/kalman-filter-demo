"""
run_filter.py

Applies Kalman filter to simulated trajectory.
"""

import numpy as np
from kalman import KalmanFilter2D
from simulate import generate_trajectory


def run(
    dt: float = 0.1,
    total_time: float = 40.0,
    process_noise_std: float = 1.5,
    measurement_noise_std: float = 8.0,
    seed: int = 42,
) -> dict:
    """
    Run simulation and Kalman filter.

    Returns:
    dict {
        times            : 1-D time stamps
        true_pos         : (N, 2) ground truth
        measurements     : (N, 2) noisy GPS
        filtered_pos     : (N, 2) Kalman estimates
        filtered_vel     : (N, 2) velocity estimates
        pos_covariances  : (N, 2, 2) position covariance at each step
        innovations      : (N, 2) residuals at each step
        rmse_gps         : scalar RMSE of raw GPS
        rmse_kf          : scalar RMSE of Kalman filter
        }
    """
    # Simulate
    sim = generate_trajectory(
        dt=dt,
        total_time=total_time,
        measurement_noise_std=measurement_noise_std,
        seed=seed,
    )
    times        = sim["times"]
    true_pos     = sim["true_pos"]
    measurements = sim["measurements"]

    # Initialize filter 
    kf = KalmanFilter2D(
        dt=dt,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )
    kf.initialize(
        px=measurements[0, 0],
        py=measurements[0, 1],
    )

    # Run filter 
    N = len(times)
    filtered_pos    = np.zeros((N, 2))
    filtered_vel    = np.zeros((N, 2))
    pos_covariances = np.zeros((N, 2, 2))
    innovations     = np.zeros((N, 2))

    for i in range(N):
        kf.predict()
        innov = kf.update(measurements[i])

        filtered_pos[i]    = kf.position
        filtered_vel[i]    = kf.velocity
        pos_covariances[i] = kf.position_covariance
        innovations[i]     = innov

    # Compute RMSE 
    def rmse(est, truth):
        return np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1)))

    rmse_gps = rmse(measurements, true_pos)
    rmse_kf  = rmse(filtered_pos, true_pos)

    print(f"GPS RMSE  : {rmse_gps:.2f} m")
    print(f"KF  RMSE  : {rmse_kf:.2f} m")
    print(f"Improvement: {rmse_gps / rmse_kf:.1f}×")

    return {
        "times":            times,
        "true_pos":         true_pos,
        "measurements":     measurements,
        "filtered_pos":     filtered_pos,
        "filtered_vel":     filtered_vel,
        "pos_covariances":  pos_covariances,
        "innovations":      innovations,
        "rmse_gps":         rmse_gps,
        "rmse_kf":          rmse_kf,
    }


if __name__ == "__main__":
    results = run()
