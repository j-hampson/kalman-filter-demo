"""
kalman.py

2D Kalman filter simulation.
"""

import numpy as np


class KalmanFilter2D:
    """
    Linear Kalman filter for tracking a 2D object with constant-velocity motion.

    Parameters:
    dt (float) - Time step between measurements (seconds).
    process_noise_std (float) - Standard deviation of process noise.
    measurement_noise_std (float) - Standard deviation of GPS measurement noise.
    """

    def __init__(self, dt: float, process_noise_std: float, measurement_noise_std: float):
        self.dt = dt
        n_states = 4   # [px, py, vx, vy]
        n_meas   = 2   # [px, py]

        # State transition matrix F 
        # Encodes new_pos = old_pos + vel * dt,  new_vel = old_vel
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        # Measurement matrix H 
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance Q 
        q = process_noise_std ** 2
        self.Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ],
        ])

        # Measurement noise covariance R
        r = measurement_noise_std ** 2
        self.R = r * np.eye(n_meas)

        # Initial state and covariance
        self.x = np.zeros((n_states, 1))        # state estimate
        self.P = np.eye(n_states) * 500.0       # initial uncertainty
        self.I = np.eye(n_states)   

    def initialize(self, px: float, py: float, vx: float = 0.0, vy: float = 0.0):
        """Set initial state from the first measurement."""
        self.x = np.array([[px], [py], [vx], [vy]], dtype=float)

    def predict(self):
        """
        Project state and covariance forward one time step.

        xhat_(k-) = F * xhat_(k-1)
        P_(k-)  = F * P_(k-1) * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        """
        Incorporate new GPS measurement.

        Parameters:
        measurement (array) - Observed [px, py] from the GPS sensor.

        Returns innovation for diagnostics.
        """
        z = np.array(measurement, dtype=float).reshape(2, 1)

        # Innovation/residual
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain  K = P * H^T * S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return y.flatten()

    @property
    def position(self) -> np.ndarray:
        """Current position estimate [px, py]."""
        return self.x[:2].flatten()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vx, vy]."""
        return self.x[2:].flatten()

    @property
    def position_covariance(self) -> np.ndarray:
        """2×2 position sub-block of the covariance matrix."""
        return self.P[:2, :2]
