"""
app.py

Streamlit web app for 2D Kalman Filter GPS tracking demo.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import streamlit as st

from run_filter import run

# Page config 
st.set_page_config(
    page_title="Kalman Filter Demo",
    page_icon="📡",
    layout="wide",
)

# Colors
BLUE  = "#2563EB"
CORAL = "#EF4444"
GREEN = "#16A34A"
AMBER = "#D97706"
GRAY  = "#6B7280"
LIGHT = "#F3F4F6"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        130,
})


def covariance_ellipse(cov, mean, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    return Ellipse(xy=mean, width=w, height=h, angle=angle)


# Plot functions 

def fig_trajectory(results):
    fig, ax = plt.subplots(figsize=(7, 6))
    true_pos, meas, filt = results["true_pos"], results["measurements"], results["filtered_pos"]

    ax.scatter(meas[:, 0], meas[:, 1], s=4, alpha=0.35, color=CORAL, label="GPS measurements", zorder=2)
    ax.plot(true_pos[:, 0], true_pos[:, 1], color=GRAY, lw=1.5, ls="--", label="True path", zorder=3)
    ax.plot(filt[:, 0], filt[:, 1], color=BLUE, lw=2.2, label="Kalman estimate", zorder=4)

    rmse_gps, rmse_kf = results["rmse_gps"], results["rmse_kf"]
    ax.text(0.02, 0.97,
            f"GPS RMSE : {rmse_gps:.1f} m\nKF  RMSE : {rmse_kf:.1f} m\nImprovement: {rmse_gps/rmse_kf:.1f}×",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="#D1D5DB", alpha=0.9))

    ax.set_title("Trajectory: True Path vs GPS vs Kalman Filter", fontsize=12, pad=10)
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def fig_error(results):
    times, true_pos = results["times"], results["true_pos"]
    meas, filt = results["measurements"], results["filtered_pos"]

    err_gps = np.linalg.norm(meas - true_pos, axis=1)
    err_kf  = np.linalg.norm(filt - true_pos, axis=1)
    window  = 20
    roll    = lambda a: np.convolve(a, np.ones(window) / window, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].fill_between(times, err_gps, alpha=0.15, color=CORAL)
    axes[0].plot(times, err_gps,       color=CORAL, lw=0.8, alpha=0.5)
    axes[0].plot(times, roll(err_gps), color=CORAL, lw=2,   label="GPS error (smoothed)")
    axes[0].fill_between(times, err_kf, alpha=0.2, color=BLUE)
    axes[0].plot(times, err_kf,       color=BLUE,  lw=0.8, alpha=0.5)
    axes[0].plot(times, roll(err_kf),  color=BLUE,  lw=2,   label="KF error (smoothed)")
    axes[0].set_ylabel("Position Error (m)")
    axes[0].set_title("Position Error Over Time", fontsize=12)
    axes[0].legend(fontsize=9)

    cum_gps = np.sqrt(np.cumsum(err_gps**2) / (np.arange(len(times)) + 1))
    cum_kf  = np.sqrt(np.cumsum(err_kf**2)  / (np.arange(len(times)) + 1))
    axes[1].plot(times, cum_gps, color=CORAL, lw=2, label="GPS cumulative RMSE")
    axes[1].plot(times, cum_kf,  color=BLUE,  lw=2, label="KF cumulative RMSE")
    axes[1].fill_between(times, cum_kf, cum_gps, alpha=0.12, color=GREEN, label="Improvement")
    axes[1].set_ylabel("Cumulative RMSE (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def fig_ellipses(results, sample_every=30):
    fig, ax = plt.subplots(figsize=(7, 6))
    true_pos = results["true_pos"]
    filt     = results["filtered_pos"]
    meas     = results["measurements"]
    covs     = results["pos_covariances"]
    times    = results["times"]

    ax.plot(true_pos[:, 0], true_pos[:, 1], color=GRAY, lw=1.2, ls="--", label="True path", alpha=0.6, zorder=2)
    ax.plot(filt[:, 0],     filt[:, 1],     color=BLUE, lw=2,              label="Kalman estimate",    zorder=3)
    ax.scatter(meas[:, 0],  meas[:, 1],     s=3, alpha=0.2, color=CORAL,   label="GPS measurements",                zorder=1)

    cmap    = plt.cm.Blues
    indices = list(range(0, len(times), sample_every))
    n       = len(indices)
    for j, i in enumerate(indices):
        ell = covariance_ellipse(covs[i], filt[i], n_std=2)
        ell.set_facecolor(cmap(0.3 + 0.5 * j / max(n - 1, 1)))
        ell.set_edgecolor(BLUE)
        ell.set_linewidth(0.8)
        ell.set_alpha(0.25 + 0.55 * j / max(n - 1, 1))
        ax.add_patch(ell)

    label_every = max(1, n // 4)
    for j, i in enumerate(indices):
        if j % label_every == 0:
            ax.text(filt[i, 0] + 3, filt[i, 1] + 3, f"t={times[i]:.0f}s",
                    fontsize=7, color=BLUE, alpha=0.8)

    ell_patch = mpatches.Patch(facecolor=cmap(0.5), edgecolor=BLUE,
                               alpha=0.5, label="2σ uncertainty ellipse")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [ell_patch], fontsize=9, loc="lower right")

    ax.set_title("Uncertainty Ellipses (2σ) Over Time", fontsize=12, pad=10)
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def fig_velocity(results):
    times    = results["times"]
    true_vel = results.get("true_vel")
    filt_vel = results["filtered_vel"]

    true_speed   = np.linalg.norm(true_vel, axis=1) if true_vel is not None else None
    kf_speed     = np.linalg.norm(filt_vel, axis=1)
    true_heading = np.degrees(np.arctan2(true_vel[:, 1], true_vel[:, 0])) if true_vel is not None else None
    kf_heading   = np.degrees(np.arctan2(filt_vel[:, 1], filt_vel[:, 0]))

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    if true_speed is not None:
        axes[0].plot(times, true_speed, color=GRAY, lw=1.5, ls="--", label="True speed", alpha=0.7)
    axes[0].plot(times, kf_speed, color=BLUE, lw=2, label="KF speed estimate")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Kalman Filter Velocity Estimates", fontsize=12)
    axes[0].legend(fontsize=9)

    if true_heading is not None:
        axes[1].plot(times, true_heading, color=GRAY, lw=1.5, ls="--", label="True heading", alpha=0.7)
    axes[1].plot(times, kf_heading, color=BLUE, lw=2, label="KF heading estimate")
    axes[1].set_ylabel("Heading (° From East)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def fig_innovations(results):
    times = results["times"]
    inns  = results["innovations"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for k, (ax, label) in enumerate(zip(axes, ["x (East)", "y (North)"])):
        inn = inns[:, k]
        std = inn.std()
        ax.plot(times, inn, color=BLUE, lw=0.8, alpha=0.6)
        ax.axhline( 2 * std, color=CORAL, lw=1.2, ls="--", alpha=0.7, label=f"±2σ = ±{2*std:.1f} m")
        ax.axhline(-2 * std, color=CORAL, lw=1.2, ls="--", alpha=0.7)
        ax.axhline(0, color=GRAY, lw=0.7, alpha=0.5)
        ax.set_ylabel(f"Innovation {label} (m)")
        ax.legend(fontsize=9)

    axes[0].set_title("Filter Innovations",
                      fontsize=11)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def fig_tuning(meas_noise_std, seed):
    """Run filter at three Q values and overlay the paths."""
    configs = [
        ("Low Q",     0.1,  GRAY),
        ("Optimal Q", 1.5,  BLUE),
        ("High Q",    10.0,  CORAL),
    ]

    base = run(process_noise_std=1.5, measurement_noise_std=meas_noise_std, seed=seed)
    true_pos = base["true_pos"]
    meas     = base["measurements"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(meas[:, 0], meas[:, 1], s=3, alpha=0.2, color=GRAY, zorder=1, label="GPS measurements")
    ax.plot(true_pos[:, 0], true_pos[:, 1], color="black", lw=1.5, ls=":", alpha=0.5,
            label="True path", zorder=2)

    for label, q, color in configs:
        r = run(process_noise_std=q, measurement_noise_std=meas_noise_std, seed=seed)
        ax.plot(r["filtered_pos"][:, 0], r["filtered_pos"][:, 1], lw=2.2, color=color,
                label=f"{label}  (RMSE={r['rmse_kf']:.1f} m)", zorder=3)

    ax.set_title("Effect of Process Noise on Filter Behaviour", fontsize=12)
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# App layout 

st.title("2D Kalman Filter — GPS Tracking Demo")
st.caption("Simulate a vehicle path, add GPS noise, and watch the Kalman filter recover the true trajectory.")

# Sidebar
with st.sidebar:
    st.header("Filter parameters")

    st.subheader("Noise tuning")
    process_noise = st.slider(
        "Process noise std (Q)",
        min_value=0.1, max_value=15.0, value=1.5, step=0.1,
        help="How much we trust the motion model.",
    )
    
    meas_noise = st.slider(
        "Measurement noise std (R)",
        min_value=1.0, max_value=25.0, value=8.0, step=0.5,
        help="Standard deviation of GPS error in meters.",
    )

    st.subheader("Simulation")
    total_time = st.slider(
        "Simulation duration (s)",
        min_value=10, max_value=80, value=40, step=5,
    )
    
    dt = st.select_slider(
        "Time step dt (s)",
        options=[0.05, 0.1, 0.2, 0.5],
        value=0.1,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0, max_value=9999, value=50, step=1,
    )

    st.divider()
    run_button = st.button("Run filter", type="primary", width='stretch')

# Run app
run_key = (process_noise, meas_noise, total_time, dt, seed)

if "last_run_key" not in st.session_state or run_button:
    with st.spinner("Running simulation and filter…"):
        st.session_state.results     = run(
            dt=dt,
            total_time=total_time,
            process_noise_std=process_noise,
            measurement_noise_std=meas_noise,
            seed=int(seed),
        )
        st.session_state.last_run_key = run_key
        st.session_state.params = dict(
            process_noise=process_noise,
            meas_noise=meas_noise,
            total_time=total_time,
            dt=dt,
            seed=int(seed),
        )

results = st.session_state.results
params  = st.session_state.params

# Top metrics
rmse_gps = results["rmse_gps"]
rmse_kf  = results["rmse_kf"]
improvement = rmse_gps / rmse_kf

col1, col2, col3, col4 = st.columns(4)
col1.metric("GPS RMSE",   f"{rmse_gps:.2f} m")
col2.metric("KF RMSE",    f"{rmse_kf:.2f} m",  delta=f"−{rmse_gps - rmse_kf:.2f} m", delta_color="normal")
col3.metric("Improvement", f"{improvement:.1f}×")
col4.metric("Timesteps",   f"{len(results['times']):,}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "__Trajectory__",
    "__Error Over Time__",
    "__Uncertainty Ellipses__",
    "__Velocity__",
    "__Innovations__",
    "__Q Tuning Comparison__",
])

with tab1:
    st.subheader("Trajectory comparison")
    st.caption(
        "The red scatter shows raw GPS measurements. The dashed grey line is the true path "
        "the vehicle followed. The blue line is what the Kalman filter estimated."
    )
    st.pyplot(fig_trajectory(results), width='stretch')

with tab2:
    st.subheader("Position error over time")
    st.caption(
        "Point-wise Euclidean distance from the true position at each timestep, "
        "with a rolling average to show the trend. The lower panel shows cumulative RMSE — "
        "the green fill is the region the Kalman filter 'saves'."
    )
    st.pyplot(fig_error(results), width='stretch')

with tab3:
    st.subheader("Uncertainty ellipses")
    sample_every = st.slider("Sample every N steps", 10, 80, 30, key="ellipse_sample")
    st.caption(
        "Each ellipse shows the filter's 2σ position uncertainty at that moment. "
        "Ellipses shrink as the filter accumulates measurements and becomes more confident. "
        "Color darkens over time (early = light blue, late = dark blue)."
    )
    st.pyplot(fig_ellipses(results, sample_every=sample_every), width='stretch')

with tab4:
    st.subheader("Velocity estimates")
    st.caption(
        "The filter tracks velocity as part of its hidden state — even though GPS only "
        "reports position. Speed and heading are inferred entirely from the sequence of "
        "position measurements."
    )
    st.pyplot(fig_velocity(results), width='stretch')

with tab5:
    st.subheader("Filter innovations (residuals)")
    st.caption(
        "The innovation is the difference between the GPS measurement and the filter's "
        "predicted measurement. If the filter is well-tuned, innovations should look like "
        "white noise centred on zero, with ~95% of values inside the ±2σ band."
    )
    st.pyplot(fig_innovations(results), width='stretch')

    inns = results["innovations"]
    within_2sigma_x = np.mean(np.abs(inns[:, 0]) < 2 * inns[:, 0].std()) * 100
    within_2sigma_y = np.mean(np.abs(inns[:, 1]) < 2 * inns[:, 1].std()) * 100
    c1, c2 = st.columns(2)
    c1.metric("Within 2σ (x)", f"{within_2sigma_x:.1f}%", help="Ideally ~95% for a well-tuned filter")
    c2.metric("Within 2σ (y)", f"{within_2sigma_y:.1f}%", help="Ideally ~95% for a well-tuned filter")

with tab6:
    st.subheader("Effect of process noise Q on filter behaviour")
    st.caption(
        "This plot always runs three fixed values of Q (0.1, 1.5, 10.0) against your current "
        "measurement noise and seed, so you can see the full spectrum of under- to over-trusting "
        "the motion model regardless of your sidebar setting."
    )
    with st.spinner("Running three filter configurations…"):
        st.pyplot(
            fig_tuning(meas_noise_std=params["meas_noise"], seed=params["seed"]),
            width='stretch',
        )
    st.info(
        "**Low Q**: filter trusts the motion model, ignores GPS — sluggish around corners.  \n"
        "**Optimal Q**: balances model and measurements.  \n"
        "**High Q**: filter chases GPS readings — noisy but responsive.",
    )

# Footer
st.divider()
with st.expander("Kalman Filter Overview"):
    st.markdown("""
**State vector:** `x = [px, py, vx, vy]ᵀ` — position and velocity in 2D.  
Only position is measured by the GPS. Velocity must be inferred by the filter.

**Predict step** (propagate forward in time):
```
x̂ₖ⁻ = F · x̂ₖ₋₁
Pₖ⁻  = F · Pₖ₋₁ · Fᵀ + Q
```

**Update step** (incorporate new GPS reading `z`):
```
y  = z − H · x̂ₖ⁻          ← innovation (residual)
S  = H · Pₖ⁻ · Hᵀ + R      ← innovation covariance
K  = Pₖ⁻ · Hᵀ · S⁻¹        ← Kalman gain
x̂ₖ = x̂ₖ⁻ + K · y          ← state update
Pₖ  = (I−KH) · Pₖ⁻ · (I−KH)ᵀ + K·R·Kᵀ   ← Joseph form
```

**Tuning:**  
- `Q` (process noise) — how much we trust the motion model  
- `R` (measurement noise) — how noisy we believe the GPS to be  
- The Kalman gain `K` automatically balances these every step
""")
