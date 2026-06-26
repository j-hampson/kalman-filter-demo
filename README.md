# 2D Kalman Filter Demo

A Python implementation of a linear Kalman filter applied to 2D vehicle tracking.


## Project files

```
kalman.py          # Core filter: KalmanFilter2D class
simulate.py        # Ground-truth trajectory and GPS noise generation
run_filter.py      # Runs filter and computes RMSE
app.py             # Streamlit web app
requirements.txt   # Python dependencies
```


## Run web app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Formulas used

### State vector
```
x = [px, py, vx, vy]ᵀ
```
2D position and velocity. Position is measured by GPS, velocity is inferred by the filter.

### State transition (constant-velocity model)
```
F = [[1, 0, dt,  0 ],
     [0, 1,  0, dt ],
     [0, 0,  1,  0 ],
     [0, 0,  0,  1 ]]
```

### Predict step
```
x̂ₖ⁻ = F · x̂ₖ₋₁
Pₖ⁻  = F · Pₖ₋₁ · Fᵀ + Q
```

### Update step
```
y  = z - H · x̂ₖ⁻              (innovation)
S  = H · Pₖ⁻ · Hᵀ + R          (innovation covariance)
K  = Pₖ⁻ · Hᵀ · S⁻¹            (Kalman gain)
x̂ₖ = x̂ₖ⁻ + K · y             (state update)
Pₖ  = (I - K·H) · Pₖ⁻ · (I - K·H)ᵀ + K·R·Kᵀ   (Joseph form)
```
