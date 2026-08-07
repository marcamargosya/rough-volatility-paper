import numpy as np
from fbm import FBM

np.random.seed(42)

N = 10000          # path length (number of increments)
Q_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
MAX_LAG = 50
H_GRID = [0.1, 0.3, 0.5, 0.7]
N_REPLICATIONS = 100


def simulate_fbm_path(H, n=N):
    """Simulate one fBm sample path of length n+1 on [0,1] via Davies-Harte."""
    f = FBM(n=n, hurst=H, length=1, method='daviesharte')
    return f.fbm()  # length n+1, includes B_H(0)=0


def empirical_moment(path, q, lag):
    """m(q, lag) = average |X_{t+lag} - X_t|^q over the path."""
    increments = np.abs(path[lag:] - path[:-lag]) ** q
    return increments.mean()


def estimate_H_single_path(path, q_values=Q_VALUES, max_lag=MAX_LAG):
    """Return H_hat averaged across q, plus the per-q zeta_q slopes."""
    lags = np.arange(1, max_lag + 1)
    log_lags = np.log(lags)
    zetas = {}
    for q in q_values:
        m_vals = np.array([empirical_moment(path, q, lag) for lag in lags])
        log_m = np.log(m_vals)
        slope, intercept = np.polyfit(log_lags, log_m, 1)
        zetas[q] = slope
    H_estimates_per_q = {q: zetas[q] / q for q in q_values}
    H_hat = np.mean(list(H_estimates_per_q.values()))
    return H_hat, zetas, H_estimates_per_q


results = {}
for H_true in H_GRID:
    H_hats = []
    for rep in range(N_REPLICATIONS):
        path = simulate_fbm_path(H_true)
        H_hat, zetas, _ = estimate_H_single_path(path)
        H_hats.append(H_hat)
    H_hats = np.array(H_hats)
    results[H_true] = (H_hats.mean(), H_hats.std())
    print(f"H_true={H_true:.1f}  mean(H_hat)={H_hats.mean():.4f}  std={H_hats.std():.4f}")

with open('results_table.txt', 'w') as f:
    f.write("H_true, mean_H_hat, std_H_hat\n")
    for H_true, (m, s) in results.items():
        f.write(f"{H_true}, {m:.4f}, {s:.4f}\n")

print("\nSaved results table.")