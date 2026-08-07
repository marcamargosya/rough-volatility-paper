import numpy as np
from fbm import FBM
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

N = 10000
Q_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
MAX_LAG = 50
H_GRID = [0.1, 0.3, 0.5, 0.7]

def simulate_fbm_path(H, n=N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    f = FBM(n=n, hurst=H, length=1, method='daviesharte')
    return f.fbm()

def empirical_moment(path, q, lag):
    increments = np.abs(path[lag:] - path[:-lag]) ** q
    return increments.mean()

# Figure 1: sample paths for four H values
fig, ax = plt.subplots(figsize=(8, 5))
t = np.linspace(0, 1, N + 1)
colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
for H, c in zip(H_GRID, colors):
    path = simulate_fbm_path(H, seed=123)
    ax.plot(t, path, label=f"$H = {H}$", color=c, linewidth=1.1)
ax.set_xlabel("$t$")
ax.set_ylabel("$B_H(t)$")
ax.set_title("Sample paths of fractional Brownian motion for varying $H$")
ax.legend()
fig.tight_layout()
fig.savefig("fig1_sample_paths.png")
plt.close(fig)

# Figure 2: log-log regression for one H
H_demo = 0.3
q_demo = 2.0
path = simulate_fbm_path(H_demo, seed=7)
lags = np.arange(1, MAX_LAG + 1)
log_lags = np.log(lags)
m_vals = np.array([empirical_moment(path, q_demo, lag) for lag in lags])
log_m = np.log(m_vals)
slope, intercept = np.polyfit(log_lags, log_m, 1)
H_hat = slope / q_demo

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(log_lags, log_m, s=18, color='#1f77b4', label="Empirical $\\log m(q,\\Delta)$")
ax.plot(log_lags, slope * log_lags + intercept, color='#d62728',
        label=f"OLS fit: slope $= {slope:.3f}$")
ax.set_xlabel("$\\log \\Delta$")
ax.set_ylabel("$\\log m(q, \\Delta)$")
ax.set_title(f"Log-log regression, $H_{{true}}={H_demo}$, $q={q_demo}$  "
             f"($\\hat{{H}} = {H_hat:.3f}$)")
ax.legend()
fig.tight_layout()
fig.savefig("fig2_loglog_regression.png")
plt.close(fig)

# Figure 3: zeta_q vs q
H_demo2 = 0.3
path2 = simulate_fbm_path(H_demo2, seed=7)
zetas = []
for q in Q_VALUES:
    m_vals_q = np.array([empirical_moment(path2, q, lag) for lag in lags])
    log_m_q = np.log(m_vals_q)
    s, _ = np.polyfit(log_lags, log_m_q, 1)
    zetas.append(s)
zetas = np.array(zetas)

H_from_zeta_fit = np.polyfit(Q_VALUES, zetas, 1)[0]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Q_VALUES, zetas, s=45, color='#1f77b4', zorder=3, label="Fitted $\\zeta_q$")
q_line = np.linspace(0, 3.2, 50)
ax.plot(q_line, H_demo2 * q_line, color='#2ca02c', linestyle='--',
        label=f"Theoretical $qH$ ($H={H_demo2}$)")
ax.plot(q_line, H_from_zeta_fit * q_line, color='#d62728',
        label=f"OLS fit: slope $= {H_from_zeta_fit:.3f}$")
ax.set_xlabel("$q$")
ax.set_ylabel("$\\zeta_q$")
ax.set_title("Linearity check: $\\zeta_q$ vs $q$")
ax.legend()
fig.tight_layout()
fig.savefig("fig3_zeta_vs_q.png")
plt.close(fig)

print("Figures saved.")
print(f"Fig2: H_hat from q={q_demo} regression = {H_hat:.4f} (true {H_demo})")
print(f"Fig3: H from zeta_q vs q slope = {H_from_zeta_fit:.4f} (true {H_demo2})")