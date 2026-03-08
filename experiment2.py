"""
Finite-Size Scaling for the SAE Page Transition
=================================================
Fixes rho = 0.10 and sweeps alpha at several values of N.
The prediction: as N -> inf, the recovery curve approaches a step function
at alpha*(rho) ~ 1 / (rho * log(1/rho)).

If the transition sharpens with N, we have evidence for a true
thermodynamic phase transition rather than a gradual crossover.

We also plot the scaling collapse: if x = (alpha - alpha*) * N^(1/nu)
rescales all curves onto a single master curve, nu is the critical exponent.
This is the standard finite-size scaling ansatz from statistical physics.

Runtime: T4 GPU recommended. Estimated ~15-20 min at these settings.
"""

import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Lasso
from tqdm.notebook import tqdm

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
rng = np.random.default_rng(42)

# ── Fixed parameters ────────────────────────────────────────────────────────
RHO       = 0.10                         # fixed sparsity fraction
N_VALS    = [50, 100, 200, 400, 800]     # neuron counts to sweep
M         = 300                          # trials per point
ALPHAS    = np.linspace(0.4, 6.5, 30)   # superposition ratios

# ── Predicted transition ────────────────────────────────────────────────────
def alpha_star(rho, C=1.14):
    return 1.0 / (C * rho * np.log(1.0 / rho))

A_STAR = alpha_star(RHO)
print(f"\nρ = {RHO}  →  α* predicted ≈ {A_STAR:.3f}")

# ── Metrics ────────────────────────────────────────────────────────────────
def support_recovery_batch(f_true, f_hat, k):
    true_supp = torch.topk(f_true.abs(), k, dim=1).indices
    pred_supp = torch.topk(f_hat.abs(),  k, dim=1).indices
    hits = 0.0
    for ts, ps in zip(true_supp, pred_supp):
        hits += len(set(ts.tolist()) & set(ps.tolist()))
    return hits / (M * k)

def nmse_batch(f_true, f_hat):
    num = ((f_true - f_hat) ** 2).sum(dim=1)
    den = (f_true ** 2).sum(dim=1).clamp(min=1e-12)
    return (num / den).mean().item()

# ── Linear decode (GPU) ─────────────────────────────────────────────────────
def linear_decode(W_t, n_t):
    sol = torch.linalg.lstsq(W_t, n_t.T).solution
    return sol.T

# ── LASSO decode (CPU) ──────────────────────────────────────────────────────
def calibrate_lambda(W_np, n_sample, k, n_calib=20):
    """Binary search for lambda hitting target sparsity k."""
    def mean_nnz(lam):
        m = Lasso(alpha=lam, fit_intercept=False, max_iter=2000, tol=1e-4)
        nnz = []
        for i in range(min(n_calib, len(n_sample))):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(W_np, n_sample[i])
            nnz.append(np.sum(np.abs(m.coef_) > 1e-6))
        return np.mean(nnz)

    lo, hi = 1e-5, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if mean_nnz(mid) > k:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def lasso_decode(W_np, n_batch_np, k, F, lam):
    model = Lasso(alpha=lam, fit_intercept=False,
                  max_iter=3000, tol=1e-4, warm_start=True)
    out = np.zeros((M, F), dtype=np.float32)
    for i in range(M):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(W_np, n_batch_np[i])
        out[i] = model.coef_
    return out

# ── Main sweep ──────────────────────────────────────────────────────────────
# results[N] = {'lin_supp': [], 'sps_supp': [], 'lin_nmse': [], 'sps_nmse': []}
results = {N: {k: [] for k in ['lin_supp', 'sps_supp', 'lin_nmse', 'sps_nmse']}
           for N in N_VALS}

for N in N_VALS:
    print(f"\nN = {N}")
    for alpha in tqdm(ALPHAS, desc=f"  N={N}"):
        F = max(int(round(alpha * N)), N + 1) if alpha > 1 \
            else max(int(round(alpha * N)), 2)
        k = max(1, int(round(RHO * F)))

        # Shared random dictionary
        W_np = rng.standard_normal((N, F)).astype(np.float32)
        W_np /= np.linalg.norm(W_np, axis=0, keepdims=True)
        W_t  = torch.tensor(W_np, device=device)

        # Batch of sparse feature vectors
        f_np = np.zeros((M, F), dtype=np.float32)
        for i in range(M):
            supp = rng.choice(F, k, replace=False)
            f_np[i, supp] = rng.standard_normal(k).astype(np.float32)

        f_t  = torch.tensor(f_np, device=device)
        n_t  = f_t @ W_t.T
        n_np = n_t.cpu().numpy()

        # Linear decode (GPU)
        f_lin_t = linear_decode(W_t, n_t)
        results[N]['lin_supp'].append(support_recovery_batch(f_t, f_lin_t, k))
        results[N]['lin_nmse'].append(nmse_batch(f_t, f_lin_t))

        # LASSO decode (CPU) — calibrate lambda once per point
        lam = calibrate_lambda(W_np, n_np, k)
        f_sps_np = lasso_decode(W_np, n_np, k, F, lam)
        f_sps_t  = torch.tensor(f_sps_np, device=device)
        results[N]['sps_supp'].append(support_recovery_batch(f_t, f_sps_t, k))
        results[N]['sps_nmse'].append(nmse_batch(f_t, f_sps_t))

# ── Colours: one per N ──────────────────────────────────────────────────────
cmap   = plt.cm.plasma
colors = [cmap(i / (len(N_VALS) - 1)) for i in range(len(N_VALS))]

# ── Plot 1: raw curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0f0f1a')

for ax in axes:
    ax.set_facecolor('#1a1a2e')
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.tick_params(colors='#aaaacc', labelsize=9)
    ax.xaxis.label.set_color('#aaaacc')
    ax.yaxis.label.set_color('#aaaacc')
    ax.axvline(A_STAR, color='white', lw=1.0, linestyle=':', alpha=0.5,
               label=f'α* ≈ {A_STAR:.2f}')
    ax.axvline(1.0, color='#aaaacc', lw=0.7, linestyle=':', alpha=0.35,
               label='α = 1')

for i, N in enumerate(N_VALS):
    c = colors[i]
    axes[0].plot(ALPHAS, results[N]['sps_supp'], color=c, lw=1.8,
                 label=f'N={N}', alpha=0.9)
    axes[1].plot(ALPHAS, results[N]['sps_nmse'], color=c, lw=1.8,
                 label=f'N={N}', alpha=0.9)

axes[0].set_xlabel('α = F/N', fontsize=10)
axes[0].set_ylabel('Support recovery accuracy', fontsize=9)
axes[0].set_ylim(-0.05, 1.05)
axes[0].set_title(f'LASSO support recovery  (ρ={RHO})', color='#ccccee', fontsize=10)
axes[0].legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444466',
               labelcolor='#ccccee', loc='upper right')

axes[1].set_xlabel('α = F/N', fontsize=10)
axes[1].set_ylabel('Normalised MSE', fontsize=9)
axes[1].set_title(f'LASSO normalised MSE  (ρ={RHO})', color='#ccccee', fontsize=10)
axes[1].legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444466',
               labelcolor='#ccccee', loc='upper left')

fig.suptitle(
    f"Finite-Size Scaling of the SAE Page Transition  (ρ = {RHO})\n"
    r"Prediction: curves sharpen toward step function at $\alpha^*$ as $N\to\infty$",
    color='#e8e8ff', fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig('finite_size_raw.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: finite_size_raw.png")

# ── Plot 2: scaling collapse ────────────────────────────────────────────────
# Ansatz: near alpha*, P(alpha, N) = f( (alpha - alpha*) * N^(1/nu) )
# We try nu=1 first (mean-field / RMT guess) and nu=2.
# A good collapse = all curves fall on one master curve.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0f0f1a')

for nu, ax in zip([1.0, 2.0], axes):
    ax.set_facecolor('#1a1a2e')
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.tick_params(colors='#aaaacc', labelsize=9)
    ax.xaxis.label.set_color('#aaaacc')
    ax.yaxis.label.set_color('#aaaacc')
    ax.axvline(0.0, color='white', lw=0.8, linestyle=':', alpha=0.4)
    ax.set_xlabel(r'$(\alpha - \alpha^*)\, N^{1/\nu}$', fontsize=10)
    ax.set_ylabel('Support recovery accuracy', fontsize=9)
    ax.set_title(f'Scaling collapse  ν = {nu}', color='#ccccee', fontsize=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)

    for i, N in enumerate(N_VALS):
        c = colors[i]
        x_scaled = (ALPHAS - A_STAR) * (N ** (1.0 / nu))
        ax.plot(x_scaled, results[N]['sps_supp'], color=c, lw=1.8,
                label=f'N={N}', alpha=0.9)

    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444466',
              labelcolor='#ccccee')

fig.suptitle(
    "Finite-Size Scaling Collapse\n"
    r"Good collapse onto one master curve → confirmed phase transition with exponent $\nu$",
    color='#e8e8ff', fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig('finite_size_collapse.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: finite_size_collapse.png")

# ── Transition location vs N ────────────────────────────────────────────────
# Plot the empirical alpha* (support=0.5 crossing) vs N on log scale.
# If it converges to A_STAR as N grows, prediction is confirmed.

def crossing(vals, threshold=0.5):
    for i in range(len(vals) - 1):
        if vals[i] >= threshold >= vals[i+1]:
            # linear interpolation
            t = (vals[i] - threshold) / (vals[i] - vals[i+1])
            return ALPHAS[i] + t * (ALPHAS[i+1] - ALPHAS[i])
    return float('nan')

empirical_stars = [crossing(results[N]['sps_supp']) for N in N_VALS]

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')
for spine in ax.spines.values():
    spine.set_color('#444466')
ax.tick_params(colors='#aaaacc', labelsize=9)
ax.xaxis.label.set_color('#aaaacc')
ax.yaxis.label.set_color('#aaaacc')

ax.semilogx(N_VALS, empirical_stars, 'o-', color='#2196F3', lw=2,
            markersize=8, label='Empirical α* (support=0.5 crossing)')
ax.axhline(A_STAR, color='white', lw=1.2, linestyle=':', alpha=0.6,
           label=f'Predicted α* = {A_STAR:.2f}')

ax.set_xlabel('N  (log scale)', fontsize=10)
ax.set_ylabel('Empirical transition α*', fontsize=9)
ax.set_title(f'Convergence of empirical α* to Donoho-Tanner prediction  (ρ={RHO})',
             color='#ccccee', fontsize=10)
ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444466',
          labelcolor='#ccccee')

plt.tight_layout()
plt.savefig('finite_size_convergence.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: finite_size_convergence.png")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n── Empirical transition locations (support recovery = 0.5) ──")
print(f"{'N':>6}  {'α*(empirical)':>15}  {'α*(predicted)':>15}")
for N, a_emp in zip(N_VALS, empirical_stars):
    print(f"{N:>6}  {a_emp:>15.3f}  {A_STAR:>15.3f}")
