"""
Page Curve for Sparse Autoencoders  —  GPU edition
=====================================================
Tests the predicted phase transition at alpha*(rho) ~ 1 / (C * rho * log(1/rho))
comparing linear (pseudoinverse) vs sparse (LASSO) decoders.

Changes from v1:
  - Core matrix ops moved to PyTorch + CUDA (GPU batching over M trials)
  - OMP replaced with LASSO: more robust at high alpha, and directly
    analogous to SAE training (L1 penalty). OMP's Cholesky decomposition
    fails when F >> N due to near-linear-dependence in the dictionary;
    LASSO has no such failure mode.
  - LASSO regularisation strength lambda auto-tuned per (alpha, rho) point
    to hit target sparsity k, avoiding manual lambda sweeps.

Variables:
  F     : number of latent features
  N     : number of neurons
  k     : number of active features per sample  (k = rho * F)
  W     : random Gaussian encoding matrix  R^(N x F), cols normalised
  n     : neuron activations  n = Wf
  alpha = F/N : superposition ratio  (control parameter)
  rho   = k/F : sparsity fraction

Runtime:  Runtime -> Change runtime type -> T4 GPU
"""

# ── Installs ───────────────────────────────────────────────────────────────
# !pip install tqdm scikit-learn -q   # already present on Colab

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

# ── Experiment parameters ──────────────────────────────────────────────────
N         = 200           # neuron dimension (fixed)
M         = 400           # trials per (alpha, rho) point  — batched on GPU
alphas    = np.linspace(0.4, 6.5, 35)
rho_vals  = [0.05, 0.10, 0.20]

# ── Predicted phase transition ─────────────────────────────────────────────
def alpha_star(rho, C=1.14):
    return 1.0 / (C * rho * np.log(1.0 / rho))

# ── Metrics ────────────────────────────────────────────────────────────────
def support_recovery_batch(f_true, f_hat, k):
    """
    f_true, f_hat: torch tensors (M, F)
    Returns mean support recovery accuracy over batch.
    """
    true_supp = torch.topk(f_true.abs(), k, dim=1).indices   # (M, k)
    pred_supp = torch.topk(f_hat.abs(),  k, dim=1).indices   # (M, k)
    hits = 0.0
    for ts, ps in zip(true_supp, pred_supp):
        hits += len(set(ts.tolist()) & set(ps.tolist()))
    return hits / (M * k)

def nmse_batch(f_true, f_hat):
    """Mean normalised MSE over batch."""
    num  = ((f_true - f_hat) ** 2).sum(dim=1)
    den  = (f_true ** 2).sum(dim=1).clamp(min=1e-12)
    return (num / den).mean().item()

# ── Linear decoder (batched pseudoinverse on GPU) ──────────────────────────
def linear_decode(W_t, n_t):
    """
    W_t : (N, F)  — shared dictionary
    n_t : (M, N)  — batch of activations
    Returns (M, F) reconstructions via least-squares.
    """
    sol = torch.linalg.lstsq(W_t, n_t.T).solution   # (F, M)
    return sol.T                                       # (M, F)

# ── LASSO decoder (sklearn, serial over M — CPU but robust at all alpha) ───
def lasso_decode(W_np, n_batch_np, k, F):
    """
    W_np      : (N, F) numpy
    n_batch_np: (M, N) numpy
    Returns   : (M, F) numpy reconstructions.

    Lambda is chosen by binary search to hit target sparsity ~ k.
    Calibrated once per (alpha, rho) point for speed.
    """
    def mean_nnz(lam):
        m = Lasso(alpha=lam, fit_intercept=False, max_iter=2000, tol=1e-4)
        nnz = []
        for i in range(min(20, M)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(W_np, n_batch_np[i])
            nnz.append(np.sum(np.abs(m.coef_) > 1e-6))
        return np.mean(nnz)

    # Binary search: find lambda s.t. mean nnz ~ k
    lo, hi = 1e-5, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if mean_nnz(mid) > k:
            lo = mid
        else:
            hi = mid
    best_lam = (lo + hi) / 2

    model = Lasso(alpha=best_lam, fit_intercept=False,
                  max_iter=3000, tol=1e-4, warm_start=True)
    out = np.zeros((M, F), dtype=np.float32)
    for i in range(M):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(W_np, n_batch_np[i])
        out[i] = model.coef_
    return out

# ── Main sweep ─────────────────────────────────────────────────────────────
results = {(rho, dec): {'supp': [], 'nmse': []}
           for rho in rho_vals for dec in ['linear', 'sparse']}

for rho in rho_vals:
    print(f"\nρ = {rho:.2f}  |  α* predicted ≈ {alpha_star(rho):.2f}")
    for alpha in tqdm(alphas, desc=f"  ρ={rho}"):
        F = max(int(round(alpha * N)), N + 1) if alpha > 1 \
            else max(int(round(alpha * N)), 2)
        k = max(1, int(round(rho * F)))

        # ── Build shared random dictionary ────────────────────────────────
        W_np = rng.standard_normal((N, F)).astype(np.float32)
        W_np /= np.linalg.norm(W_np, axis=0, keepdims=True)
        W_t  = torch.tensor(W_np, device=device)

        # ── Build batch of sparse feature vectors ─────────────────────────
        f_np = np.zeros((M, F), dtype=np.float32)
        for i in range(M):
            supp = rng.choice(F, k, replace=False)
            f_np[i, supp] = rng.standard_normal(k).astype(np.float32)

        f_t  = torch.tensor(f_np, device=device)   # (M, F)
        n_t  = (f_t @ W_t.T)                        # (M, N)
        n_np = n_t.cpu().numpy()

        # ── Linear decode (GPU) ───────────────────────────────────────────
        f_lin_t = linear_decode(W_t, n_t)
        results[(rho, 'linear')]['supp'].append(
            support_recovery_batch(f_t, f_lin_t, k))
        results[(rho, 'linear')]['nmse'].append(
            nmse_batch(f_t, f_lin_t))

        # ── LASSO decode (CPU, robust at all alpha) ───────────────────────
        f_sps_np = lasso_decode(W_np, n_np, k, F)
        f_sps_t  = torch.tensor(f_sps_np, device=device)
        results[(rho, 'sparse')]['supp'].append(
            support_recovery_batch(f_t, f_sps_t, k))
        results[(rho, 'sparse')]['nmse'].append(
            nmse_batch(f_t, f_sps_t))

# ── Plotting ───────────────────────────────────────────────────────────────
colors = ['#2196F3', '#E91E63', '#4CAF50']
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#0f0f1a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

for col, rho in enumerate(rho_vals):
    a_star = alpha_star(rho)
    c = colors[col]

    for row, (metric, ylabel, ylim) in enumerate([
        ('supp', 'Support recovery accuracy', (-0.05, 1.05)),
        ('nmse', 'Normalised MSE',             (-0.05, 1.8)),
    ]):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values():
            spine.set_color('#444466')
        ax.tick_params(colors='#aaaacc', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')

        lin_vals = results[(rho, 'linear')][metric]
        sps_vals = results[(rho, 'sparse')][metric]

        ax.plot(alphas, lin_vals, color='#ff7043', lw=1.8,
                linestyle='--', label='Linear (pseudoinverse)', alpha=0.85)
        ax.plot(alphas, sps_vals, color=c, lw=2.2,
                label='Sparse (LASSO)', alpha=0.95)

        ax.axvline(a_star, color='white', lw=1.2, linestyle=':',
                   alpha=0.6, label=f'α* ≈ {a_star:.2f}')
        ax.axvline(1.0,    color='#aaaacc', lw=0.8, linestyle=':',
                   alpha=0.4, label='α = 1')

        ax.set_xlabel('α = F/N', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ylim)
        ax.set_title(f'ρ = {rho}  (k = ρF active)', color='#ccccee',
                     fontsize=9, pad=6)

        if row == 0 and col == 1:
            ax.legend(fontsize=7, loc='lower left',
                      facecolor='#1a1a2e', edgecolor='#444466',
                      labelcolor='#ccccee')

#fig.suptitle(
#    "Classical Page Curve for Sparse Recovery\n"
#    r"Phase transition at $\alpha^*(\rho) \approx 1\,/\,(C\,\rho\,\log(1/\rho))$",
#    color='#e8e8ff', fontsize=13, y=0.98
#)
plt.savefig('page_curve.svg', dpi=400, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: page_curve.svg")

# ── Summary table ──────────────────────────────────────────────────────────
def crossing(vals, threshold=0.5):
    for i, v in enumerate(vals):
        if v < threshold:
            return alphas[i]
    return float('nan')

print(f"\n── Phase transition summary (support recovery crossing 0.5) ──")
print(f"{'rho':>6} {'α*(pred)':>10} {'α*(linear)':>12} {'α*(LASSO)':>12}")
for rho in rho_vals:
    pred  = alpha_star(rho)
    lin_c = crossing(results[(rho, 'linear')]['supp'])
    sps_c = crossing(results[(rho, 'sparse')]['supp'])
    print(f"{rho:>6.2f} {pred:>10.2f} {lin_c:>12.2f} {sps_c:>12.2f}")
