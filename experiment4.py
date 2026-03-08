"""
Lambda Sweep: Finding the Optimal Regularisation at the Phase Transition
=========================================================================
Motivation: the basis pursuit experiment showed a non-monotonic relationship
between lambda and support recovery. FISTA lambda=1e-3 beat the Donoho-Tanner
prediction slightly, while lambda=1e-5 (~BP) performed no better than LASSO.

This suggests there is an optimal lambda that maximises support recovery,
and both over-regularisation (large lambda) and under-regularisation (small
lambda) degrade performance.

Experiment:
  - Fix rho = 0.10, N = 400
  - Run at three alpha values:
      alpha_pre   = alpha* - 1.0   (safely pre-transition, should be easy)
      alpha_crit  = alpha*         (at the predicted transition)
      alpha_post  = alpha* + 1.0   (post-transition, should be hard)
  - Sweep lambda over ~3 decades
  - Measure support recovery, NMSE, and mean active coefficients (sparsity)
  - Find empirical optimal lambda at each alpha

This tells us:
  1. Whether the optimal lambda is alpha-dependent (it should be)
  2. Where exactly the information-theoretic limit sits relative to achievable performance
  3. What the "sparsity-recovery tradeoff curve" looks like — directly relevant
     to SAE design in practice
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.notebook import tqdm

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
rng = np.random.default_rng(42)

# ── Parameters ───────────────────────────────────────────────────────────────
RHO    = 0.10
N      = 400
M      = 400    # trials per point — more for stability at the critical point

A_STAR = 1.0 / (RHO * np.log(1.0 / RHO))
print(f"ρ = {RHO},  N = {N},  α* = {A_STAR:.3f}")

ALPHA_VALS = {
    'pre-transition  (α*−1)' : A_STAR - 1.0,
    'at transition   (α*)'   : A_STAR,
    'post-transition (α*+1)' : A_STAR + 1.0,
}

# Log-spaced lambda sweep: from near-zero to large regularisation
LAMBDAS = np.logspace(-5, 0, 30)

# ── FISTA (batched, GPU) ─────────────────────────────────────────────────────
def fista_batch(W_t, n_t, lam, n_iter=2000, tol=1e-6):
    N_dim, F = W_t.shape
    M_batch  = n_t.shape[0]

    # Lipschitz constant via power iteration
    with torch.no_grad():
        v = torch.randn(F, device=device)
        v = v / v.norm()
        for _ in range(30):
            v = W_t.T @ (W_t @ v)
            nrm = v.norm()
            v = v / nrm
        L = nrm.item()

    step  = 1.0 / L
    f     = torch.zeros(M_batch, F, device=device)
    f_old = f.clone()
    t     = 1.0

    for i in range(n_iter):
        residual = f @ W_t.T - n_t
        grad     = residual @ W_t
        f_new    = torch.sign(f - step * grad) * \
                   torch.clamp((f - step * grad).abs() - step * lam, min=0.0)

        t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        f     = f_new + ((t - 1.0) / t_new) * (f_new - f_old)

        if i % 100 == 0:
            delta = (f_new - f_old).norm() / (f_old.norm() + 1e-12)
            if delta.item() < tol:
                break

        f_old = f_new
        t     = t_new

    return f_new

# ── Metrics ──────────────────────────────────────────────────────────────────
def support_recovery_batch(f_true, f_hat, k):
    true_supp = torch.topk(f_true.abs(), k, dim=1).indices
    pred_supp = torch.topk(f_hat.abs(), k, dim=1).indices
    hits = sum(
        len(set(ts.tolist()) & set(ps.tolist()))
        for ts, ps in zip(true_supp, pred_supp)
    )
    return hits / (f_true.shape[0] * k)

def nmse_batch(f_true, f_hat):
    num = ((f_true - f_hat) ** 2).sum(dim=1)
    den = (f_true ** 2).sum(dim=1).clamp(min=1e-12)
    return (num / den).mean().item()

def mean_active(f_hat, thr=1e-4):
    """Mean number of coefficients above threshold."""
    return (f_hat.abs() > thr).float().sum(dim=1).mean().item()

# ── Pre-generate data for each alpha (shared across lambda sweep) ─────────────
print("\nPre-generating data...")
data = {}
for label, alpha in ALPHA_VALS.items():
    F = max(int(round(alpha * N)), N + 1) if alpha > 1 \
        else max(int(round(alpha * N)), 2)
    k = max(1, int(round(RHO * F)))

    W_np = rng.standard_normal((N, F)).astype(np.float32)
    W_np /= np.linalg.norm(W_np, axis=0, keepdims=True)
    W_t  = torch.tensor(W_np, device=device)

    f_np = np.zeros((M, F), dtype=np.float32)
    for i in range(M):
        supp = rng.choice(F, k, replace=False)
        f_np[i, supp] = rng.standard_normal(k).astype(np.float32)

    f_t = torch.tensor(f_np, device=device)
    n_t = f_t @ W_t.T

    data[label] = dict(alpha=alpha, F=F, k=k, W_t=W_t, f_t=f_t, n_t=n_t)
    print(f"  {label}: F={F}, k={k}")

# ── Lambda sweep ──────────────────────────────────────────────────────────────
results = {label: {'supp': [], 'nmse': [], 'active': []}
           for label in ALPHA_VALS}

for label in ALPHA_VALS:
    d = data[label]
    print(f"\nSweeping λ for {label}")
    for lam in tqdm(LAMBDAS, desc=f"  λ sweep"):
        f_hat = fista_batch(d['W_t'], d['n_t'], lam=lam, n_iter=2000)
        results[label]['supp'].append(
            support_recovery_batch(d['f_t'], f_hat, d['k']))
        results[label]['nmse'].append(
            nmse_batch(d['f_t'], f_hat))
        results[label]['active'].append(
            mean_active(f_hat))

# ── Plotting ──────────────────────────────────────────────────────────────────
colors = {
    'pre-transition  (α*−1)' : '#4CAF50',
    'at transition   (α*)'   : '#2196F3',
    'post-transition (α*+1)' : '#E91E63',
}
label_short = {
    'pre-transition  (α*−1)' : f'α = {A_STAR-1:.2f}  (pre)',
    'at transition   (α*)'   : f'α = {A_STAR:.2f}  (critical)',
    'post-transition (α*+1)' : f'α = {A_STAR+1:.2f}  (post)',
}

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('#0f0f1a')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.32)

metrics = [
    ('supp',   'Support recovery accuracy', (-0.05, 1.05)),
    ('nmse',   'Normalised MSE',             (-0.05, 1.05)),
    ('active', f'Mean active coefficients\n(true k = ρF)',  (0, None)),
]

for col, (label, alpha) in enumerate(ALPHA_VALS.items()):
    c  = colors[label]
    d  = data[label]
    k  = d['k']

    for row, (metric, ylabel, ylim) in enumerate(metrics):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values():
            spine.set_color('#444466')
        ax.tick_params(colors='#aaaacc', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')

        vals = results[label][metric]
        ax.semilogx(LAMBDAS, vals, color=c, lw=2.2, alpha=0.9)

        # Mark optimal lambda (best support recovery)
        if metric == 'supp':
            best_idx = int(np.argmax(vals))
            best_lam = LAMBDAS[best_idx]
            best_val = vals[best_idx]
            ax.axvline(best_lam, color='white', lw=1.0, linestyle='--',
                       alpha=0.6, label=f'λ_opt = {best_lam:.2e}')
            ax.plot(best_lam, best_val, 'o', color='white', ms=6, zorder=5)
            ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444466',
                      labelcolor='#ccccee')

        # Mark true k for active coefficients panel
        if metric == 'active':
            ax.axhline(k, color='white', lw=0.8, linestyle=':',
                       alpha=0.5, label=f'true k = {k}')
            ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444466',
                      labelcolor='#ccccee')

        if ylim[1] is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel('λ  (log scale)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(label_short[label], color=c, fontsize=9, pad=5)

fig.suptitle(
    f"Lambda Sweep at Fixed α Values  (ρ={RHO}, N={N})\n"
    "Optimal λ maximises support recovery — both extremes degrade performance",
    color='#e8e8ff', fontsize=12, y=1.01
)
plt.savefig('lambda_sweep.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: lambda_sweep.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── Optimal lambda and peak support recovery ──")
print(f"  α* (Donoho-Tanner) = {A_STAR:.3f}")
print(f"\n  {'α label':<30} {'α':>6} {'λ_opt':>10} {'peak supp':>12} {'true k':>8}")
for label, alpha in ALPHA_VALS.items():
    supp = results[label]['supp']
    best_idx = int(np.argmax(supp))
    print(f"  {label:<30} {alpha:>6.2f} "
          f"{LAMBDAS[best_idx]:>10.2e} "
          f"{supp[best_idx]:>12.3f} "
          f"{data[label]['k']:>8}")
