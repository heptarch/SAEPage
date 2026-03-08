"""
Basis Pursuit vs LASSO: Testing the Donoho-Tanner Bound
=========================================================
Motivation: the finite-size scaling experiment showed no N-dependence and
empirical alpha* ~3.7-3.9, well below the Donoho-Tanner prediction of 4.34.
Hypothesis: the LASSO decoder (finite lambda) is the bottleneck, not the
information-theoretic limit.

Test: replace LASSO with basis pursuit (BP), which is the *optimal* l1
decoder that the Donoho-Tanner theory actually describes:

    BP:   min ||f||_1   s.t.   Wf = n          (lambda -> 0)
    LASSO: min (1/2)||Wf-n||^2 + lambda*||f||_1  (finite lambda)

We implement BP via FISTA on the penalised problem with lambda -> 0,
batched entirely on GPU. This avoids the LASSO lambda-calibration bias
and should recover the theoretical threshold.

Three decoders compared:
  1. LASSO   : sklearn, finite lambda (calibrated to sparsity k)  [previous]
  2. FISTA-lo: lambda = 1e-3, many iterations                     [near-BP]
  3. FISTA-bp: lambda = 1e-5, many iterations                     [approx BP]

If FISTA-bp recovers alpha* ~ 4.34, the decoder was the bottleneck.
If it still falls short, there is something deeper (distribution, geometry).

Runtime: T4 GPU. ~10-15 min at default settings.
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Lasso
from tqdm.notebook import tqdm

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
rng = np.random.default_rng(42)

# ── Parameters ───────────────────────────────────────────────────────────────
RHO       = 0.10
N         = 400                           # fixed, large enough to be meaningful
M         = 300                           # trials per alpha point

# Focus sweep around the predicted transition — denser near alpha*
A_STAR    = 1.0 / (1.14 * RHO * np.log(1.0 / RHO))
ALPHAS    = np.concatenate([
    np.linspace(0.5, A_STAR - 1.0, 8),   # pre-transition
    np.linspace(A_STAR - 1.0, A_STAR + 1.5, 16),  # near transition (dense)
    np.linspace(A_STAR + 1.5, 6.5, 5),   # post-transition
])
ALPHAS    = np.unique(np.round(ALPHAS, 4))

print(f"ρ = {RHO},  N = {N},  α* predicted = {A_STAR:.3f}")
print(f"Alpha sweep: {len(ALPHAS)} points from {ALPHAS[0]:.2f} to {ALPHAS[-1]:.2f}")

# ── FISTA (batched, GPU) ─────────────────────────────────────────────────────
def fista_batch(W_t, n_t, lam, n_iter=2000, tol=1e-6):
    """
    Solves M instances of:
        min_f  (1/2)||Wf - n||^2 + lam * ||f||_1
    simultaneously on GPU.

    W_t  : (N, F)  shared dictionary
    n_t  : (M, N)  batch of observations
    lam  : scalar  regularisation (small -> approaches basis pursuit)

    Returns f_hat : (M, F)

    Step size: 1 / sigma_max(W)^2  (Lipschitz constant of gradient)
    Estimated via power iteration — cheap since W is shared.
    """
    N_dim, F = W_t.shape
    M_batch  = n_t.shape[0]

    # Lipschitz constant L = sigma_max(W)^2
    # For normalised columns: L <= 1 + off-diagonal coherence * F,
    # but safe upper bound via power iteration
    with torch.no_grad():
        v = torch.randn(F, device=device)
        v = v / v.norm()
        for _ in range(30):
            v = W_t.T @ (W_t @ v)
            nrm = v.norm()
            v = v / nrm
        L = nrm.item()

    step = 1.0 / L

    # Initialise
    f     = torch.zeros(M_batch, F, device=device)
    f_old = f.clone()
    t     = 1.0

    for _ in range(n_iter):
        # Gradient of smooth part: W^T(Wf - n),  shape (M, F)
        residual = f @ W_t.T - n_t          # (M, N)
        grad     = residual @ W_t            # (M, F)

        # Gradient step + soft threshold (proximal operator for l1)
        f_new = torch.sign(f - step * grad) * \
                torch.clamp(torch.abs(f - step * grad) - step * lam, min=0.0)

        # FISTA momentum
        t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        f     = f_new + ((t - 1.0) / t_new) * (f_new - f_old)

        # Convergence check (every 100 steps for speed)
        if _ % 100 == 0:
            delta = (f_new - f_old).norm() / (f_old.norm() + 1e-12)
            if delta.item() < tol:
                break

        f_old = f_new
        t     = t_new

    return f_new

# ── Metrics ──────────────────────────────────────────────────────────────────
def support_recovery_batch(f_true, f_hat, k):
    true_supp = torch.topk(f_true.abs(), k, dim=1).indices
    pred_supp = torch.topk(f_hat.abs(),  k, dim=1).indices
    hits = sum(
        len(set(ts.tolist()) & set(ps.tolist()))
        for ts, ps in zip(true_supp, pred_supp)
    )
    return hits / (f_true.shape[0] * k)

def nmse_batch(f_true, f_hat):
    num = ((f_true - f_hat) ** 2).sum(dim=1)
    den = (f_true ** 2).sum(dim=1).clamp(min=1e-12)
    return (num / den).mean().item()

# ── LASSO decoder (calibrated, CPU) ─────────────────────────────────────────
def calibrate_and_decode_lasso(W_np, n_np, k, F):
    def mean_nnz(lam):
        m = Lasso(alpha=lam, fit_intercept=False, max_iter=2000, tol=1e-4)
        nnz = []
        for i in range(min(20, M)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(W_np, n_np[i])
            nnz.append(np.sum(np.abs(m.coef_) > 1e-6))
        return np.mean(nnz)

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
            model.fit(W_np, n_np[i])
        out[i] = model.coef_
    return out

# ── Main sweep ───────────────────────────────────────────────────────────────
decoders = {
    'LASSO (calibrated)' : {'supp': [], 'nmse': [], 'color': '#ff7043', 'ls': '--'},
    'FISTA  λ=1e-3'      : {'supp': [], 'nmse': [], 'color': '#2196F3', 'ls': '-'},
    'FISTA  λ=1e-5 (≈BP)': {'supp': [], 'nmse': [], 'color': '#4CAF50', 'ls': '-'},
}

for alpha in tqdm(ALPHAS, desc="Alpha sweep"):
    F = max(int(round(alpha * N)), N + 1) if alpha > 1 \
        else max(int(round(alpha * N)), 2)
    k = max(1, int(round(RHO * F)))

    # Shared dictionary
    W_np = rng.standard_normal((N, F)).astype(np.float32)
    W_np /= np.linalg.norm(W_np, axis=0, keepdims=True)
    W_t  = torch.tensor(W_np, device=device)

    # Sparse feature batch
    f_np = np.zeros((M, F), dtype=np.float32)
    for i in range(M):
        supp = rng.choice(F, k, replace=False)
        f_np[i, supp] = rng.standard_normal(k).astype(np.float32)

    f_t  = torch.tensor(f_np, device=device)
    n_t  = f_t @ W_t.T
    n_np = n_t.cpu().numpy()

    # ── LASSO (CPU) ────────────────────────────────────────────────────────
    f_lasso_np = calibrate_and_decode_lasso(W_np, n_np, k, F)
    f_lasso_t  = torch.tensor(f_lasso_np, device=device)
    decoders['LASSO (calibrated)']['supp'].append(
        support_recovery_batch(f_t, f_lasso_t, k))
    decoders['LASSO (calibrated)']['nmse'].append(
        nmse_batch(f_t, f_lasso_t))

    # ── FISTA λ=1e-3 (GPU) ────────────────────────────────────────────────
    f_flo = fista_batch(W_t, n_t, lam=1e-3, n_iter=2000)
    decoders['FISTA  λ=1e-3']['supp'].append(
        support_recovery_batch(f_t, f_flo, k))
    decoders['FISTA  λ=1e-3']['nmse'].append(
        nmse_batch(f_t, f_flo))

    # ── FISTA λ=1e-5 ≈ BP (GPU) ───────────────────────────────────────────
    f_fbp = fista_batch(W_t, n_t, lam=1e-5, n_iter=3000)
    decoders['FISTA  λ=1e-5 (≈BP)']['supp'].append(
        support_recovery_batch(f_t, f_fbp, k))
    decoders['FISTA  λ=1e-5 (≈BP)']['nmse'].append(
        nmse_batch(f_t, f_fbp))

# ── Plotting ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0f0f1a')

for ax, (metric, ylabel, ylim) in zip(axes, [
    ('supp', 'Support recovery accuracy', (-0.05, 1.05)),
    ('nmse', 'Normalised MSE',            (-0.05, 1.0)),
]):
    ax.set_facecolor('#1a1a2e')
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.tick_params(colors='#aaaacc', labelsize=9)
    ax.xaxis.label.set_color('#aaaacc')
    ax.yaxis.label.set_color('#aaaacc')

    ax.axvline(A_STAR, color='white', lw=1.2, linestyle=':',
               alpha=0.6, label=f'α* ≈ {A_STAR:.2f}  (D-T)')
    ax.axvline(1.0,    color='#aaaacc', lw=0.7, linestyle=':',
               alpha=0.35, label='α = 1')

    for name, d in decoders.items():
        ax.plot(ALPHAS, d[metric], color=d['color'], lw=2.0,
                linestyle=d['ls'], label=name, alpha=0.9)

    ax.set_xlabel('α = F/N', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(ylim)
    ax.set_title(
        f'{"Support recovery" if metric == "supp" else "Normalised MSE"}'
        f'  (ρ={RHO}, N={N})',
        color='#ccccee', fontsize=10
    )
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444466',
              labelcolor='#ccccee', loc='upper right' if metric == 'supp' else 'upper left')

fig.suptitle(
    "Basis Pursuit vs LASSO: Is the Decoder the Bottleneck?\n"
    "If FISTA λ→0 recovers α* ≈ 4.34, LASSO calibration bias explains the gap",
    color='#e8e8ff', fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig('basis_pursuit.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: basis_pursuit.png")

# ── Summary table ─────────────────────────────────────────────────────────────
def crossing(vals, alphas, threshold=0.5):
    for i in range(len(vals) - 1):
        if vals[i] >= threshold >= vals[i+1]:
            t = (vals[i] - threshold) / (vals[i] - vals[i+1])
            return alphas[i] + t * (alphas[i+1] - alphas[i])
    return float('nan')

print(f"\n── Empirical α* (support recovery = 0.5) ──")
print(f"  Predicted (Donoho-Tanner):   {A_STAR:.3f}")
for name, d in decoders.items():
    emp = crossing(d['supp'], ALPHAS)
    gap = emp - A_STAR
    print(f"  {name:<28}  α* = {emp:.3f}   (gap = {gap:+.3f})")
