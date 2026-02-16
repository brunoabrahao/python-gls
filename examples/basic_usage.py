"""Basic usage example for python_gls.

Demonstrates GLS estimation with learned correlation and variance
structures on a simulated repeated-measures dataset.
"""

import numpy as np
import pandas as pd

from python_gls import GLS
from python_gls.correlation import CorAR1, CorCompSymm, CorSymm
from python_gls.variance import VarIdent

# ── Generate simulated panel data ────────────────────────────────────────────
np.random.seed(42)
n_subjects = 100
n_times = 5
N = n_subjects * n_times

subject = np.repeat(np.arange(n_subjects), n_times)
time = np.tile(np.arange(n_times), n_subjects)
treatment = np.random.choice([0, 1], n_subjects)[subject]
x = np.random.randn(N)

# True model: y = 1 + 0.5*treatment + 2*x + AR(1) errors (phi=0.6)
phi_true = 0.6
errors = np.zeros(N)
for s in range(n_subjects):
    idx = slice(s * n_times, (s + 1) * n_times)
    e = np.random.randn(n_times)
    for t in range(1, n_times):
        e[t] = phi_true * e[t - 1] + np.sqrt(1 - phi_true**2) * e[t]
    errors[idx] = e

y = 1.0 + 0.5 * treatment + 2.0 * x + errors

df = pd.DataFrame({
    "y": y, "x": x, "treatment": treatment,
    "subject": subject, "time": time,
})

# ── 1. OLS (ignoring correlation) ────────────────────────────────────────────
print("=" * 60)
print("1. OLS (no correlation structure)")
print("=" * 60)
result_ols = GLS.from_formula("y ~ treatment + x", data=df).fit()
print(result_ols.summary())
print()

# ── 2. GLS with AR(1) correlation ────────────────────────────────────────────
print("=" * 60)
print("2. GLS with AR(1) correlation")
print("=" * 60)
result_ar1 = GLS.from_formula(
    "y ~ treatment + x",
    data=df,
    correlation=CorAR1(),
    groups="subject",
).fit()
print(result_ar1.summary())
print(f"\nEstimated phi: {result_ar1.correlation_params[0]:.3f} (true: {phi_true})")
print()

# ── 3. GLS with compound symmetry ────────────────────────────────────────────
print("=" * 60)
print("3. GLS with Compound Symmetry")
print("=" * 60)
result_cs = GLS.from_formula(
    "y ~ treatment + x",
    data=df,
    correlation=CorCompSymm(),
    groups="subject",
).fit()
print(result_cs.summary())
print()

# ── 4. Model comparison ──────────────────────────────────────────────────────
print("=" * 60)
print("4. Model comparison (AIC)")
print("=" * 60)
print(f"  OLS:              AIC = {result_ols.aic:.1f}")
print(f"  AR(1):            AIC = {result_ar1.aic:.1f}")
print(f"  Compound Symm:    AIC = {result_cs.aic:.1f}")
best = min([result_ols, result_ar1, result_cs], key=lambda r: r.aic)
print(f"  Best model: {'OLS' if best is result_ols else 'AR1' if best is result_ar1 else 'CS'}")
