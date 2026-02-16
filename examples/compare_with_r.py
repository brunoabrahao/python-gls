"""Side-by-side comparison of python_gls with R's nlme::gls.

Run this script, then compare with the equivalent R code below.

R equivalent:
    library(nlme)
    set.seed(42)
    n <- 30; T <- 4; N <- n * T
    subject <- rep(1:n, each=T)
    x <- rnorm(N)
    y <- 2 + 1.5*x + rnorm(N, sd=0.5)
    df <- data.frame(y=y, x=x, subject=factor(subject))

    # OLS
    m0 <- gls(y ~ x, data=df)
    summary(m0)

    # AR1
    m1 <- gls(y ~ x, data=df, correlation=corAR1(form=~1|subject))
    summary(m1)

    # Compound Symmetry
    m2 <- gls(y ~ x, data=df, correlation=corCompSymm(form=~1|subject))
    summary(m2)
"""

import numpy as np
import pandas as pd

from python_gls import GLS
from python_gls.correlation import CorAR1, CorCompSymm

# Generate data (matching R's set.seed(42) output is not possible,
# but the structure is the same)
np.random.seed(42)
n, T = 30, 4
N = n * T
subject = np.repeat(np.arange(n), T)
x = np.random.randn(N)
y = 2.0 + 1.5 * x + np.random.randn(N) * 0.5

df = pd.DataFrame({"y": y, "x": x, "subject": subject})

print("Python GLS Results")
print("=" * 60)

# OLS
print("\n--- OLS ---")
r0 = GLS.from_formula("y ~ x", data=df, method="ML").fit()
print(f"  Intercept: {r0.params['Intercept']:.4f} (SE={r0.bse['Intercept']:.4f})")
print(f"  x:         {r0.params['x']:.4f} (SE={r0.bse['x']:.4f})")
print(f"  sigma:     {np.sqrt(r0.sigma2):.4f}")
print(f"  AIC:       {r0.aic:.2f}")

# AR1
print("\n--- GLS + AR(1) ---")
r1 = GLS.from_formula("y ~ x", data=df, correlation=CorAR1(), groups="subject", method="ML").fit()
print(f"  Intercept: {r1.params['Intercept']:.4f} (SE={r1.bse['Intercept']:.4f})")
print(f"  x:         {r1.params['x']:.4f} (SE={r1.bse['x']:.4f})")
print(f"  phi:       {r1.correlation_params[0]:.4f}")
print(f"  sigma:     {np.sqrt(r1.sigma2):.4f}")
print(f"  AIC:       {r1.aic:.2f}")

# Compound Symmetry
print("\n--- GLS + Compound Symmetry ---")
r2 = GLS.from_formula("y ~ x", data=df, correlation=CorCompSymm(), groups="subject", method="ML").fit()
print(f"  Intercept: {r2.params['Intercept']:.4f} (SE={r2.bse['Intercept']:.4f})")
print(f"  x:         {r2.params['x']:.4f} (SE={r2.bse['x']:.4f})")
print(f"  rho:       {r2.correlation_params[0]:.4f}")
print(f"  sigma:     {np.sqrt(r2.sigma2):.4f}")
print(f"  AIC:       {r2.aic:.2f}")
