# python-gls

**GLS with learned correlation and variance structures for Python.**

The missing Python equivalent of R's `nlme::gls()`. Unlike `statsmodels.GLS` (which requires you to supply a pre-computed covariance matrix), `python-gls` *estimates* the correlation and variance structure from your data via maximum likelihood (ML) or restricted maximum likelihood (REML) — exactly like R's `nlme::gls()`.

## Why this library?

If you work with **panel data**, **repeated measures**, **longitudinal studies**, or **clustered observations**, your errors are probably correlated and possibly heteroscedastic. Ignoring this gives you wrong standard errors and misleading p-values.

R has had `nlme::gls()` for 25+ years. Python hasn't had an equivalent. Until now.

| Feature | `statsmodels.GLS` | `python-gls` | R `nlme::gls` |
|---|---|---|---|
| Estimate correlation from data | No (manual Omega) | Yes | Yes |
| AR(1), compound symmetry, etc. | No | Yes (11 structures) | Yes |
| Heteroscedastic variance models | No | Yes (6 functions) | Yes |
| ML / REML estimation | No | Yes | Yes |
| R-style formulas | No | Yes | Yes |

## Installation

```bash
pip install python-gls
```

Or from source:

```bash
git clone https://github.com/brunoabrahao/python-gls.git
cd python-gls
pip install -e ".[dev]"
```

## Quick Start

```python
from python_gls import GLS
from python_gls.correlation import CorAR1
from python_gls.variance import VarIdent

result = GLS.from_formula(
    "response ~ treatment + time",
    data=df,
    correlation=CorAR1(),       # Learn AR(1) correlation
    variance=VarIdent("group"),  # Learn group-specific variances
    groups="subject",            # Define independent clusters
).fit()

print(result.summary())
print(f"Estimated AR(1) phi: {result.correlation_params[0]:.3f}")
```

Output:

```
==============================================================================
                      Generalized Least Squares Results
==============================================================================
Method:          REML                 Log-Likelihood:        -615.0544
No. Observations:500                  AIC:                   1240.1088
Df Model:        2                    BIC:                   1261.1818
Df Residuals:    497                  Sigma^2:                0.984576
Converged:       Yes                  Iterations:                    6
------------------------------------------------------------------------------
                           coef    std err          t      P>|t|     [0.025     0.975]
------------------------------------------------------------------------------
           Intercept     1.0368     0.1069     9.7013     0.0000     0.8268     1.2468
           treatment     0.6465     0.1428     4.5272     0.0000     0.3659     0.9271
                   x     1.9734     0.0323    61.0960     0.0000     1.9099     2.0368
==============================================================================
Correlation Structure: CorAR1
  Parameters: [0.61312872]
```

## Correlation Structures

All correlation structures are in `python_gls.correlation`.

### Temporal / Serial Correlation

| Class | R Equivalent | Parameters | Description |
|---|---|---|---|
| `CorAR1(phi=None)` | `corAR1()` | 1 | First-order autoregressive. R[i,j] = phi^&#124;i-j&#124; |
| `CorARMA(p, q)` | `corARMA(p, q)` | p + q | ARMA(p,q) autocorrelation |
| `CorCAR1(phi=None)` | `corCAR1()` | 1 | Continuous-time AR(1) for irregular spacing |
| `CorCompSymm(rho=None)` | `corCompSymm()` | 1 | Exchangeable / compound symmetry. All pairs equal rho |
| `CorSymm(dim=None)` | `corSymm()` | d(d-1)/2 | General unstructured. Free correlation for every pair |

### Spatial Correlation

| Class | R Equivalent | Parameters | Description |
|---|---|---|---|
| `CorExp(range_param, nugget=False)` | `corExp()` | 1-2 | Exponential: exp(-d/range) |
| `CorGaus(range_param, nugget=False)` | `corGaus()` | 1-2 | Gaussian: exp(-(d/range)^2) |
| `CorLin(range_param, nugget=False)` | `corLin()` | 1-2 | Linear: max(1 - d/range, 0) |
| `CorRatio(range_param, nugget=False)` | `corRatio()` | 1-2 | Rational quadratic: 1/(1 + (d/range)^2) |
| `CorSpher(range_param, nugget=False)` | `corSpher()` | 1-2 | Spherical: cubic polynomial, zero beyond range |

All spatial structures accept an optional `nugget=True` parameter for a discontinuity at distance zero.

### Usage

```python
from python_gls.correlation import CorAR1, CorSymm, CorExp

# Serial: AR(1) with optional initial value
cor = CorAR1(phi=0.5)

# Unstructured: all pairs free
cor = CorSymm()  # dimension inferred from data

# Spatial: set coordinates per group
cor = CorExp(range_param=10.0, nugget=True)
cor.set_coordinates(group_id=0, coords=np.array([[0,0], [1,0], [0,1]]))
```

## Variance Functions

All variance functions are in `python_gls.variance`.

| Class | R Equivalent | Parameters | Description |
|---|---|---|---|
| `VarIdent(group_var)` | `varIdent(form=~1\|group)` | G-1 | Different variance per group level |
| `VarPower(covariate)` | `varPower(form=~cov)` | 1 | sd = &#124;v&#124;^delta |
| `VarExp(covariate)` | `varExp(form=~cov)` | 1 | sd = exp(delta * v) |
| `VarConstPower(covariate)` | `varConstPower(form=~cov)` | 2 | sd = (c + &#124;v&#124;^delta) |
| `VarFixed(weights_var)` | `varFixed(~cov)` | 0 | Pre-specified weights (not estimated) |
| `VarComb(*varfuncs)` | `varComb(...)` | sum | Product of multiple variance functions |

### Usage

```python
from python_gls.variance import VarIdent, VarPower, VarComb

# Different variance for treatment vs. control
var = VarIdent("treatment_group")

# Variance increases with fitted values
var = VarPower("fitted_values")

# Combine: group-specific + covariate-dependent
var = VarComb(VarIdent("group"), VarPower("x"))
```

## API Reference

### `GLS` Class

#### Construction

```python
# From formula (recommended)
model = GLS.from_formula(
    formula,          # R-style formula: "y ~ x1 + x2"
    data,             # pandas DataFrame
    correlation=None, # CorStruct instance
    variance=None,    # VarFunc instance
    groups=None,      # str: column name for groups
    method="REML",    # "ML" or "REML"
)

# From arrays
model = GLS(
    endog=y,          # response vector
    exog=X,           # design matrix (include intercept column)
    correlation=None,
    variance=None,
    groups=None,      # array of group labels
    method="REML",
)
```

#### Fitting

```python
result = model.fit(
    maxiter=200,     # max optimization iterations
    tol=1e-8,        # convergence tolerance
    verbose=False,   # print optimization progress
)
```

### `GLSResults` Class

| Property / Method | Type | Description |
|---|---|---|
| `params` | Series | Estimated coefficients |
| `bse` | Series | Standard errors |
| `tvalues` | Series | t-statistics |
| `pvalues` | Series | Two-sided p-values |
| `conf_int(alpha=0.05)` | DataFrame | Confidence intervals |
| `sigma2` | float | Estimated residual variance |
| `loglik` | float | Log-likelihood at convergence |
| `aic` | float | Akaike Information Criterion |
| `bic` | float | Bayesian Information Criterion |
| `resid` | array | Residuals (y - X*beta) |
| `fittedvalues` | array | Fitted values (X*beta) |
| `correlation_params` | array | Estimated correlation parameters |
| `variance_params` | array | Estimated variance parameters |
| `cov_params_func()` | DataFrame | Covariance matrix of beta |
| `summary()` | str | Formatted results table |
| `converged` | bool | Optimization convergence status |
| `n_iter` | int | Number of iterations |
| `method` | str | "ML" or "REML" |

## How It Works

### The Statistical Model

GLS models the response as:

**y = X*beta + epsilon**, where **Var(epsilon) = sigma^2 * Omega**

The covariance matrix Omega is block-diagonal by group:

**Omega_g = A_g^{1/2} R_g A_g^{1/2}**

where:
- **R_g** is the correlation matrix (from the correlation structure)
- **A_g** is a diagonal matrix of variance weights (from the variance function)

### Estimation

1. **OLS initial fit** to get starting residuals
2. **Initialize** correlation and variance parameters from residuals
3. **Optimize** profile log-likelihood over correlation/variance parameters using L-BFGS-B. At each step, beta and sigma^2 are profiled out analytically.
4. **Compute** final GLS estimates at the converged parameters:
   - beta = (X' Omega^{-1} X)^{-1} X' Omega^{-1} y
   - Cov(beta) = sigma^2 (X' Omega^{-1} X)^{-1}

### Key Design Decisions

**Spherical parametrization** for `CorSymm`: The unstructured correlation matrix is parametrized via angles that map to a Cholesky factor, guaranteeing positive-definiteness without constrained optimization. Based on [Pinheiro & Bates (1996)](https://doi.org/10.1007/BF00140873).

**Block-diagonal inversion**: Omega is inverted per-group (O(n*m^3)) rather than as a full matrix (O(N^3)), where n = number of groups and m = group size.

**REML**: Restricted maximum likelihood integrates out the fixed effects from the likelihood, giving unbiased variance estimates. This is the default, matching R's `nlme::gls()`.

## Formula Syntax

Powered by [formulaic](https://github.com/matthewwardrop/formulaic), supporting:

```python
# Simple linear
"y ~ x1 + x2"

# Categorical variables
"y ~ C(treatment)"

# Interactions
"y ~ x1 * x2"          # x1 + x2 + x1:x2
"y ~ x1 : x2"          # just the interaction

# Transformations
"y ~ np.log(x1) + x2"

# Remove intercept
"y ~ x1 + x2 - 1"
```

## ML vs. REML

| | ML | REML |
|---|---|---|
| Variance estimate | Biased (divides by N) | Unbiased (divides by N-k) |
| Default in R's gls | No | Yes |
| Default here | No | Yes |
| Use for model comparison | AIC/BIC of nested & non-nested models | Only models with same fixed effects |
| `method=` | `"ML"` | `"REML"` |

## Translating from R

### R code → Python equivalent

```r
# R
library(nlme)
m <- gls(y ~ x1 + x2,
         data = df,
         correlation = corAR1(form = ~1|subject),
         weights = varIdent(form = ~1|group),
         method = "REML")
summary(m)
intervals(m)
```

```python
# Python
from python_gls import GLS
from python_gls.correlation import CorAR1
from python_gls.variance import VarIdent

r = GLS.from_formula(
    "y ~ x1 + x2",
    data=df,
    correlation=CorAR1(),
    variance=VarIdent("group"),
    groups="subject",
    method="REML",
).fit()

print(r.summary())
print(r.conf_int())
```

### Parameter name mapping

| R | Python | Notes |
|---|---|---|
| `corAR1(form=~1\|subject)` | `CorAR1(), groups="subject"` | Groups specified at model level |
| `corCompSymm(form=~1\|id)` | `CorCompSymm(), groups="id"` | |
| `corSymm(form=~1\|id)` | `CorSymm(), groups="id"` | |
| `corExp(form=~x+y\|id)` | `CorExp(); cor.set_coordinates(...)` | Coordinates set per group |
| `varIdent(form=~1\|group)` | `VarIdent("group")` | Group variable as string |
| `varPower(form=~fitted)` | `VarPower("fitted")` | Covariate name as string |
| `method="REML"` | `method="REML"` | Same |

## Dependencies

- **numpy** >= 1.24
- **scipy** >= 1.10
- **pandas** >= 2.0
- **formulaic** >= 1.0

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
