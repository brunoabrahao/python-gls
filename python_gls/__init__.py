"""python_gls: GLS with learned correlation and variance structures.

Python equivalent of R's nlme::gls(). Estimates Generalized Least Squares
models where the correlation and variance structures are learned from data
via ML/REML, not pre-specified.

Basic usage::

    from python_gls import GLS
    from python_gls.correlation import CorAR1
    from python_gls.variance import VarIdent

    result = GLS.from_formula(
        "y ~ x1 + x2",
        data=df,
        correlation=CorAR1(),
        variance=VarIdent("group"),
        groups="subject",
    ).fit()

    print(result.summary())
"""

from python_gls.model import GLS
from python_gls.results import GLSResults

__version__ = "0.1.0"

__all__ = ["GLS", "GLSResults"]
