"""Tests for correlation structures."""

import numpy as np
import pytest

from python_gls.correlation import (
    CorAR1,
    CorARMA,
    CorCAR1,
    CorCompSymm,
    CorSymm,
    CorExp,
    CorGaus,
    CorLin,
    CorRatio,
    CorSpher,
)
from python_gls._parametrization import (
    angles_to_corr,
    corr_to_angles,
    unconstrained_to_corr,
    corr_to_unconstrained,
)


class TestParametrization:
    """Test spherical parametrization roundtrips."""

    def test_roundtrip_2x2(self):
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        angles = corr_to_angles(R)
        R2 = angles_to_corr(angles, 2)
        np.testing.assert_allclose(R, R2, atol=1e-10)

    def test_roundtrip_3x3(self):
        R = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]])
        angles = corr_to_angles(R)
        R2 = angles_to_corr(angles, 3)
        np.testing.assert_allclose(R, R2, atol=1e-10)

    def test_unconstrained_roundtrip(self):
        R = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
        u = corr_to_unconstrained(R)
        R2 = unconstrained_to_corr(u, 3)
        np.testing.assert_allclose(R, R2, atol=1e-6)

    def test_positive_definite(self):
        """Random unconstrained params should always give PD matrix."""
        np.random.seed(42)
        for _ in range(20):
            u = np.random.randn(6)  # 4x4 matrix
            R = unconstrained_to_corr(u, 4)
            eigvals = np.linalg.eigvalsh(R)
            assert np.all(eigvals > 0), f"Not PD: {eigvals}"
            np.testing.assert_allclose(np.diag(R), 1.0)


class TestCorAR1:
    def test_known_matrix(self):
        cor = CorAR1(phi=0.5)
        R = cor.get_correlation_matrix(4)
        expected = np.array([
            [1.0, 0.5, 0.25, 0.125],
            [0.5, 1.0, 0.5, 0.25],
            [0.25, 0.5, 1.0, 0.5],
            [0.125, 0.25, 0.5, 1.0],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_identity_at_zero(self):
        cor = CorAR1(phi=0.0)
        R = cor.get_correlation_matrix(3)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_param_roundtrip(self):
        cor = CorAR1(phi=0.7)
        u = cor.get_unconstrained_params()
        cor2 = CorAR1()
        cor2._params = np.array([0.0])  # dummy init
        cor2.set_unconstrained_params(u)
        np.testing.assert_allclose(cor2.get_params(), [0.7], atol=1e-10)


class TestCorCompSymm:
    def test_known_matrix(self):
        cor = CorCompSymm(rho=0.3)
        R = cor.get_correlation_matrix(3)
        expected = np.array([
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_identity_at_zero(self):
        cor = CorCompSymm(rho=0.0)
        R = cor.get_correlation_matrix(3)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


class TestCorARMA:
    def test_ar1_special_case(self):
        """CorARMA(p=1, q=0) should approximate CorAR1."""
        arma = CorARMA(p=1, q=0)
        arma.set_params(np.array([0.5]))
        R_arma = arma.get_correlation_matrix(4)

        ar1 = CorAR1(phi=0.5)
        R_ar1 = ar1.get_correlation_matrix(4)

        np.testing.assert_allclose(R_arma, R_ar1, atol=0.05)

    def test_ma1(self):
        arma = CorARMA(p=0, q=1)
        arma.set_params(np.array([0.5]))
        R = arma.get_correlation_matrix(3)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.diag(R), 1.0)


class TestCorCAR1:
    def test_equal_spacing_matches_ar1(self):
        """With equal spacing, CAR1 should match AR1."""
        car = CorCAR1(phi=0.5)
        R_car = car.get_correlation_matrix(4)

        ar = CorAR1(phi=0.5)
        R_ar = ar.get_correlation_matrix(4)

        np.testing.assert_allclose(R_car, R_ar, atol=1e-10)

    def test_irregular_spacing(self):
        car = CorCAR1(phi=0.5)
        car.set_time_points(0, np.array([0, 1, 3, 7]))
        R = car.get_correlation_matrix(4, group_id=0)
        # R[0,2] = 0.5^3 = 0.125
        np.testing.assert_allclose(R[0, 2], 0.5**3, atol=1e-10)
        # R[0,3] = 0.5^7
        np.testing.assert_allclose(R[0, 3], 0.5**7, atol=1e-10)


class TestSpatialCorrelation:
    def test_cor_exp(self):
        cor = CorExp(range_param=2.0)
        cor.set_coordinates(0, np.array([0, 1, 2, 3]))
        R = cor.get_correlation_matrix(4, group_id=0)
        # R[0,1] = exp(-1/2)
        np.testing.assert_allclose(R[0, 1], np.exp(-0.5), atol=1e-10)

    def test_cor_gaus(self):
        cor = CorGaus(range_param=2.0)
        cor.set_coordinates(0, np.array([0, 1, 2]))
        R = cor.get_correlation_matrix(3, group_id=0)
        np.testing.assert_allclose(R[0, 1], np.exp(-0.25), atol=1e-10)

    def test_cor_lin(self):
        cor = CorLin(range_param=3.0)
        cor.set_coordinates(0, np.array([0, 1, 4]))
        R = cor.get_correlation_matrix(3, group_id=0)
        np.testing.assert_allclose(R[0, 1], 1 - 1/3, atol=1e-10)
        np.testing.assert_allclose(R[0, 2], 0.0, atol=1e-10)  # d >= range

    def test_cor_spher(self):
        cor = CorSpher(range_param=3.0)
        cor.set_coordinates(0, np.array([0, 1, 4]))
        R = cor.get_correlation_matrix(3, group_id=0)
        # d=1, range=3: 1 - 1.5*(1/3) + 0.5*(1/3)^3
        expected = 1 - 1.5 * (1/3) + 0.5 * (1/3)**3
        np.testing.assert_allclose(R[0, 1], expected, atol=1e-10)
        np.testing.assert_allclose(R[0, 2], 0.0, atol=1e-10)

    def test_all_positive_definite(self):
        """All spatial structures should produce PD matrices."""
        for CorClass in [CorExp, CorGaus, CorLin, CorRatio, CorSpher]:
            cor = CorClass(range_param=5.0)
            R = cor.get_correlation_matrix(5)
            eigvals = np.linalg.eigvalsh(R)
            assert np.all(eigvals > -1e-10), f"{CorClass.__name__} not PD"
