"""Tests for variance functions."""

import numpy as np
import pytest

from python_gls.variance import (
    VarIdent,
    VarPower,
    VarExp,
    VarConstPower,
    VarFixed,
    VarComb,
)


class TestVarIdent:
    def test_equal_groups(self):
        vi = VarIdent("group")
        data = {"group": np.array(["A", "A", "B", "B", "C", "C"])}
        residuals = np.array([1, 2, 3, 4, 5, 6])
        vi.initialize(residuals, data)
        # After init, weights should be computed
        w = vi.get_weights(data, np.array([0, 1]))  # group A (reference)
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_different_groups(self):
        vi = VarIdent("g")
        data = {"g": np.array(["X", "X", "Y", "Y"])}
        vi._levels = ["X", "Y"]
        vi._ref_level = "X"
        vi.set_params(np.array([0.5]))  # log ratio
        w = vi.get_weights(data, np.array([2, 3]))
        np.testing.assert_allclose(w, [np.exp(0.5), np.exp(0.5)])


class TestVarPower:
    def test_known_weights(self):
        vp = VarPower("x")
        vp.set_params(np.array([2.0]))
        data = {"x": np.array([1.0, 2.0, 3.0])}
        w = vp.get_weights(data, np.array([0, 1, 2]))
        np.testing.assert_allclose(w, [1.0, 4.0, 9.0])

    def test_zero_power(self):
        vp = VarPower("x")
        vp.set_params(np.array([0.0]))
        data = {"x": np.array([1.0, 5.0, 10.0])}
        w = vp.get_weights(data, np.array([0, 1, 2]))
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0])


class TestVarExp:
    def test_known_weights(self):
        ve = VarExp("x")
        ve.set_params(np.array([0.5]))
        data = {"x": np.array([0.0, 1.0, 2.0])}
        w = ve.get_weights(data, np.array([0, 1, 2]))
        np.testing.assert_allclose(w, [1.0, np.exp(0.5), np.exp(1.0)])


class TestVarConstPower:
    def test_known_weights(self):
        vc = VarConstPower("x")
        vc.set_params(np.array([0.0, 1.0]))  # c=exp(0)=1, delta=1
        data = {"x": np.array([1.0, 2.0, 3.0])}
        w = vc.get_weights(data, np.array([0, 1, 2]))
        # c + |x|^delta = 1 + x
        np.testing.assert_allclose(w, [2.0, 3.0, 4.0])


class TestVarFixed:
    def test_fixed_weights(self):
        vf = VarFixed("w")
        data = {"w": np.array([1.0, 4.0, 9.0])}
        vf.initialize(np.array([0, 0, 0]), data)
        w = vf.get_weights(data, np.array([0, 1, 2]))
        np.testing.assert_allclose(w, [1.0, 2.0, 3.0])  # sqrt of weights

    def test_no_params(self):
        vf = VarFixed("w")
        assert vf.n_params == 0


class TestVarComb:
    def test_product_of_weights(self):
        vp = VarPower("x")
        vp.set_params(np.array([1.0]))
        ve = VarExp("z")
        ve.set_params(np.array([1.0]))

        vc = VarComb(vp, ve)
        data = {"x": np.array([2.0, 3.0]), "z": np.array([0.0, 1.0])}
        w = vc.get_weights(data, np.array([0, 1]))
        expected = np.array([2.0 * 1.0, 3.0 * np.exp(1.0)])
        np.testing.assert_allclose(w, expected)

    def test_n_params(self):
        vc = VarComb(VarPower("x"), VarExp("z"))
        assert vc.n_params == 2

    def test_requires_two(self):
        with pytest.raises(ValueError):
            VarComb(VarPower("x"))
