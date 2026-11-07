"""Tests of the validation helpers, plus a small end-to-end run of the solver when it is built."""
from __future__ import annotations

import numpy as np
import pytest

from eddy_validation.common import Fields, binary, ghia, order, run
from eddy_validation.validate import duct_exact


def test_order_of_a_second_order_sequence():
    assert order([4e-3, 1e-3, 2.5e-4], [8, 16, 32]) == pytest.approx([2.0, 2.0])


def test_ghia_tables_have_their_rows():
    u, v = ghia("u"), ghia("v")
    assert set(u) == {100, 400, 1000}
    assert all(len(c) == 17 for c, _ in u.values())
    assert len(v[100][0]) == 17 and len(v[400][0]) == 16  # one point of Re 400 left out
    # No slip at the walls, the moving lid at the top.
    c, vel = u[100]
    assert vel[np.argmin(c)] == pytest.approx(0) and vel[np.argmax(c)] == pytest.approx(1)


def test_duct_series_is_symmetric_and_matches_the_known_centre_value():
    n, force, nu = 40, 1e-6, 0.1
    u = duct_exact(n, force, nu)
    assert np.allclose(u, u.T) and np.allclose(u, u[::-1])
    # Centre-line velocity of a square duct: 0.2947 * g a^2 / nu (a: half-width).
    a = n / 2
    assert u.max() == pytest.approx(0.2947 * force * a * a / nu, rel=2e-3)


def built() -> bool:
    try:
        binary()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not built(), reason="the solver is not built")
def test_channel_flow_matches_the_parabola(tmp_path):
    n = 12
    s = run("poiseuille", tmp_path, n=n, tau=0.8)
    f = Fields(tmp_path / "fields.bin")
    yy = np.arange(1, n + 1) - 0.5
    exact = s["force"] * yy * (n - yy) / (2 * s["viscosity"])
    got = f.u[0, 1 : n + 1, 1, 0]
    assert np.max(np.abs(got / exact - 1)) < 1e-3


@pytest.mark.skipif(not built(), reason="the solver is not built")
def test_cavity_is_close_to_ghia(tmp_path):
    run("cavity", tmp_path, n=48, re=100)
    f = Fields(tmp_path / "fields.bin")
    n = f.nx - 2
    # The vertical centre line: u / U against y.
    x = n // 2
    u = 0.5 * (f.u[0, 1 : n + 1, x, 0] + f.u[0, 1 : n + 1, x + 1, 0]) / 0.1
    y = (np.arange(n) + 0.5) / n
    c, ref = ghia("u")[100]
    inner = (c > 0) & (c < 1)
    assert np.max(np.abs(np.interp(c[inner], y, u) - ref[inner])) < 0.02
