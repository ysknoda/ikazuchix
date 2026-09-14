"""Unit tests for ikazuchix.geometry."""

import math

import numpy as np
import pytest

from ikazuchix.geometry import (
    average_dielectric,
    build_dielectric_tensor,
    build_lattice_matrix,
    lattice_volume,
    reciprocal_lattice,
)


def test_build_lattice_matrix():
    sc = {"lattice_vectors": {"a": [5, 0, 0], "b": [0, 6, 0], "c": [0, 0, 7]}}
    L = build_lattice_matrix(sc)
    np.testing.assert_array_equal(L, [[5, 0, 0], [0, 6, 0], [0, 0, 7]])


def test_lattice_volume_simple_cubic():
    L = np.eye(3) * 4.0
    assert lattice_volume(L) == pytest.approx(64.0)


def test_lattice_volume_is_positive_for_left_handed_cell():
    # A left-handed / negative-determinant cell still has a positive volume.
    L = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    assert lattice_volume(L) == pytest.approx(1.0)


def test_reciprocal_lattice_orthogonality():
    L = np.array([[9.57, 0.0, 0.0],
                  [-4.785, 8.2877, 0.0],
                  [0.0, 0.0, 10.38]])
    G = reciprocal_lattice(L)
    # L_i . G_j = 2*pi*delta_ij  (rows convention used throughout ikazuchix)
    product = L @ G.T
    np.testing.assert_allclose(product, 2.0 * math.pi * np.eye(3), atol=1e-10)


def test_average_dielectric_diagonal():
    eps = np.diag([9.5, 9.5, 10.4])
    assert average_dielectric(eps) == pytest.approx((9.5 + 9.5 + 10.4) / 3.0)


def test_build_dielectric_tensor_rejects_wrong_shape():
    with pytest.raises(ValueError):
        build_dielectric_tensor({"tensor": [[1, 0], [0, 1]]})


def test_build_dielectric_tensor_accepts_positive_definite():
    raw = [[9.5, 0, 0], [0, 9.5, 0], [0, 0, 10.4]]
    np.testing.assert_array_equal(
        build_dielectric_tensor({"tensor": raw}), raw)


def test_build_dielectric_tensor_rejects_indefinite_with_positive_trace_and_det():
    # trace = 3 > 0, det = 5 > 0, yet eigenvalues are -1, -1, 5: NOT
    # positive-definite. A trace/det-only check would miss this.
    with pytest.raises(ValueError, match="positive-definite"):
        build_dielectric_tensor(
            {"tensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 5]]})


def test_build_dielectric_tensor_rejects_non_symmetric():
    with pytest.raises(ValueError, match="symmetric"):
        build_dielectric_tensor(
            {"tensor": [[9.5, 1.0, 0], [0.0, 9.5, 0], [0, 0, 10.4]]})


def test_build_lattice_matrix_rejects_degenerate_lattice():
    # b = 2*a: linearly dependent, zero-volume cell.
    sc = {"lattice_vectors": {"a": [5, 0, 0], "b": [10, 0, 0], "c": [0, 0, 7]}}
    with pytest.raises(ValueError, match="Degenerate"):
        build_lattice_matrix(sc)
