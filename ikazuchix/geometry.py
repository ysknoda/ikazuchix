"""Lattice and dielectric-tensor geometry helpers."""

import math

import numpy as np


_MIN_LATTICE_VOLUME_ANG3 = 1e-6


def build_lattice_matrix(supercell_dict):
    lv = supercell_dict["lattice_vectors"]
    L = np.array([lv["a"], lv["b"], lv["c"]], dtype=float)
    vol = abs(np.linalg.det(L))
    if vol < _MIN_LATTICE_VOLUME_ANG3:
        raise ValueError(
            f"Degenerate (near-zero-volume) lattice: "
            f"|det(lattice_vectors)| = {vol:.3e} Ang^3.  Check for "
            "duplicate or near-parallel lattice vectors.")
    return L


def lattice_volume(L):
    return abs(np.linalg.det(L))


def reciprocal_lattice(L):
    """Reciprocal lattice matrix (rows = G-vectors, 2pi convention) [Ang^-1]."""
    return 2.0 * math.pi * np.linalg.inv(L).T


def build_dielectric_tensor(dielectric_dict):
    tensor = np.array(dielectric_dict["tensor"], dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError("Dielectric tensor must be 3x3.")
    if not np.allclose(tensor, tensor.T):
        raise ValueError("Dielectric tensor must be symmetric.")
    eigvals = np.linalg.eigvalsh(tensor)
    if np.any(eigvals <= 0):
        raise ValueError(
            f"Dielectric tensor must be positive-definite (eigenvalues: "
            f"{eigvals}).  A trace>0/det>0 check alone would miss "
            "indefinite patterns like diag(-1, -1, 5), which also "
            "satisfy those two necessary-but-insufficient conditions.")
    return tensor


def average_dielectric(eps):
    return float(np.trace(eps) / 3.0)
