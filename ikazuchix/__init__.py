"""ikazuchix -- Defect Formation Energy Calculator package.

Implements the Kumagai-Oba (KO) finite-size charge correction for
semiconductor point-defect formation energies:

  Y. Kumagai and F. Oba, Phys. Rev. B 89, 195205 (2014).

Run as a CLI via ``python -m ikazuchix``; see its ``--help`` output for
usage (also documented in the project README).
"""

import importlib.metadata

from ikazuchix.constants import (
    BOHR_RADIUS_ANG,
    COULOMB_PREFACTOR,
    EV_PER_HARTREE,
    EWALD_CONV_TOL,
    N_FERMI_POINTS,
)
from ikazuchix.ewald import KumagaiObaCorrectionCalculator
from ikazuchix.formation_energy import formation_energy
from ikazuchix.io_utils import load_input

try:
    __version__ = importlib.metadata.version("ikazuchix")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0a2"  # not installed (e.g. run from a source checkout)

__all__ = [
    "__version__",
    "BOHR_RADIUS_ANG",
    "COULOMB_PREFACTOR",
    "EV_PER_HARTREE",
    "EWALD_CONV_TOL",
    "N_FERMI_POINTS",
    "KumagaiObaCorrectionCalculator",
    "formation_energy",
    "load_input",
]
