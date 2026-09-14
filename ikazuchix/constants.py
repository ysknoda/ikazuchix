"""Physical constants and global numerical settings."""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EV_PER_HARTREE = 27.211386245988  # 1 Hartree in eV
BOHR_RADIUS_ANG = 0.529177210903  # Bohr radius in Angstrom
COULOMB_PREFACTOR = EV_PER_HARTREE * BOHR_RADIUS_ANG  # e^2/(4*pi*eps_0) ~ 14.3996 eV*Ang

# CODATA/SI values, used by ikazuchix/equilibrium.py (Feat-1) to derive
# the effective density of states from first principles rather than a
# hardcoded reference value at a specific temperature.
ELECTRON_MASS_KG = 9.1093837015e-31    # free electron mass m_0 [kg]
PLANCK_CONSTANT_J_S = 6.62607015e-34   # Planck constant h [J s] (exact, SI 2019)
BOLTZMANN_J_PER_K = 1.380649e-23       # Boltzmann constant k_B [J/K] (exact, SI 2019)
BOLTZMANN_EV_PER_K = 8.617333262e-5    # k_B [eV/K] (k_B[J/K] / elementary charge [C])

N_FERMI_POINTS = 400  # Fermi-level grid density
EWALD_CONV_TOL = 1e-6  # convergence threshold for Ewald sum shell check (eV)
