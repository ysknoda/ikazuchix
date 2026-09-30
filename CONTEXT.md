# ikazuchix

Point-defect formation energies in semiconductors, with the Kumagai–Oba finite-size charge correction, computed from DFT-derived numbers supplied by the user.

## Language

**Host**:
The perfect bulk compound in which defects are formed; its composition defines the host-stability equality on chemical potentials.
_Avoid_: bulk material, matrix

**Charge state**:
One integer charge q of a defect, with its own supercell total energy; negative q means extra electrons.
_Avoid_: charge, ionization state

**Elemental reference**:
The DFT total energy per atom of a species' stable elemental phase (e.g. fcc Al, ½ N2, ½ triplet O2), the zero of that species' Δμ.
_Avoid_: reference energy, atomic energy

**Chemical-potential point**:
One concrete vector of Δμ values (relative to elemental references) at which formation energies are evaluated.
_Avoid_: growth condition, chemical potential set

**Competing phase**:
A secondary compound whose formation bounds the chemical-potential point from above (e.g. Al2O3 for O in AlN).
_Avoid_: impurity phase, limiting phase

**Potential alignment (`delta_v`)**:
The raw far-from-defect electrostatic potential difference V_defect(far) − V_bulk(far) of one charge state; ikazuchix multiplies it by q without any model-potential subtraction.
_Avoid_: alignment correction, ΔV_PC

**Smoke test**:
An end-to-end run on real but unconverged small-cell DFT data that checks I/O and behaviour, never physical accuracy.
_Avoid_: benchmark, validation study
