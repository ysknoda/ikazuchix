"""Enables ``python -m ikazuchix ...`` as the CLI entry point.

ikazuchix -- Defect Formation Energy Calculator
=================================================
Calculates defect formation energies in semiconductors from DFT total
energies, applying the Kumagai-Oba (KO) finite-size charge correction:

  Y. Kumagai and F. Oba, Phys. Rev. B 89, 195205 (2014).

Run with -h or --help for full usage information.
"""

from ikazuchix.main import main

if __name__ == "__main__":
    main()
