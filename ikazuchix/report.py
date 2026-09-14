"""Plain-text multi-defect report formatting."""

import math

from ikazuchix.geometry import average_dielectric, build_dielectric_tensor, build_lattice_matrix, lattice_volume


# ===========================================================================
# Text report
# ===========================================================================

def format_report(data, all_results, fermi_ref, ewald_eta_override=None):
    """Build the multi-defect plain-text report.

    ewald_eta_override: the value of --ewald-eta, if the user gave one
    explicitly (None -- the default -- reproduces the original,
    always-auto-heuristic display exactly). Without this, the report
    always showed the auto-heuristic eta = sqrt(pi)/V^(1/3), mislabeled
    as if it were the actual value used, even when --ewald-eta had
    overridden it for the calculation itself.
    """
    sep  = "=" * 65
    sep2 = "-" * 65
    bs   = data["band_structure"]
    eps  = build_dielectric_tensor(data["dielectric"])
    gap  = bs["cbm"] - bs["vbm"]

    lines = [
        sep,
        "  DEFECT FORMATION ENERGY REPORT",
        "  Method: Kumagai-Oba (PRB 89, 195205, 2014)",
        sep, "",
        "[System]",
        f"  Host material      : {data['bulk']['formula']}",
        f"  Defects computed   : {[r['name'] for r in all_results]}",
        "",
        "[Band Structure]",
        f"  VBM                : {bs['vbm']:.4f} eV",
        f"  CBM                : {bs['cbm']:.4f} eV",
        f"  Band gap           : {bs['bandgap']:.4f} eV",
        f"  Fermi level (ref)  : {fermi_ref:.4f} eV  (relative to VBM)",
        f"  Fermi sweep range  : [{all_results[0]['fermi_grid'][0]:.4f}, "
                               f"{all_results[0]['fermi_grid'][-1]:.4f}] eV",
        f"  Fermi grid points  : {len(all_results[0]['fermi_grid'])}",
        "",
        "[Dielectric]",
        f"  Scalar average eps : {average_dielectric(eps):.4f}",
        f"  Full tensor        : {eps.tolist()}",
        "",
        "[Supercell Geometry]",
    ]
    # D-5: show supercell lattice and Ewald parameter for each defect
    sc_dict_top = data.get("supercell", {})
    if sc_dict_top:
        L_top = build_lattice_matrix(sc_dict_top)
        V_top = lattice_volume(L_top)
        if ewald_eta_override is not None:
            eta_top, eta_label = float(ewald_eta_override), "Ewald eta (--ewald-eta)"
        else:
            eta_top = math.sqrt(math.pi) / (V_top ** (1.0 / 3.0))
            eta_label = "Ewald eta (auto)"
        lines += [
            f"  Lattice a          : {sc_dict_top['lattice_vectors']['a']}",
            f"  Lattice b          : {sc_dict_top['lattice_vectors']['b']}",
            f"  Lattice c          : {sc_dict_top['lattice_vectors']['c']}",
            f"  Volume             : {V_top:.4f} Ang^3",
            f"  {eta_label:20s} : {eta_top:.6f} Ang^-1",
        ]
    lines += [""]

    for res in all_results:
        lines += [
            "", sep,
            f"  DEFECT: {res['name']}  ({res['defect_type']})",
            sep,
            f"  Charge states : {res['charges']}",
            "",
            "  [KO Correction Breakdown]",
        ]
        for q in res["charges"]:
            ko = res["ko"][q]
            lines += [
                f"    q = {q:+d}",
                f"      E_real        : {ko['E_real']:+.6f} eV",
                f"      E_recip       : {ko['E_recip']:+.6f} eV",
                f"      E_self(Ewald) : {ko['E_self_ewald']:+.6f} eV",
                f"      E_neutralize  : {ko['E_neutralize']:+.6f} eV",
                f"      E_lat         : {ko['E_lat']:+.6f} eV",
                f"      E_alignment   : {ko['E_alignment']:+.6f} eV",
                f"      E_KO (total)  : {ko['E_KO']:+.6f} eV",
                f"      Chem.pot.corr.: {res['chem_pot_corr'][q]:+.6f} eV",
                f"      E_f(E_F={fermi_ref:.3f} eV) : "
                f"{res['ef_at_fermi'][q]:+.6f} eV",
            ]

        lines += ["", "  [Stable CTLs  (on the envelope, within band gap)]"]
        if res["stable_ctls"]:
            for ctl in sorted(res["stable_ctls"], key=lambda x: x["E_F_CTL"]):
                lines.append(
                    f"    eps({ctl['q1']:+d}/{ctl['q2']:+d})  at  "
                    f"E_F = {ctl['E_F_CTL']:+.4f} eV  "
                    f"[E_f = {ctl['E_f_CTL']:+.4f} eV]")
        else:
            lines.append("    None within the band gap.")

        if res["meta_ctls"]:
            lines += ["", "  [Meta-stable CTLs  (off envelope or outside gap)]"]
            for ctl in sorted(res["meta_ctls"], key=lambda x: x["E_F_CTL"]):
                lines.append(
                    f"    eps({ctl['q1']:+d}/{ctl['q2']:+d})  at  "
                    f"E_F = {ctl['E_F_CTL']:+.4f} eV  "
                    f"[E_f = {ctl['E_f_CTL']:+.4f} eV]")

    lines += [
        "", sep2,
        "  REFERENCE",
        sep2,
        "  Y. Kumagai and F. Oba, Phys. Rev. B 89, 195205 (2014).",
        sep,
    ]
    return lines
