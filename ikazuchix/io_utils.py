"""YAML input loading and plain-text report writers."""

import os
import tempfile
import warnings

import yaml


# ===========================================================================
# I/O helpers
# ===========================================================================

_REQUIRED_TOP_SECTIONS = (
    "bulk", "chemical_potentials", "elemental_references",
    "band_structure", "dielectric", "supercell",
)


def load_input(path):
    """
    Load the YAML input file and normalise to the multi-defect layout.

    Accepted layouts
    ----------------
    Multi-defect  : top-level key 'defects' is a list of defect blocks.
    Single-defect : legacy keys 'defect' + 'charge_states' at top level.
                    Normalised into the multi-defect format *before* any
                    validation runs, so both layouts are checked by the
                    same rules (e.g. a missing 'defect_type' raises the
                    same way in either layout).

    Auto-completion
    ---------------
    If a charge_state entry omits 'composition', it is derived from the
    defect's removed_atoms / added_atoms relative to the bulk composition.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Input file '{path}' did not parse to a YAML mapping "
            f"(got {type(raw).__name__}).  Check that the file contains "
            "top-level 'key: value' sections.")

    for key in _REQUIRED_TOP_SECTIONS:
        if key not in raw:
            raise KeyError(f"Missing required section '{key}' in input file.")

    # --- normalise to multi-defect layout, *then* validate uniformly --------
    has_multi  = "defects" in raw
    has_legacy = "defect" in raw and "charge_states" in raw
    if has_multi and has_legacy:
        warnings.warn(
            "[Input] Both 'defects' (multi-defect format) and "
            "'defect'+'charge_states' (legacy format) are present; the "
            "legacy keys are ignored.  Remove one of the two to avoid "
            "ambiguity.",
            UserWarning, stacklevel=2)
    elif has_legacy:
        defect_block = dict(raw["defect"])
        defect_block["charge_states"] = raw["charge_states"]
        raw["defects"] = [defect_block]
    elif not has_multi:
        raise KeyError(
            "Input file must contain either a 'defects' list "
            "(multi-defect format) or both 'defect' and 'charge_states' "
            "(legacy single-defect format).")

    if not raw["defects"]:
        raise ValueError("'defects' list must contain at least one entry.")
    for d in raw["defects"]:
        if not isinstance(d, dict):
            raise ValueError(
                f"Each entry in 'defects' must be a mapping "
                f"(got {type(d).__name__}).")
        for k in ("name", "defect_type", "charge_states"):
            if k not in d:
                raise KeyError(
                    f"Defect entry is missing required key '{k}'.")
        if not isinstance(d["charge_states"], list):
            raise ValueError(
                f"Defect '{d['name']}': 'charge_states' must be a list.")
        if not d["charge_states"]:
            raise ValueError(
                f"Defect '{d['name']}' has an empty 'charge_states' list.")
        for cs in d["charge_states"]:
            context = f"Defect '{d['name']}'"
            _validate_and_normalize_charge(cs, context)
            cs_context = f"{context} q={cs['charge']:+d}"
            _validate_potential_alignment(cs, cs_context)
            _validate_optional_positive_number(cs, "degeneracy", cs_context)
            if "energy" not in cs:
                raise KeyError(
                    f"{cs_context}: charge_state is missing required "
                    "key 'energy'.")
            if not _is_number(cs["energy"]):
                raise ValueError(
                    f"{cs_context}: 'energy' must be a numeric value "
                    f"(got {cs['energy']!r}).")
            if cs.get("composition") is not None:
                if not isinstance(cs["composition"], dict):
                    raise ValueError(
                        f"{cs_context}: 'composition' must be a mapping "
                        "of {species: count, ...} if given "
                        f"(got {type(cs['composition']).__name__}).")
                _validate_composition_values(
                    cs["composition"], f"{cs_context} composition")
        _check_duplicate_charges(d)

    _validate_nested_required_keys(raw)

    # --- auto-complete composition for each charge_state --------------------
    bulk_comp = raw["bulk"]["composition"]
    for d in raw["defects"]:
        removed = d.get("removed_atoms", {}) or {}
        added   = d.get("added_atoms",   {}) or {}
        _validate_composition_values(
            removed, f"Defect '{d['name']}' removed_atoms")
        _validate_composition_values(
            added, f"Defect '{d['name']}' added_atoms")
        # sorted(), not set(): auto_comp's key order becomes the
        # charge state's composition order, which chem_pot_correction()
        # then accumulates floats over -- see its own comment. A set's
        # str iteration order varies between processes (randomized
        # string hashing), which would make the result differ in the
        # last ULP from run to run.
        all_sp  = sorted(set(bulk_comp) | set(removed) | set(added))
        auto_comp = {
            sp: bulk_comp.get(sp, 0) - removed.get(sp, 0) + added.get(sp, 0)
            for sp in all_sp if bulk_comp.get(sp, 0)
                                 - removed.get(sp, 0)
                                 + added.get(sp, 0) != 0
        }
        negative = {sp: n for sp, n in auto_comp.items() if n < 0}
        if negative:
            raise ValueError(
                f"Defect '{d['name']}': auto-completed composition has a "
                f"negative atom count {negative} -- 'removed_atoms' "
                f"({dict(removed)}) removes more atoms than "
                "'bulk.composition' contains.  Check for a typo, or give "
                "an explicit 'composition' per charge state.")
        for cs in d["charge_states"]:
            if "composition" not in cs or cs["composition"] is None:
                if not removed and not added:
                    warnings.warn(
                        f"[Input] {d['name']} q={cs['charge']:+d}: no "
                        "'composition' given and both 'removed_atoms'/"
                        "'added_atoms' are empty; the auto-completed "
                        "composition will equal the bulk composition "
                        "unchanged.  If this is not intentional, check "
                        "for a typo in 'removed_atoms'/'added_atoms'.",
                        UserWarning, stacklevel=2)
                cs["composition"] = auto_comp
                print(f"[INFO] Auto-completed composition for "
                      f"{d['name']} q={cs['charge']:+d}: {auto_comp}")
            _check_num_atoms(f"{d['name']} q={cs['charge']:+d}", cs)

    _check_num_atoms("bulk", raw["bulk"])

    return raw


def _is_number(x):
    """True for int/float, explicitly excluding bool.

    YAML 1.1 parses unquoted yes/no/on/off as booleans, which would
    otherwise silently pass through as 0/1 in a numeric field.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_3x3_numeric(tensor):
    return (isinstance(tensor, (list, tuple)) and len(tensor) == 3
            and all(isinstance(row, (list, tuple)) and len(row) == 3
                    and all(_is_number(v) for v in row)
                    for row in tensor))


def _validate_and_normalize_charge(cs, context):
    """Coerce cs['charge'] to a plain int in place, or raise with context.

    Accepts a real int, an integer-valued float (e.g. -1.0), or a numeric
    string (e.g. "-1"); rejects fractional and non-numeric values -- this
    is also what makes the later f"{cs['charge']:+d}" formatting (used in
    auto-completion messages and duplicate-charge detection) safe.
    """
    if not isinstance(cs, dict):
        raise ValueError(
            f"{context}: charge_state entries must be mappings "
            f"(got {type(cs).__name__}).")
    if "charge" not in cs:
        raise KeyError(
            f"{context}: charge_state is missing required key 'charge'.")
    raw_charge = cs["charge"]
    q = None
    if _is_number(raw_charge):
        if not (isinstance(raw_charge, float) and not raw_charge.is_integer()):
            q = int(raw_charge)
    elif isinstance(raw_charge, str):
        try:
            as_float = float(raw_charge)
        except ValueError:
            as_float = None
        if as_float is not None and as_float.is_integer():
            q = int(as_float)
    if q is None:
        raise ValueError(
            f"{context}: 'charge' must be an integer (got {raw_charge!r}); "
            "fractional or non-numeric charge values are not supported.")
    cs["charge"] = q


def _validate_potential_alignment(cs, context):
    """Raise with context on a malformed 'potential_alignment' block."""
    if "potential_alignment" not in cs:
        return
    pa = cs["potential_alignment"]
    if not isinstance(pa, dict):
        raise ValueError(
            f"{context}: 'potential_alignment' must be a mapping with a "
            f"'delta_v' key (got {type(pa).__name__}).")
    if "delta_v" in pa:
        if pa["delta_v"] is None:
            raise ValueError(
                f"{context}: 'potential_alignment.delta_v' is null; use "
                "an explicit numeric value (0.0 for the neutral charge "
                "state).")
        if not _is_number(pa["delta_v"]):
            raise ValueError(
                f"{context}: 'potential_alignment.delta_v' must be "
                f"numeric (got {pa['delta_v']!r}).")


def _validate_optional_positive_number(container, key, context):
    """If `key` is present (and not null) in `container`, it must be a
    positive number. Used for the optional Feat-1/Feat-2 YAML fields
    (effective_mass_e/h, degeneracy, site_density), which are all
    ignored on the default code path but should still fail fast, with
    context, if malformed."""
    if key not in container or container[key] is None:
        return
    value = container[key]
    if not _is_number(value) or value <= 0:
        raise ValueError(
            f"{context}: '{key}' must be a positive number "
            f"(got {value!r}).")


def _validate_composition_values(comp, context):
    """Every atom count in a composition mapping must be a
    non-negative number.

    Without this check, a sign typo in 'bulk.composition' (e.g. a
    stray '-') propagates silently: it is trusted as-is by the
    auto-completion loop below, producing a negative atom count in
    every auto-completed defect composition, which then feeds
    chem_pot_correction() and (with --solve-fermi) site-density
    calculations -- all without ever raising or warning.
    """
    for sp, n in comp.items():
        if not _is_number(n) or n < 0:
            raise ValueError(
                f"{context}['{sp}'] must be a non-negative number "
                f"(got {n!r}).")


def _check_duplicate_charges(defect_block):
    """Raise if the same (already-normalized, integer) charge appears
    more than once within one defect -- silently overwritten downstream
    otherwise, since charge states are assembled into a {charge: ...}
    dict keyed by this value."""
    seen = set()
    for cs in defect_block["charge_states"]:
        q = cs["charge"]
        if q in seen:
            raise ValueError(
                f"Defect '{defect_block['name']}' has duplicate charge "
                f"state q={q:+d}; each charge value must appear at most "
                "once per defect.")
        seen.add(q)


def _validate_nested_required_keys(raw):
    """Fail fast, with context, on malformed nested sections that would
    otherwise surface as an uncontextualized KeyError/TypeError deep
    inside geometry/Ewald computation (or, under MPI, only on rank 0)."""
    comp = raw["bulk"].get("composition")
    if not isinstance(comp, dict) or not comp:
        raise ValueError(
            "'bulk.composition' must be a non-empty mapping of "
            "{species: count, ...}.")
    _validate_composition_values(comp, "'bulk.composition'")

    if "energy" not in raw["bulk"]:
        raise KeyError("'bulk' is missing required key 'energy'.")
    if not _is_number(raw["bulk"]["energy"]):
        raise ValueError(
            f"'bulk.energy' must be a numeric value "
            f"(got {raw['bulk']['energy']!r}).")
    _validate_formation_reference(raw["bulk"])

    if "formula" not in raw["bulk"]:
        raise KeyError("'bulk' is missing required key 'formula'.")
    formula = raw["bulk"]["formula"]
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(
            f"'bulk.formula' must be a non-empty string "
            f"(got {formula!r}).")

    bs = raw["band_structure"]
    for key in ("vbm", "cbm", "bandgap", "fermi_level"):
        if key not in bs or not _is_number(bs[key]):
            raise ValueError(
                f"'band_structure.{key}' must be a numeric value "
                f"(got {bs.get(key)!r}).")
    if bs["cbm"] <= bs["vbm"]:
        raise ValueError(
            f"'band_structure.cbm' ({bs['cbm']}) must be greater than "
            f"'band_structure.vbm' ({bs['vbm']}).")
    _validate_optional_positive_number(bs, "effective_mass_e", "band_structure")
    _validate_optional_positive_number(bs, "effective_mass_h", "band_structure")

    tensor = raw["dielectric"].get("tensor")
    if not _is_3x3_numeric(tensor):
        raise ValueError(
            "'dielectric.tensor' must be a 3x3 matrix of numeric values.")

    top_supercell = raw.get("supercell")
    for d in raw["defects"]:
        sc = d.get("supercell", top_supercell)
        context = f"Defect '{d['name']}'"
        if sc is None:
            raise KeyError(
                f"{context} has no 'supercell' block and no top-level "
                "'supercell' is defined.")
        _validate_supercell_block(sc, context)
        _validate_optional_positive_number(d, "site_density", context)
        _validate_alignment_sampling_sites(d, context)

    competing_phases = raw.get("competing_phases") or []
    if not isinstance(competing_phases, list):
        raise ValueError("'competing_phases' must be a list.")
    for i, phase in enumerate(competing_phases):
        context = f"'competing_phases[{i}]'"
        if not isinstance(phase, dict):
            raise ValueError(f"{context} must be a mapping.")
        for key in ("formula", "energy", "composition"):
            if key not in phase:
                raise KeyError(f"{context} is missing required key '{key}'.")
        if not _is_number(phase["energy"]):
            raise ValueError(
                f"{context}: 'energy' must be numeric "
                f"(got {phase['energy']!r}).")
        if not isinstance(phase["composition"], dict) or not phase["composition"]:
            raise ValueError(
                f"{context}: 'composition' must be a non-empty mapping "
                "of {species: count, ...}.")


def _validate_alignment_sampling_sites(defect_block, context):
    """Optional per-defect 'alignment_sampling_sites' (docs/adr/0009): a
    non-empty list of fractional [x, y, z] coordinates."""
    if "alignment_sampling_sites" not in defect_block:
        return
    sites = defect_block["alignment_sampling_sites"]
    ok = (isinstance(sites, list) and len(sites) > 0 and all(
        isinstance(s, (list, tuple)) and len(s) == 3 and all(_is_number(x) for x in s)
        for s in sites))
    if not ok:
        raise ValueError(
            f"{context}: 'alignment_sampling_sites' must be a non-empty list "
            "of fractional [x, y, z] coordinates.")


def _validate_formation_reference(bulk):
    """Optional 'bulk.formation_reference' {energy, composition}
    (docs/adr/0007): must describe the same species in the same
    stoichiometric ratios as 'bulk.composition' (1e-6 relative), since a
    different stoichiometry would silently produce a wrong dH_f."""
    ref = bulk.get("formation_reference")
    if ref is None:
        return
    context = "'bulk.formation_reference'"
    if not isinstance(ref, dict):
        raise ValueError(f"{context} must be a mapping with 'energy' and 'composition'.")
    for key in ("energy", "composition"):
        if key not in ref:
            raise KeyError(f"{context} is missing required key '{key}'.")
    if not _is_number(ref["energy"]):
        raise ValueError(f"{context}: 'energy' must be numeric (got {ref['energy']!r}).")
    comp = ref["composition"]
    if not isinstance(comp, dict) or not comp:
        raise ValueError(f"{context}: 'composition' must be a non-empty mapping.")
    _validate_composition_values(comp, f"{context} composition")
    host = {sp: n for sp, n in bulk["composition"].items() if n}
    ref_nz = {sp: n for sp, n in comp.items() if n}
    if set(host) != set(ref_nz):
        raise ValueError(
            f"{context}: species {sorted(ref_nz)} differ from "
            f"'bulk.composition' species {sorted(host)}.")
    scale = sum(ref_nz.values()) / sum(host.values())
    for sp in sorted(host):
        if abs(ref_nz[sp] - scale * host[sp]) > 1e-6 * max(1.0, ref_nz[sp]):
            raise ValueError(
                f"{context}: composition {ref_nz} does not have the same "
                f"stoichiometry as 'bulk.composition' {host}.")


def _validate_supercell_block(sc, context):
    if not isinstance(sc, dict):
        raise ValueError(f"{context}: 'supercell' must be a mapping.")
    lv = sc.get("lattice_vectors")
    if not isinstance(lv, dict) or not {"a", "b", "c"} <= set(lv):
        raise ValueError(
            f"{context}: 'supercell.lattice_vectors' must be a mapping "
            "with keys 'a', 'b', 'c'.")
    for label in ("a", "b", "c"):
        vec = lv[label]
        if (not isinstance(vec, (list, tuple)) or len(vec) != 3
                or not all(_is_number(v) for v in vec)):
            raise ValueError(
                f"{context}: 'supercell.lattice_vectors.{label}' must be "
                "a 3-element list of numeric values.")
    pos = sc.get("defect_position")
    if (not isinstance(pos, (list, tuple)) or len(pos) != 3
            or not all(_is_number(v) for v in pos)):
        raise ValueError(
            f"{context}: 'supercell.defect_position' must be a 3-element "
            "list of numeric values.")


def _check_num_atoms(label, block):
    """Warn if a declared 'num_atoms' disagrees with sum(composition)."""
    comp = block.get("composition")
    if "num_atoms" not in block or not comp:
        return
    n_declared  = block["num_atoms"]
    n_from_comp = sum(comp.values())
    try:
        mismatch = float(n_declared) != float(n_from_comp)
    except (TypeError, ValueError):
        warnings.warn(
            f"[Input] {label}: num_atoms ({n_declared!r}) is not "
            "numeric; cannot cross-check against "
            f"sum(composition.values()) ({n_from_comp}).",
            UserWarning, stacklevel=3)
        return
    if mismatch:
        warnings.warn(
            f"[Input] {label}: num_atoms ({n_declared}) does not match "
            f"sum(composition.values()) ({n_from_comp}).  "
            "Verify your input data.",
            UserWarning, stacklevel=3)


def _safe_mkdir(path):
    """Create directory if it does not already exist."""
    if path and path != ".":
        os.makedirs(path, exist_ok=True)


def _out(base_dir, *parts):
    """Join base_dir with path parts, creating dirs as needed."""
    p = os.path.join(base_dir, *parts)
    _safe_mkdir(os.path.dirname(p))
    return p


def _atomic_write_text(path, content):
    """Write `content` to `path` atomically via write-to-temp + os.replace().

    Eliminates the "partially-written / corrupted file" failure mode --
    e.g. two MPI ranks that both believe themselves to be rank 0 (an
    undetected launcher; see mpi_utils._detected_external_launcher)
    writing the same path concurrently. This is not mutual exclusion:
    every rank writes identical bytes from identical input, so whichever
    replace() lands last simply wins -- but no reader can ever observe a
    half-written file.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_text(path, lines):
    _atomic_write_text(path, "\n".join(lines) + "\n")
    print(f"[INFO] Report written to: {path}")


def write_ef_table(path, defect_name, fermi_grid, ef_by_charge, envelope):
    """Write E_f(q, E_F) + envelope as a fixed-width plain-text table."""
    charges    = sorted(ef_by_charge.keys())
    col_labels = (["EF_rel_VBM[eV]"]
                  + [f"Ef(q={q:+d})[eV]" for q in charges]
                  + ["Ef_envelope[eV]"])
    COL_W = 18
    SEP   = "  "

    def row_str(vals):
        return SEP.join(f"{str(v):>{COL_W}}" for v in vals)

    header_str = row_str(col_labels)
    ruler      = "-" * len(header_str)
    lines = [
        f"# Defect formation energy vs. Fermi level  [{defect_name}]",
        "# Method: Kumagai-Oba correction (Phys. Rev. B 89, 195205, 2014)",
        "#",
        f"# Charge states included : {charges}",
        f"# Fermi sweep range      : [{fermi_grid[0]:.6f}, "
                                    f"{fermi_grid[-1]:.6f}] eV (rel. VBM)",
        f"# Number of E_F points   : {len(fermi_grid)}",
        "#",
        "# EF_rel_VBM : Fermi level measured from the VBM (eV)",
        "# Ef(q=...)  : Formation energy for charge state q (eV)",
        "# Ef_envelope: Min E_f over all q at each E_F (stable envelope) (eV)",
        "#",
        ruler, header_str, ruler,
    ]
    for i, ef_val in enumerate(fermi_grid):
        vals = ([f"{ef_val:+.6f}"]
                + [f"{ef_by_charge[q][i]:+.6f}" for q in charges]
                + [f"{envelope[i]:+.6f}"])
        lines.append(row_str(vals))
    lines.append(ruler)
    _atomic_write_text(path, "\n".join(lines) + "\n")
    print(f"[INFO] E_f table written to: {path}")


def write_summary_txt(path, defect_name, summary_rows, fermi_ref):
    """Write per-charge KO breakdown + E_f as a plain-text table."""
    col_labels = [
        "q", "E_KO[eV]", "E_real[eV]", "E_recip[eV]",
        "E_self[eV]", "E_neut[eV]", "E_PC[eV]", "E_lat[eV]",
        "E_align[eV]", f"Ef(EF={fermi_ref:.3f}eV)[eV]",
    ]
    COL_W_Q = 5
    COL_W   = 18
    SEP     = "  "

    def row_str(vals):
        parts = [f"{str(vals[0]):>{COL_W_Q}}"]
        parts += [f"{str(v):>{COL_W}}" for v in vals[1:]]
        return SEP.join(parts)

    header_str = row_str(col_labels)
    ruler      = "-" * len(header_str)
    lines = [
        f"# Defect formation energy summary  [{defect_name}]",
        "# Method: Kumagai-Oba corrected",
        f"# Fermi level reference : {fermi_ref:.4f} eV "
        "(relative to VBM; negative = below VBM)",
        "#",
        "# q       : Charge state (electrons; negative = acceptor-like)",
        "# E_KO    : Total Kumagai-Oba finite-size correction = E_lat + E_align (eV)",
        "# E_real  : Real-space Ewald sum (eV)",
        "# E_recip : Reciprocal-space Ewald sum (eV)",
        "# E_self  : Ewald Gaussian self-energy (eV)",
        "# E_neut  : Jellium background neutralization (eV)",
        "# E_PC    : Point-charge lattice energy = E_real+E_recip+E_self+E_neut (eV)",
        "# E_lat   : Lattice correction = -E_PC (eV)",
        "# E_align : Potential alignment = q * delta_V (eV)",
        "# Ef      : Formation energy at the reference Fermi level (eV)",
        "#",
        ruler, header_str, ruler,
    ]
    for row in summary_rows:
        vals = [
            f"{row['q']:+d}",
            f"{row['E_KO']:+.6f}", f"{row['E_real']:+.6f}",
            f"{row['E_recip']:+.6f}", f"{row['E_self_ewald']:+.6f}",
            f"{row['E_neutralize']:+.6f}", f"{row['E_PC']:+.6f}",
            f"{row['E_lat']:+.6f}",
            f"{row['E_alignment']:+.6f}", f"{row['Ef_at_fermi']:+.6f}",
        ]
        lines.append(row_str(vals))
    lines.append(ruler)
    _atomic_write_text(path, "\n".join(lines) + "\n")
    print(f"[INFO] Summary written to: {path}")


def write_ctls_summary(path, all_results, gap):
    """
    Write a unified CTL summary table for all defects.

    Columns: Defect, q1, q2, E_F_CTL[eV], Ef_CTL[eV], on_envelope
    'on_envelope' = yes  when the CTL is within [0, gap] AND both charge
                         states are adjacent on the stable envelope.
    """
    col_labels = ["Defect", "q1", "q2",
                  "E_F_CTL[eV]", "Ef_CTL[eV]", "on_envelope"]
    COL_W  = [12, 5, 5, 14, 14, 12]
    SEP    = "  "

    def row_str(vals):
        return SEP.join(f"{str(v):>{w}}" for v, w in zip(vals, COL_W))

    header_str = row_str(col_labels)
    ruler      = "-" * len(header_str)
    lines = [
        "# Charge-State Transition Level (CTL) summary -- all defects",
        "# Method: Kumagai-Oba corrected",
        "#",
        "# on_envelope : 'yes' = CTL lies within the band gap [0, gap] AND",
        "#               represents an adjacency on the stable envelope.",
        "#               'no'  = intersection outside the gap or between",
        "#               charge states that are not envelope-adjacent.",
        "#",
        ruler, header_str, ruler,
    ]
    for res in all_results:
        stable_ctls, meta_ctls = res["stable_ctls"], res["meta_ctls"]
        for ctl in sorted(stable_ctls, key=lambda x: x["E_F_CTL"]):
            lines.append(row_str([
                res["name"],
                f"{ctl['q1']:+d}", f"{ctl['q2']:+d}",
                f"{ctl['E_F_CTL']:+.4f}", f"{ctl['E_f_CTL']:+.4f}", "yes",
            ]))
        for ctl in sorted(meta_ctls, key=lambda x: x["E_F_CTL"]):
            lines.append(row_str([
                res["name"],
                f"{ctl['q1']:+d}", f"{ctl['q2']:+d}",
                f"{ctl['E_F_CTL']:+.4f}", f"{ctl['E_f_CTL']:+.4f}", "no",
            ]))
    lines.append(ruler)
    _atomic_write_text(path, "\n".join(lines) + "\n")
    print(f"[INFO] CTL summary written to: {path}")


def write_equilibrium_summary(path, data, equilibrium_result):
    """
    Write the --solve-fermi self-consistent equilibrium summary: solved
    Fermi level, carrier concentrations, and per-(defect, charge)
    equilibrium concentrations (see ikazuchix.equilibrium.solve_equilibrium).
    """
    r   = equilibrium_result
    bs  = data["band_structure"]
    gap = bs["cbm"] - bs["vbm"]

    header_lines = [
        "# Self-consistent equilibrium Fermi level and concentrations",
        "# Method: charge-neutrality bisection "
        "(dilute limit, non-degenerate/Boltzmann statistics)",
        "#",
        f"# Temperature              : {r['temperature_k']:.2f} K",
        f"# Carrier terms included   : {r['include_carriers']}",
        f"# Band gap                 : {gap:.4f} eV",
        "#",
        f"# Solved Fermi level (rel. VBM) : {r['fermi_level_rel_vbm']:+.6f} eV",
        f"# Solved Fermi level (absolute) : {r['fermi_level_abs']:+.6f} eV",
        f"# Electron concentration n      : {r['electron_conc_cm3']:.6e} cm^-3",
        f"# Hole concentration p          : {r['hole_conc_cm3']:.6e} cm^-3",
        "#",
    ]

    col_labels = ["Defect", "q", "Concentration[cm^-3]"]
    COL_W = [16, 5, 22]
    SEP   = "  "

    def row_str(vals):
        return SEP.join(f"{str(v):>{w}}" for v, w in zip(vals, COL_W))

    header_str = row_str(col_labels)
    ruler      = "-" * len(header_str)
    lines = header_lines + [ruler, header_str, ruler]
    for c in r["defect_concentrations"]:
        lines.append(row_str([
            c["name"], f"{c['q']:+d}", f"{c['concentration_cm3']:.6e}"]))
    lines.append(ruler)

    _atomic_write_text(path, "\n".join(lines) + "\n")
    print(f"[INFO] Equilibrium summary written to: {path}")


def write_stability_summary(path, data, stability_result):
    """
    Write the --stability-diagram summary: feasibility of the input
    file's 'chemical_potentials' against every constraint (host
    equality, elemental bounds, competing phases), and -- for a
    ternary host -- the 2D stability-region polygon vertices (see
    ikazuchix.stability_region.stability_summary).
    """
    r = stability_result
    lines = [
        "# Chemical potential stability region summary",
        "# Method: linear constraint check "
        "(host equality, elemental bounds, competing phases)",
        "#",
        f"# Host species                        : {', '.join(r['species'])}",
        f"# dH_f(host)                          : {r['dH_f_host']:+.4f} eV",
        f"# Competing phases checked            : {r['n_competing_phases']}",
        f"# Input chemical potentials feasible  : {r['feasible']}",
        "#",
    ]
    if r["violations"]:
        lines.append("# Violations:")
        lines += [f"#   - {v}" for v in r["violations"]]
    else:
        lines.append("# No violations.")
    lines.append("#")

    if "region_vertices" in r:
        lines.append(
            f"# 2D stability region  (axes: delta_mu_{r['region_species_x']}, "
            f"delta_mu_{r['region_species_y']}; "
            f"delta_mu_{r['region_species_z']} eliminated via the host "
            "equality constraint)")
        if r["region_vertices"]:
            lines.append("#   Vertices [eV]:")
            lines += [f"#     ({x:+.4f}, {y:+.4f})"
                      for x, y in r["region_vertices"]]
        else:
            lines.append("#   Region is empty: the host is unstable "
                          "against the given competing phases.")
    else:
        lines.append("# (2D stability region only computed for a ternary "
                      "(3-species) host.)")

    _atomic_write_text(path, "\n".join(lines) + "\n")
    print(f"[INFO] Stability summary written to: {path}")
