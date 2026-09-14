"""Chemical-potential stability region analysis (Feat-2) -- fully
opt-in, engaged only via --stability-diagram; nothing here is imported
or executed on the default code path.

Physical model
--------------
For a host compound with (raw, whatever atom count the input's
'bulk.composition' happens to describe) composition {species: n_i},
thermodynamic equilibrium constrains the chemical-potential deviations
Δμ_i (relative to each species' elemental reference) to:

    host equality   : sum_i n_i * Δμ_i  =  dH_f(host)
    elemental bound  : Δμ_i <= 0                          for every i
    competing phase  : sum_i m_i * Δμ_i <= dH_f(phase)     for every
                        explicit competing phase {formula, energy,
                        composition} in the optional 'competing_phases'
                        input list (m_i = that phase's own composition)

Any concrete numeric Δμ vector can be checked against all of the above
directly (is_feasible_point, works for any number of species). For a
ternary host (exactly 3 species) the *set* of all Δμ satisfying every
constraint simultaneously is a 2D polygon (the host equality removes
one degree of freedom from 3D Δμ-space) -- feasible_region_2d computes
it via Sutherland-Hodgman half-plane clipping. For 4+ species that
region lives in too many dimensions to usefully enumerate as a polygon
here, so only the direct point check is offered (task scope).
"""

from ikazuchix.chem_potentials import competing_phase_bound


def free_species_order(host_composition):
    """Deterministic species ordering used throughout this module (and
    as the default axis assignment for feasible_region_2d)."""
    return sorted(host_composition)


def _host_dH_f(bulk_energy, host_composition, elemental_references):
    return float(bulk_energy) - sum(
        n * elemental_references.get(sp, 0.0)
        for sp, n in host_composition.items())


def is_feasible_point(deltas, host_composition, elemental_references,
                       dH_f_host, competing_phases, tol=1e-6):
    """Direct feasibility check of a SPECIFIC chemical-potential point
    `deltas` ({species: Δμ}) against every constraint: the host
    equality, each species' elemental bound, and each competing
    phase's linear bound. Works for any number of species -- no
    polygon/vertex enumeration is needed for a single-point check.

    The host-equality check uses a looser tolerance than `tol`
    (max(tol, 0.05 eV/atom * n_atoms)), matching
    check_chemical_potentials' own 0.05 eV/atom host-stability slack --
    DFT-derived chemical potentials routinely miss the equality by
    that much without being considered "wrong"; the inequality
    constraints (which define the polygon's actual edges) are checked
    at the tighter `tol` instead.

    Returns (feasible, violations): violations is a list of
    human-readable strings, empty iff feasible is True.
    """
    violations = []
    n_atoms = sum(host_composition.values()) or 1
    host_tol = max(tol, 0.05 * n_atoms)

    lhs_host = sum(host_composition.get(sp, 0) * deltas.get(sp, 0.0)
                   for sp in host_composition)
    if abs(lhs_host - dH_f_host) > host_tol:
        violations.append(
            f"host stability: sum n_i*delta_mu_i = {lhs_host:+.4f} eV != "
            f"dH_f(host) = {dH_f_host:+.4f} eV (tol={host_tol:.4f} eV)")

    for sp in host_composition:
        dmu = deltas.get(sp, 0.0)
        if dmu > tol:
            violations.append(
                f"elemental bound: delta_mu_{sp} = {dmu:+.4f} eV > 0")

    for phase in competing_phases:
        phase_comp, dH_f_phase = competing_phase_bound(phase, elemental_references)
        lhs = sum(n * deltas.get(sp, 0.0) for sp, n in phase_comp.items())
        if lhs > dH_f_phase + tol:
            violations.append(
                f"competing phase '{phase.get('formula', '?')}': "
                f"sum n_i*delta_mu_i = {lhs:+.4f} eV > "
                f"dH_f = {dH_f_phase:+.4f} eV")

    return not violations, violations


def _clip_polygon(polygon, coeff_x, coeff_y, rhs, tol=1e-9):
    """Sutherland-Hodgman: clip `polygon` (list of (x, y) vertices, in
    order) to the half-plane coeff_x*X + coeff_y*Y <= rhs."""
    if not polygon:
        return polygon

    def value(pt):
        return coeff_x * pt[0] + coeff_y * pt[1]

    def intersect(p1, p2):
        v1, v2 = value(p1), value(p2)
        t = (v1 - rhs) / (v1 - v2)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    result = []
    n = len(polygon)
    for i in range(n):
        cur, prev = polygon[i], polygon[i - 1]
        cur_in  = value(cur)  <= rhs + tol
        prev_in = value(prev) <= rhs + tol
        if cur_in:
            if not prev_in:
                result.append(intersect(prev, cur))
            result.append(cur)
        elif prev_in:
            result.append(intersect(prev, cur))
    return result


def feasible_region_2d(host_composition, elemental_references, dH_f_host,
                        competing_phases, box=None):
    """Sutherland-Hodgman half-plane-clipped stability-region polygon
    for a TERNARY (exactly 3 species) host, parameterized by the first
    two species in free_species_order() as the 2D plot axes; the third
    is eliminated via the host equality constraint
    sum_i n_i*Δμ_i = dH_f_host.

    box: optional explicit (lo, hi) starting bounding square [eV] for
    both axes. If None, sized from dH_f_host and every competing
    phase's own dH_f (so the box is always large enough to contain the
    true region regardless of how negative those enthalpies are --
    deliberately not a fixed numeric cutoff, which could be too small
    for an unusual input and silently truncate the true region).

    Returns (species_x, species_y, species_z, vertices): vertices is a
    list of (Δμ_x, Δμ_y) polygon corners in order (empty if infeasible).
    """
    species = free_species_order(host_composition)
    if len(species) != 3:
        raise ValueError(
            "feasible_region_2d requires exactly 3 species in the host "
            f"composition (got {len(species)}: {species}).")
    sx, sy, sz = species
    a, b, c = (host_composition[sx], host_composition[sy], host_composition[sz])
    if c == 0:
        raise ValueError(
            f"Host composition has zero atoms of '{sz}'; cannot "
            "eliminate it via the host equality constraint.")

    if box is None:
        scales = [abs(dH_f_host)] + [
            abs(competing_phase_bound(p, elemental_references)[1])
            for p in competing_phases]
        span = max(scales + [1.0]) * 2.0 + 1.0
        box = (-span, span)
    lo, hi = box
    polygon = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]

    # Elemental bounds: Δμ_x <= 0, Δμ_y <= 0 directly; Δμ_z <= 0 after
    # substituting Δμ_z = (dH_f_host - a*Δμ_x - b*Δμ_y) / c (flip the
    # inequality if c < 0, since dividing by it would).
    polygon = _clip_polygon(polygon, 1.0, 0.0, 0.0)
    polygon = _clip_polygon(polygon, 0.0, 1.0, 0.0)
    if c > 0:
        polygon = _clip_polygon(polygon, -a, -b, -dH_f_host)
    else:
        polygon = _clip_polygon(polygon, a, b, dH_f_host)

    for phase in competing_phases:
        phase_comp, dH_f_phase = competing_phase_bound(phase, elemental_references)
        nx, ny, nz = (phase_comp.get(sx, 0), phase_comp.get(sy, 0),
                      phase_comp.get(sz, 0))
        # n_x*Δμ_x + n_y*Δμ_y + n_z*Δμ_z <= dH_f_phase; substitute Δμ_z:
        coeff_x = nx - nz * a / c
        coeff_y = ny - nz * b / c
        rhs     = dH_f_phase - nz * dH_f_host / c
        polygon = _clip_polygon(polygon, coeff_x, coeff_y, rhs)

    return sx, sy, sz, polygon


def stability_summary(data, tol=1e-6):
    """Top-level --stability-diagram orchestrator: validate the input
    file's 'chemical_potentials' against every competing-phase and
    elemental-bound constraint, and (for a ternary host only) compute
    the 2D stability-region polygon for plotting.

    Returns a dict: species, point_deltas, feasible, violations,
    dH_f_host, n_competing_phases, and -- only when the host has
    exactly 3 species -- region_species_x/y/z and region_vertices.
    """
    comp   = data["bulk"]["composition"]
    refs   = data["elemental_references"]
    deltas = data["chemical_potentials"]
    competing_phases = data.get("competing_phases") or []

    dH_f_host = _host_dH_f(data["bulk"]["energy"], comp, refs)
    feasible, violations = is_feasible_point(
        deltas, comp, refs, dH_f_host, competing_phases, tol=tol)

    result = dict(
        species=free_species_order(comp), point_deltas=dict(deltas),
        feasible=feasible, violations=violations, dH_f_host=dH_f_host,
        n_competing_phases=len(competing_phases))

    if len(comp) == 3:
        sx, sy, sz, vertices = feasible_region_2d(
            comp, refs, dH_f_host, competing_phases)
        result.update(
            region_species_x=sx, region_species_y=sy, region_species_z=sz,
            region_vertices=vertices)

    return result
