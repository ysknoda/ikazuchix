"""Unit tests for ikazuchix.runner: decomposition consistency and rank/size
invariance of the MPI work-distribution algorithm (simulated in-process;
a real multi-process mpirun check lives in test_mpi_consistency.py)."""

import copy
import warnings

import numpy as np
import pytest
import yaml

import ikazuchix.ewald as ewald_module
import ikazuchix.runner as runner_module
from ikazuchix.io_utils import load_input
from ikazuchix.mpi_utils import NullComm
from ikazuchix.runner import (
    DefectComputationError,
    _resolve_defect_geometry,
    assemble_defect_result,
    compute_charge_state_result,
    run_all_defects,
    run_one_defect,
)

_DATA = dict(
    bulk=dict(formula="GaN", energy=-462.924, num_atoms=72,
               composition={"Ga": 36, "N": 36}),
    elemental_references={"Ga": -3.031, "N": -8.328},
    chemical_potentials={"Ga": 0.0, "N": -1.5},
    band_structure=dict(vbm=5.0, cbm=6.7, bandgap=1.7, fermi_level=5.85),
    dielectric=dict(tensor=[[9.5, 0, 0], [0, 9.5, 0], [0, 0, 10.4]]),
    supercell=dict(
        lattice_vectors={"a": [9.57, 0, 0], "b": [-4.785, 8.2877, 0], "c": [0, 0, 10.38]},
        defect_position=[0.16667, 0.33333, 0.0],
    ),
    defects=[
        dict(name="V_Ga", defect_type="vacancy", removed_atoms={"Ga": 1}, charge_states=[
            dict(charge=0, energy=-454.393, num_atoms=71, potential_alignment={"delta_v": 0.0}),
            dict(charge=-1, energy=-451.0689, num_atoms=71, potential_alignment={"delta_v": 0.02}),
        ]),
        dict(name="V_N", defect_type="vacancy", removed_atoms={"N": 1}, charge_states=[
            dict(charge=0, energy=-449.096, num_atoms=71, potential_alignment={"delta_v": 0.0}),
            dict(charge=1, energy=-457.2919, num_atoms=71, potential_alignment={"delta_v": 0.0}),
        ]),
    ],
)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(_DATA), encoding="utf-8")
    return load_input(path)


@pytest.fixture
def grid():
    return np.linspace(-0.17, 1.87, 400)


def _run_serial_all_defects(data, fermi_ref, grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [run_one_defect(d, data, fermi_ref, grid) for d in data["defects"]]


def _simulate_n_rank_run(data, fermi_ref, grid, size):
    """Manually replicate run_all_defects' distribute -> gather ->
    assemble algorithm for `size` simulated ranks within this one
    process, using the same primitives run_all_defects itself calls."""
    defect_geoms = [_resolve_defect_geometry(d, data) for d in data["defects"]]
    work_items = [
        (d_idx, cs_idx)
        for d_idx, d in enumerate(data["defects"])
        for cs_idx in range(len(d["charge_states"]))
    ]

    merged = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for rank in range(size):
            for d_idx, cs_idx in work_items[rank::size]:
                defect_block = data["defects"][d_idx]
                cs = defect_block["charge_states"][cs_idx]
                lattice, epsilon, def_pos = defect_geoms[d_idx]
                merged[(d_idx, cs_idx)] = compute_charge_state_result(
                    defect_block, cs, data, fermi_ref, grid,
                    lattice, epsilon, def_pos)

    bs = data["band_structure"]
    gap = bs["cbm"] - bs["vbm"]
    results = []
    for d_idx, defect_block in enumerate(data["defects"]):
        charge_results = [merged[(d_idx, cs_idx)]
                           for cs_idx in range(len(defect_block["charge_states"]))]
        results.append(assemble_defect_result(defect_block, charge_results, grid, gap))
    return results


def _strip_console_only_fields(results):
    """Numeric-relevant view of a results list, for cross-rank-count
    comparison (console_lines/captured_warnings are not part of the
    per-defect result dict, so nothing to strip there -- kept as a
    single comparison point in case that ever changes)."""
    return results


def test_compute_and_assemble_compose_to_run_one_defect(data, grid):
    fermi_ref = 0.85
    defect_block = data["defects"][0]
    lattice, epsilon, def_pos = _resolve_defect_geometry(defect_block, data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        charge_results = [
            compute_charge_state_result(
                defect_block, cs, data, fermi_ref, grid, lattice, epsilon, def_pos)
            for cs in defect_block["charge_states"]
        ]
        manual = assemble_defect_result(
            defect_block, charge_results, grid,
            data["band_structure"]["cbm"] - data["band_structure"]["vbm"])
        via_run_one_defect = run_one_defect(defect_block, data, fermi_ref, grid)

    assert manual["charges"] == via_run_one_defect["charges"]
    assert manual["stable_ctls"] == via_run_one_defect["stable_ctls"]
    for q in manual["charges"]:
        assert manual["ko"][q] == via_run_one_defect["ko"][q]
        np.testing.assert_array_equal(
            manual["ef_by_charge"][q], via_run_one_defect["ef_by_charge"][q])


def test_run_all_defects_with_nullcomm_matches_serial(data, grid):
    fermi_ref = 0.85
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        via_pipeline = run_all_defects(data, fermi_ref, grid, comm=NullComm())
    serial = _run_serial_all_defects(data, fermi_ref, grid)

    assert [r["name"] for r in via_pipeline] == [r["name"] for r in serial]
    for a, b in zip(via_pipeline, serial):
        assert a["charges"] == b["charges"]
        assert a["stable_ctls"] == b["stable_ctls"]
        assert a["meta_ctls"] == b["meta_ctls"]
        for q in a["charges"]:
            assert a["ko"][q] == b["ko"][q]
            np.testing.assert_array_equal(a["ef_by_charge"][q], b["ef_by_charge"][q])


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_work_distribution_is_rank_count_invariant(data, grid, size):
    """The physics must not depend on how many (simulated) ranks the
    (defect, charge_state) work list was split across."""
    fermi_ref = 0.85
    reference = _simulate_n_rank_run(data, fermi_ref, grid, size=1)
    candidate = _simulate_n_rank_run(data, fermi_ref, grid, size=size)

    for a, b in zip(_strip_console_only_fields(reference),
                     _strip_console_only_fields(candidate)):
        assert a["charges"] == b["charges"]
        assert a["stable_ctls"] == b["stable_ctls"]
        assert a["meta_ctls"] == b["meta_ctls"]
        for q in a["charges"]:
            assert a["ko"][q] == b["ko"][q]
            np.testing.assert_array_equal(a["ef_by_charge"][q], b["ef_by_charge"][q])


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_round_robin_work_distribution_covers_every_item_exactly_once(size):
    work_items = [(d, c) for d in range(3) for c in range(4)]  # 12 items
    assigned = []
    for rank in range(size):
        assigned.extend(work_items[rank::size])
    assert sorted(assigned) == sorted(work_items)
    assert len(assigned) == len(work_items)


# ===========================================================================
# Fix-1: a failing work item must not block comm.gather() for other ranks
# ===========================================================================

def _make_flaky_compute(defect_name, charge):
    """A drop-in replacement for compute_charge_state_result that raises
    for exactly one (defect, charge) pair and delegates to the real
    implementation for every other work item."""
    original = runner_module.compute_charge_state_result

    def flaky(defect_block, cs, *args, **kwargs):
        if defect_block["name"] == defect_name and cs["charge"] == charge:
            raise ValueError("synthetic failure for testing")
        return original(defect_block, cs, *args, **kwargs)

    return flaky


def test_run_all_defects_aggregates_errors_and_still_gathers(data, grid, monkeypatch):
    """Single (NullComm) rank: one work item fails, comm.gather() still
    completes (there is nothing else to hang), and the failure is
    reported via DefectComputationError rather than the raw ValueError
    propagating uncontextualized."""
    monkeypatch.setattr(runner_module, "compute_charge_state_result",
                         _make_flaky_compute("V_N", 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(DefectComputationError, match="V_N"):
            runner_module.run_all_defects(data, 0.85, grid, comm=NullComm())


def test_run_all_defects_error_on_worker_rank_does_not_crash_worker(
        data, grid, monkeypatch):
    """The real risk Fix-1 addresses: a work item that fails on a
    *non-root* rank must be caught locally (recorded into that rank's
    local_results) so every rank still reaches comm.gather() -- only
    rank 0, after gathering, raises the aggregated error. Simulated with
    a 2-rank fake comm sharing a mailbox dict, calling run_all_defects
    once per simulated rank (worker before root, so the root's gather
    sees the worker's already-deposited results)."""
    # V_Ga has charge_states [0, -1]; V_N has [0, 1]. work_items are built
    # as [(V_Ga,0), (V_Ga,-1), (V_N,0), (V_N,1)]; with round-robin size=2,
    # rank 1 owns indices 1 and 3 -- i.e. (V_Ga, q=-1) and (V_N, q=+1).
    monkeypatch.setattr(runner_module, "compute_charge_state_result",
                         _make_flaky_compute("V_N", 1))

    mailbox = {}

    class _FakeComm:
        def __init__(self, rank):
            self._rank = rank

        def Get_rank(self):
            return self._rank

        def Get_size(self):
            return 2

        def bcast(self, obj, root=0):
            return obj

        def gather(self, obj, root=0):
            mailbox[self._rank] = obj
            if self._rank != root:
                return None
            return [mailbox.get(r) for r in range(2)]

        def Barrier(self):
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Worker rank: must return None quietly, NOT raise, even though
        # one of its two work items (V_N, q=+1) is the flaky one.
        worker_result = runner_module.run_all_defects(
            data, 0.85, grid, comm=_FakeComm(1))
        assert worker_result is None

        # Root rank: gathers both ranks' results (including the
        # worker's error record) and raises exactly once, aggregated.
        with pytest.raises(DefectComputationError, match="V_N"):
            runner_module.run_all_defects(data, 0.85, grid, comm=_FakeComm(0))


# ===========================================================================
# Perf-1: EwaldShellCache reuse across a defect's charge_states loop
# ===========================================================================

def test_run_one_defect_reuses_kernel_calls_across_charge_states(tmp_path, grid, monkeypatch):
    """Two non-zero charge states of the SAME defect, at the same
    (default) cutoff, must only trigger the underlying Ewald kernel
    once per space -- confirming run_one_defect's per-defect
    EwaldShellCache is actually shared across compute_charge_state_result
    calls, not just constructed and left unused."""
    data_dict = copy.deepcopy(_DATA)
    data_dict["defects"] = [dict(
        name="V_Ga", defect_type="vacancy", removed_atoms={"Ga": 1},
        charge_states=[
            dict(charge=-1, energy=-451.0689, num_atoms=71,
                 potential_alignment={"delta_v": 0.02}),
            dict(charge=-2, energy=-451.3766, num_atoms=71,
                 potential_alignment={"delta_v": 0.05}),
        ],
    )]
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data_dict), encoding="utf-8")
    two_charge_data = load_input(path)

    call_count = {"real": 0, "recip": 0}
    original_real  = ewald_module.ewald_real_shell_sum
    original_recip = ewald_module.ewald_recip_shell_sum

    def counting_real(*args, **kwargs):
        call_count["real"] += 1
        return original_real(*args, **kwargs)
    counting_real.py_func = counting_real  # keep use_kernel() working with --no-numba

    def counting_recip(*args, **kwargs):
        call_count["recip"] += 1
        return original_recip(*args, **kwargs)
    counting_recip.py_func = counting_recip

    monkeypatch.setattr(ewald_module, "ewald_real_shell_sum", counting_real)
    monkeypatch.setattr(ewald_module, "ewald_recip_shell_sum", counting_recip)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_one_defect(two_charge_data["defects"][0], two_charge_data, 0.85, grid)

    assert call_count["real"] == 1
    assert call_count["recip"] == 1
