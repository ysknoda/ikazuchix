"""Unit tests for ikazuchix.mpi_utils (MPI/NullComm fallback + warning replay)."""

import warnings

import pytest

import ikazuchix.mpi_utils as mpi_utils
from ikazuchix.mpi_utils import (
    NullComm,
    capture_warnings_as_picklable,
    get_comm,
    raise_consistently,
    replay_warning,
)


def test_nullcomm_rank_and_size():
    comm = NullComm()
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1


def test_nullcomm_bcast_is_identity():
    comm = NullComm()
    obj = {"a": 1, "b": [1, 2, 3]}
    assert comm.bcast(obj, root=0) is obj


def test_nullcomm_gather_wraps_in_singleton_list():
    comm = NullComm()
    assert comm.gather("payload", root=0) == ["payload"]


def test_nullcomm_barrier_is_noop():
    NullComm().Barrier()  # must not raise


def test_get_comm_default_matches_has_mpi():
    comm = get_comm()
    if mpi_utils.HAS_MPI:
        assert comm is mpi_utils.MPI.COMM_WORLD
    else:
        assert isinstance(comm, NullComm)


def test_get_comm_force_disable_always_returns_nullcomm(monkeypatch):
    # Simulate a real MPI being available, and confirm force_disable still
    # yields a single-process NullComm (this is the --no-mpi mechanism).
    monkeypatch.setattr(mpi_utils, "HAS_MPI", True)
    monkeypatch.setattr(mpi_utils, "MPI", object())  # any non-None sentinel
    assert isinstance(get_comm(force_disable=True), NullComm)


def test_get_comm_uses_real_mpi_when_available_and_not_disabled(monkeypatch):
    sentinel_comm = object()

    class FakeMPI:
        COMM_WORLD = sentinel_comm

    monkeypatch.setattr(mpi_utils, "HAS_MPI", True)
    monkeypatch.setattr(mpi_utils, "MPI", FakeMPI)
    assert get_comm(force_disable=False) is sentinel_comm


def test_capture_warnings_as_picklable_records_message_and_category():
    def trigger():
        warnings.warn("boom", RuntimeWarning)
        return 42

    result, captured = capture_warnings_as_picklable(trigger)
    assert result == 42
    assert captured == [("boom", "RuntimeWarning")]


def test_capture_warnings_as_picklable_records_duplicates_independently():
    def trigger():
        warnings.warn("same message", RuntimeWarning)
        warnings.warn("same message", RuntimeWarning)

    _, captured = capture_warnings_as_picklable(trigger)
    assert captured == [("same message", "RuntimeWarning")] * 2


def test_capture_warnings_as_picklable_no_warnings():
    result, captured = capture_warnings_as_picklable(lambda: "ok")
    assert result == "ok"
    assert captured == []


def test_replay_warning_reissues_with_correct_category():
    with pytest.warns(RuntimeWarning, match="replayed message"):
        replay_warning("replayed message", "RuntimeWarning")


def test_replay_warning_unknown_category_falls_back_to_user_warning():
    with pytest.warns(UserWarning, match="mystery"):
        replay_warning("mystery", "SomeCategoryNotInLookup")


_LAUNCHER_ENV_VARS = (
    "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
    "PMI_SIZE", "PMI_RANK",
    "SLURM_NTASKS", "SLURM_PROCID",
)


def _clear_launcher_env(monkeypatch, keep=()):
    for var in _LAUNCHER_ENV_VARS:
        if var not in keep:
            monkeypatch.delenv(var, raising=False)


def test_detected_external_launcher_none_when_no_env_vars(monkeypatch):
    _clear_launcher_env(monkeypatch)
    assert mpi_utils._detected_external_launcher() is None


def test_detected_external_launcher_reads_ompi_vars(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    assert mpi_utils._detected_external_launcher() == (4, 2)


def test_detected_external_launcher_reads_pmi_vars(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("PMI_SIZE", "3")
    monkeypatch.setenv("PMI_RANK", "1")
    assert mpi_utils._detected_external_launcher() == (3, 1)


def test_detected_external_launcher_reads_slurm_vars(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "5")
    assert mpi_utils._detected_external_launcher() == (8, 5)


def test_detected_external_launcher_slurm_rank_defaults_to_zero(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert mpi_utils._detected_external_launcher() == (8, 0)


class _FakeComm:
    """A comm stub for raise_consistently(): bcast() returns a fixed,
    caller-supplied value regardless of the rank calling it or what was
    passed in, simulating "this is what rank 0 actually broadcast"
    without needing real multi-process communication."""

    def __init__(self, rank, size, bcast_returns=None):
        self._rank = rank
        self._size = size
        self._bcast_returns = bcast_returns

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def bcast(self, obj, root=0):
        return self._bcast_returns


def test_raise_consistently_single_rank_noop_when_no_exception():
    raise_consistently(_FakeComm(rank=0, size=1), None)  # must not raise


def test_raise_consistently_single_rank_reraises_local_exception():
    with pytest.raises(ValueError, match="boom"):
        raise_consistently(_FakeComm(rank=0, size=1), ValueError("boom"))


def test_raise_consistently_multi_rank_root_success_stays_quiet():
    raise_consistently(_FakeComm(rank=0, size=4, bcast_returns=False), None)


def test_raise_consistently_multi_rank_root_failure_reraises_original_on_root():
    with pytest.raises(ValueError, match="root failed"):
        raise_consistently(
            _FakeComm(rank=0, size=4, bcast_returns=True),
            ValueError("root failed"))


def test_raise_consistently_multi_rank_worker_quiet_when_root_succeeds():
    raise_consistently(_FakeComm(rank=2, size=4, bcast_returns=False), None)


def test_raise_consistently_multi_rank_worker_raises_when_root_fails():
    with pytest.raises(RuntimeError, match="Rank 0 failed"):
        raise_consistently(_FakeComm(rank=2, size=4, bcast_returns=True), None)


def test_capture_then_replay_round_trip_deduplicates_like_normal_warnings():
    """End-to-end: two independent captures of the identical warning,
    replayed under the default filter, should collapse to one visible
    warning -- exactly like a single-process run that triggers the same
    warnings.warn() call twice would (Python's default 'once' filter)."""

    def trigger():
        warnings.warn("dup", RuntimeWarning)

    _, captured_a = capture_warnings_as_picklable(trigger)
    _, captured_b = capture_warnings_as_picklable(trigger)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default")
        for message, category in captured_a + captured_b:
            replay_warning(message, category)

    assert len(seen) == 1
