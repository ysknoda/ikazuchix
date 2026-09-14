"""Optional MPI parallelization with a transparent single-process fallback.

If mpi4py is installed AND a working MPI runtime is available,
MPI.COMM_WORLD is used to distribute (defect, charge_state) work items
across ranks. Otherwise a NullComm stand-in makes the rest of the code
behave exactly as a single rank of size 1 -- this is also the path
taken when --no-mpi is given even though real MPI is available.

Two independent failure modes are handled:
  - ImportError : mpi4py itself is not installed.
  - RuntimeError : mpi4py IS installed (as a plain Python package) but
    its native MPI backend could not be loaded at import time --
    observed directly on this project's dev machine, where mpi4py is
    present via pip/conda but no system MPI runtime (OpenMPI/MPICH,
    providing libmpi.dylib/.so) is installed. mpi4py >= 4 resolves its
    MPI backend by dynamically loading a system library at
    ``from mpi4py import MPI`` time, so the plain package import can
    succeed while this second import still fails.
"""

import os
import warnings

try:
    from mpi4py import MPI
    HAS_MPI = True
except (ImportError, RuntimeError):
    MPI = None
    HAS_MPI = False


def _detected_external_launcher():
    """Best-effort (size, rank) from OpenMPI/MPICH/Slurm environment
    variables that the launcher sets on every spawned process *before*
    MPI_Init -- the only way this process can tell it was actually
    started via a real multi-rank `mpirun`/`mpiexec`/`srun` when mpi4py
    itself is unusable (not installed, or its native MPI library failed
    to load) here, e.g. a venv/conda env that has an mpirun binary on
    PATH but no mpi4py. Slurm's native launch (`srun` without going
    through mpirun) sets SLURM_NTASKS/SLURM_PROCID instead of the
    OpenMPI/PMI variables above, so without this every rank would
    otherwise believe itself to be rank 0 and race to write the same
    output files."""
    for size_var, rank_var in (("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK"),
                                ("PMI_SIZE", "PMI_RANK"),
                                ("SLURM_NTASKS", "SLURM_PROCID")):
        if size_var in os.environ:
            try:
                return int(os.environ[size_var]), int(os.environ.get(rank_var, 0))
            except ValueError:
                continue
    return None


class NullComm:
    """Degenerate single-process communicator (rank 0 of size 1).

    Implements just the subset of the mpi4py.MPI.Comm interface this
    package actually uses, so calling code can treat a real
    MPI.COMM_WORLD and this fallback uniformly.
    """

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def Barrier(self):
        pass


def get_comm(force_disable=False):
    """Return a communicator: real MPI.COMM_WORLD when mpi4py loaded
    successfully and not disabled, otherwise a single-process NullComm."""
    if force_disable:
        return NullComm()
    if HAS_MPI:
        return MPI.COMM_WORLD

    launcher = _detected_external_launcher()
    if launcher is not None:
        size, rank = launcher
        if size > 1 and rank == 0:
            warnings.warn(
                f"This process appears to be launched under an MPI job "
                f"with {size} ranks, but mpi4py is not usable in this "
                "Python environment (not installed, or its native MPI "
                "library could not be loaded) -- falling back to "
                f"{size} independent single-process runs, each "
                "redundantly computing and writing the full output. "
                "Install mpi4py (with a working MPI runtime) in this "
                "environment to enable proper MPI distribution.",
                RuntimeWarning, stacklevel=2)
    return NullComm()


# ---------------------------------------------------------------------------
# Collective-boundary-safe exception propagation
#
# A rank-0-only failure (e.g. load_input() raising, or run_all_defects()
# raising DefectComputationError -- itself only ever raised on rank 0,
# after gathering) must never be allowed to simply propagate on rank 0
# alone: every other rank is typically about to enter (or already past)
# a collective call expecting rank 0's participation, and once rank 0
# has exited, that collective can never complete -- the other ranks
# would hang forever. raise_consistently() closes that gap generically.
# ---------------------------------------------------------------------------

def raise_consistently(comm, exc):
    """Broadcast whether `exc` (expected to be set on at most rank 0 in
    the common case -- see module docstring above) is present, then
    raise -- on every rank, not just whichever one(s) actually failed --
    so no rank is ever left blocked at a later collective call waiting
    for a peer that has already exited.

    A single-rank comm (real size-1 MPI, or the NullComm fallback)
    skips the broadcast and just re-raises `exc` locally, so this is a
    no-op wrapper on the default, non-MPI code path. Must be called by
    every rank, the same number of times in the same order, exactly
    like any other collective operation.
    """
    if comm.Get_size() > 1:
        rank0_failed = comm.bcast(exc is not None, root=0)
        if rank0_failed and comm.Get_rank() != 0 and exc is None:
            raise RuntimeError(
                "Rank 0 failed; see rank 0's output for the original "
                "error.  Aborting here too, to avoid hanging.")
    if exc is not None:
        raise exc


# ---------------------------------------------------------------------------
# Warning capture / replay
#
# Ewald non-convergence warnings raised while computing a work item
# assigned to a non-root rank would otherwise never reach the user (only
# rank 0 prints anything). Each rank instead captures its own warnings as
# plain (message, category_name) tuples -- picklable, so they travel with
# the rest of that work item's result through comm.gather() -- and rank 0
# re-issues them all after gathering, via a single fixed call site so that
# Python's normal duplicate-warning suppression still collapses identical
# messages exactly as it would in a single-process run.
# ---------------------------------------------------------------------------

_WARNING_CATEGORIES = {"RuntimeWarning": RuntimeWarning, "UserWarning": UserWarning}


def capture_warnings_as_picklable(func, *args, **kwargs):
    """Call func(*args, **kwargs), returning (result, captured) where
    captured is a list of (message_str, category_name) tuples for every
    warning raised during the call (duplicates included)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    captured = [(str(w.message), w.category.__name__) for w in caught]
    return result, captured


def replay_warning(message, category_name, stacklevel=2):
    """Re-issue a warning captured (possibly on another MPI rank) via
    capture_warnings_as_picklable, under the ambient (default) warning
    filter so normal duplicate suppression applies."""
    category = _WARNING_CATEGORIES.get(category_name, UserWarning)
    warnings.warn(message, category, stacklevel=stacklevel)
