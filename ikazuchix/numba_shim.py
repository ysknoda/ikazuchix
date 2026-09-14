"""Optional numba JIT acceleration with a transparent no-op fallback.

If numba is installed, ``njit`` is ``numba.njit`` itself; kernels are
still decorated with it unconditionally (numba's default lazy
compilation defers the actual machine-code build to the *first call*,
not to decoration/import time, so decorating is always cheap). If numba
is not installed, ``njit`` is a no-op decorator that returns the
original function unchanged.

Either way, the decorated function exposes the original undecorated
Python function via ``.py_func`` -- mirroring real numba's own
attribute -- so calling code can choose the pure-Python path at
*runtime* (e.g. for ``--no-numba``, or in tests) via
``use_kernel(fn)``, independent of whether numba happens to be
importable. This also enables a zero-duplication numerical-equivalence
check: ``kernel(...) == kernel.py_func(...)``.
"""

try:
    from numba import njit as _numba_njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _fallback_njit(*args, **kwargs):
    """No-op fallback supporting both ``@njit`` and ``@njit(cache=True)``."""

    def _decorate(func):
        func.py_func = func
        return func

    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Used as @njit directly (no parentheses/arguments).
        return _decorate(args[0])
    # Used as @njit(...) with keyword/positional options.
    return _decorate


njit = _numba_njit if HAS_NUMBA else _fallback_njit

# Runtime toggle (independent of whether numba is installed) so a CLI
# flag such as --no-numba can force the pure-Python kernel path even
# when JIT compilation is available. Checked lazily on each call via
# use_kernel(), never at import time.
_numba_runtime_enabled = True


def set_numba_enabled(enabled):
    """Enable/disable numba at runtime (e.g. from a --no-numba CLI flag)."""
    global _numba_runtime_enabled
    _numba_runtime_enabled = bool(enabled)


def numba_active():
    """True if numba is both installed and not runtime-disabled."""
    return HAS_NUMBA and _numba_runtime_enabled


def use_kernel(kernel):
    """Return the JIT-compiled callable, or its pure-Python .py_func,
    depending on whether numba is currently active."""
    return kernel if numba_active() else kernel.py_func
