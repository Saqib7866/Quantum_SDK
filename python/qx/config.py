"""Central configuration for QX runtime.

Provide a single place to read and override runtime settings such as the
maximum number of qubits supported by the local simulator. This keeps
environment handling and defaults centralized.
"""
from typing import Optional
import os

_DEFAULT_MAX_QUBITS = 5
# in-memory overrides (preferred when set programmatically)
_override_max_q: Optional[int] = None
_default_shots: int = 1024
_runs_dir: Optional[str] = None
_logging_level: Optional[str] = None


def get_max_qubits() -> int:
    """Return the configured max qubits.

    Priority: in-memory override > QX_MAX_QUBITS env var > default.
    """
    global _override_max_q
    if _override_max_q is not None:
        return _override_max_q
    try:
        return int(os.environ.get("QX_MAX_QUBITS", _DEFAULT_MAX_QUBITS))
    except Exception:
        return _DEFAULT_MAX_QUBITS


def set_max_qubits(n: Optional[int]):
    """Set a programmatic override for max qubits.

    If n is None the override is cleared and the env var is removed.
    This function only sets an in-memory override. Existing code that
    reads the QX_MAX_QUBITS environment variable will continue to work
    if that env var is set externally, but callers should prefer the
    config API for programmatic control.
    """
    global _override_max_q
    if n is None:
        _override_max_q = None
    else:
        _override_max_q = int(n)


def get_default_shots() -> int:
    return _default_shots


def set_default_shots(n: int):
    global _default_shots
    _default_shots = int(n)


def get_runs_dir() -> str:
    global _runs_dir
    if _runs_dir:
        return _runs_dir
    # allow environment override for project-level persistent runs
    env = os.environ.get("QX_RUNS_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    d = os.path.join(os.getcwd(), ".qx_runs")
    os.makedirs(d, exist_ok=True)
    return d


def set_runs_dir(path: str):
    global _runs_dir
    _runs_dir = path


def get_logging_level() -> Optional[str]:
    return _logging_level


def set_logging_level(level: Optional[str]):
    global _logging_level
    _logging_level = level


def reset_all():
    """Reset all in-memory overrides (useful in tests)."""
    set_max_qubits(None)
    set_default_shots(1024)
    set_runs_dir(None)
    set_logging_level(None)
