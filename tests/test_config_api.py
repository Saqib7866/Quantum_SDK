import os
from qx.config import (
    set_max_qubits,
    get_max_qubits,
    set_default_shots,
    get_default_shots,
    set_runs_dir,
    get_runs_dir,
    reset_all,
)


def test_config_getters_setters(tmp_path, monkeypatch):
    # ensure clean state
    reset_all()

    # default shots
    assert isinstance(get_default_shots(), int)

    set_default_shots(777)
    assert get_default_shots() == 777

    # max qubits override
    reset_all()
    assert get_max_qubits() is not None
    set_max_qubits(11)
    assert get_max_qubits() == 11

    # runs dir via setter
    p = str(tmp_path / "my_runs")
    set_runs_dir(p)
    assert get_runs_dir() == p

    # env override for runs dir
    reset_all()
    monkeypatch.setenv("QX_RUNS_DIR", p)
    assert get_runs_dir() == p

    # cleanup
    reset_all()
