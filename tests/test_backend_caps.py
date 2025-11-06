import os
from qx import backend, Circuit, run


def _build(n):
    with Circuit() as qc:
        q = qc.allocate(n)
        qc.h(q[0])
        qc.measure(*q)
    return qc


def test_backend_caps_reflect_env(monkeypatch, tmp_path):
    # run in isolated tmpdir
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("QX_MAX_QUBITS", "7")
    b = backend("sim-local")
    assert b["caps"]["n_qubits_max"] == 7

    # building a circuit that uses up to the cap should succeed
    qc_ok = _build(7)
    job = run(qc_ok, b, shots=10)
    assert job.result().metadata["n_qubits"] == 7

    # circuit exceeding the cap should raise (sim-local enforces via ValueError)
    qc_bad = _build(8)
    try:
        run(qc_bad, b, shots=1)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when running circuit exceeding QX_MAX_QUBITS"
