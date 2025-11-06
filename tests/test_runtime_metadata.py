import os
from qx import Circuit, backend, run


def _build_circ():
    with Circuit() as qc:
        q = qc.allocate(3)
        qc.h(q[0])
        qc.measure(*q)
    return qc


def test_runtime_metadata_and_seed(tmp_path, monkeypatch):
    # run in isolated tmpdir so .qx_runs doesn't collide with other tests
    monkeypatch.chdir(tmp_path)

    qc = _build_circ()
    bk = backend("sim-local")

    # run without seed: metadata should include keys and seed == None
    job = run(qc, bk, shots=20)
    meta = job.result().metadata
    for k in ("shots", "n_qubits", "clbits", "noise", "target_name", "run_started_at", "seed"):
        assert k in meta
    assert meta["seed"] is None

    # run twice with same seed and same shots -> deterministic counts
    j1 = run(qc, bk, shots=50, seed=42)
    j2 = run(qc, bk, shots=50, seed=42)
    assert j1.result().metadata["seed"] == 42
    assert j1.result().counts == j2.result().counts
