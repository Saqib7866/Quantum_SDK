from qx import Circuit, backend
from qx.runtime import run, list_runs, load_run, sweep_circuits

def test_job_artifact_and_load():
    b = backend("sim-local")
    with Circuit() as qc:
        a,bq = qc.allocate(2)
        qc.h(a); qc.cx(a,bq); qc.measure(a,bq)
    j = run(qc, b, shots=128)
    rid = j.run_id()  # load exactly the run we just created
    res = load_run(rid)
    assert sum(res.counts.values()) == 128

def test_sweep_circuits():
    b = backend("sim-local")
    circs = []
    for _ in range(3):
        with Circuit() as qc:
            (q0,) = qc.allocate(1); qc.h(q0); qc.measure(q0)
        circs.append(qc)
    out = sweep_circuits(circs, b, shots=64)
    assert len(out) == 3
    for r in out:
        assert sum(r.counts.values()) == 64
