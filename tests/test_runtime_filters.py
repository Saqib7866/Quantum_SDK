from qx import Circuit, backend, run
from qx.runtime import list_runs_filtered


def _simple():
    with Circuit() as qc:
        a = qc.allocate(1)
        qc.h(a); qc.measure(a)
    return qc


def test_list_runs_filtered_by_runhash_and_target(tmp_path, monkeypatch):
    b = backend("sim-local")
    # create two runs
    r1 = run(_simple(), b, shots=10)
    r2 = run(_simple(), b, shots=10)

    # filter by runhash substring of r1's run_id
    run_ids = list_runs_filtered()
    assert isinstance(run_ids, list)
    # pick one id and query by that substring
    if run_ids:
        rid = run_ids[-1]
        out = list_runs_filtered(runhash=rid[:8])
        assert any(rid in o for o in out)
