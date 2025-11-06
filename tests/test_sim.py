from qx import Circuit, backend, run

def test_hadamard_fair_coin():
    with Circuit() as qc:
        (q0,) = qc.allocate(1)
        qc.h(q0); qc.measure(q0)
    counts = run(qc, backend("sim-local"), shots=2000).result().counts
    p1 = counts.get("1", 0)/2000
    assert 0.35 < p1 < 0.65
