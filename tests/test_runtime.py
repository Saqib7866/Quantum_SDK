from qx import Circuit, backend, run

def test_bell_pairs():
    with Circuit() as qc:
        a,b = qc.allocate(2)
        qc.h(a); qc.cx(a,b); qc.measure(a,b)
    counts = run(qc, backend("sim-local"), shots=2000).result().counts
    p = (counts.get("00",0)+counts.get("11",0))/2000
    assert p > 0.90
