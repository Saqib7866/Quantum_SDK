from qx.circuit import Circuit

def test_ir_hash_changes_when_ops_change():
    with Circuit() as qc:
        q0, = qc.allocate(1)
        qc.h(q0); qc.measure(q0)
    h1 = qc.program.sha256()
    with Circuit() as qc2:
        q0, = qc2.allocate(1)
        qc2.x(q0); qc2.measure(q0)
    h2 = qc2.program.sha256()
    assert h1 != h2
