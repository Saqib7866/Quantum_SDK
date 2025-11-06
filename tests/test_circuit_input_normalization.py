from qx import Circuit


def test_single_qubit_tuple_acceptance():
    # allocate(1) returns a tuple; ensure passing that tuple to gates works
    with Circuit() as qc:
        q = qc.allocate(1)
        # q is a tuple like (0,)
        qc.h(q)
        qc.x(q[0])
        qc.measure(q)
    prog = qc.program
    # Expect three ops: h, x, measure
    names = [op.name for op in prog.ops]
    assert names == ["h", "x", "measure"]
