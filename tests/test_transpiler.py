from qx.circuit import Circuit
from qx.passes import compile_pipeline

def ops_of(qc): return [op.name for op in qc.program.ops]

def test_cancel_double_x():
    with Circuit() as qc:
        q0, = qc.allocate(1)
        qc.x(q0); qc.x(q0); qc.measure(q0)
    prog = compile_pipeline(qc.program, {})
    names = [op.name for op in prog.ops]
    assert names.count("x") == 0

def test_merge_rz():
    with Circuit() as qc:
        q0, = qc.allocate(1)
        qc.rz(q0, 0.2); qc.rz(q0, 0.3); qc.measure(q0)
    prog = compile_pipeline(qc.program, {})
    names = [op.name for op in prog.ops]
    assert names.count("rz") == 1