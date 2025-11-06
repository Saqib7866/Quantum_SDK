import pytest
from qx import Circuit
from qx.passes import compile_pipeline

def test_routing_linear_connectivity():
    caps = {"n_qubits_max":5, "native_gates":{"h","cx","measure","swap"}, "connectivity":[[0,1],[1,2],[2,3],[3,4]]}
    with Circuit() as qc:
        q = qc.allocate(5)
        qc.h(q[0]); qc.cx(q[0], q[4]); qc.measure(q[0], q[4])
    prog = compile_pipeline(qc.program, caps)
    # should have swaps inserted
    assert any(op.name=="swap" for op in prog.ops)

def test_impossible_mapping_raises():
    caps = {"n_qubits_max":2, "native_gates":{"h","cx","measure"}, "connectivity":[[0,1]]}
    # requesting 3 qubits should fail
    with Circuit() as qc:
        q = qc.allocate(3)
        qc.h(q[0]); qc.measure(q[0])
    with pytest.raises(ValueError):
        compile_pipeline(qc.program, caps)
