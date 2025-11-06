import pytest
from qx.ir import Program, Op

def test_invalid_qubit_index():
    p = Program(n_qubits=1, n_clbits=1)
    p.ops.append(Op("x", (2,), ()))
    with pytest.raises(Exception):
        # execution should fail due to bad index
        from qx.sim.local import LocalSimulator
        LocalSimulator().execute(p, shots=1)

def test_empty_circuit_measureless():
    p = Program(n_qubits=1, n_clbits=0)
    from qx.sim.local import LocalSimulator
    counts, meta = LocalSimulator().execute(p, shots=1)
    assert meta["notes"] == "no measurement"

def test_mismatched_clbits():
    p = Program(n_qubits=1, n_clbits=0)
    p.ops.append(Op("measure",(0,),(),(0,)))
    from qx.sim.local import LocalSimulator
    with pytest.raises(Exception):
        LocalSimulator().execute(p, shots=1)
