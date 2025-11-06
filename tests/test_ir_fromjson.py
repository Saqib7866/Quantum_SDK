import json
from qx.circuit import Circuit
from qx.ir import Program

def test_program_roundtrip():
    with Circuit() as qc:
        a, b = qc.allocate(2)
        qc.h(a); qc.cx(a, b); qc.measure(a, b)
    prog = qc.program
    s = prog.to_json_canonical()
    p2 = Program.from_json(s)
    assert p2.n_qubits == prog.n_qubits
    assert len(p2.ops) == len(prog.ops)
    # verify op names
    assert [o.name for o in p2.ops] == [o.name for o in prog.ops]
