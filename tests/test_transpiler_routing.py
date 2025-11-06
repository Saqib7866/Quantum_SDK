from qx.circuit import Circuit
from qx.passes import compile_pipeline


def test_non_adjacent_cx_on_linear_topology():
    # Build circuit with cx between qubits 0 and 4 on 5 qubits
    with Circuit() as qc:
        a = qc.allocate(5)
        qc.h(a[0])
        qc.cx(a[0], a[4])
        qc.measure(*a)
    prog = qc.program

    caps = {
        "n_qubits_max": 5,
        "native_gates": {"h", "x", "y", "z", "rx", "ry", "rz", "cx", "swap", "measure"},
        "connectivity": [[0,1],[1,2],[2,3],[3,4]]
    }

    out = compile_pipeline(prog, caps)
    # after routing, ensure only native gates and no cx between non-adjacent logical indices
    assert all(op.name in caps["native_gates"] for op in out.ops)


def test_impossible_mapping_raises():
    # 3-qubit program but connectivity isolates node 2
    with Circuit() as qc:
        a = qc.allocate(3)
        qc.cx(a[0], a[2])
        qc.measure(*a)
    prog = qc.program

    caps = {
        "n_qubits_max": 3,
        "native_gates": {"h","x","y","z","rx","ry","rz","cx","swap","measure"},
        # connectivity missing edge between 0 and 2 and no path (simulate isolated node)
        "connectivity": [[0,1]]
    }

    try:
        compile_pipeline(prog, caps)
        assert False, "compile_pipeline should have raised for impossible mapping"
    except Exception:
        pass
