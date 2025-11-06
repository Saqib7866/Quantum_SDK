from qx import Circuit
from qx.passes import compile_pipeline, compile_report

def test_fidelity_uses_caps_noise():
    caps1 = {"native_gates":{"h","cx","measure"}, "connectivity":"full",
             "n_qubits_max":5, "noise":{"oneq_error":0.0,"twoq_error":0.0,"readout_error":0.0}}
    caps2 = {"native_gates":{"h","cx","measure"}, "connectivity":"full",
             "n_qubits_max":5, "noise":{"oneq_error":0.02,"twoq_error":0.1,"readout_error":0.05}}
    with Circuit() as qc:
        a,b=qc.allocate(2); qc.h(a); qc.cx(a,b); qc.measure(a,b)
    p1 = compile_pipeline(qc.program, caps1)
    r1 = compile_report(p1, caps1)
    p2 = compile_pipeline(qc.program, caps2)
    r2 = compile_report(p2, caps2)
    assert "EstFidelity≈" in r1 and "EstFidelity≈" in r2
    assert r2 != r1
