import subprocess, sys, json, os, textwrap

def _make_circuit_py(path):
    open(path,"w").write(textwrap.dedent("""
    from qx import Circuit
    def build():
        with Circuit() as qc:
            a,b = qc.allocate(2)
            qc.h(a); qc.cx(a,b); qc.measure(a,b)
        return qc
    """))

def test_qx_run(tmp_path):
    src = tmp_path/"bell.py"; _make_circuit_py(src)
    out = subprocess.run([sys.executable, "bin/qx-run", "--module", str(src), "--target", "sim-local", "--shots", "64"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "Counts" in out.stdout or "Saved" in out.stdout

def test_qx_report(tmp_path):
    # needs at least one artifact existing
    subprocess.run([sys.executable, "bin/qx-report"], check=False)
    out = subprocess.run([sys.executable, "bin/qx-report"], capture_output=True, text=True)
    assert out.returncode == 0
