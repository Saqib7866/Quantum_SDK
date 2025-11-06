import sys
import json
import tempfile
import textwrap
import subprocess
from pathlib import Path


def _write_module(tmp_path: Path, n_qubits: int) -> Path:
    p = tmp_path / "mod_build.py"
    src = textwrap.dedent(f"""
    from qx import Circuit

    def build():
        with Circuit() as qc:
            q = qc.allocate({n_qubits})
            qc.h(q[0])
            qc.measure(*q)
        return qc
    """)
    p.write_text(src)
    return p


def test_bin_qx_run_cli(tmp_path):
    # create a small module that builds a 3-qubit circuit
    mod = _write_module(tmp_path, 3)

    cmd = [sys.executable, str(Path("bin") / "qx-run"), "--module", str(mod), "--target", "sim-local", "--shots", "10", "--max-qubits", "8", "--seed", "123"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # qx-run prints a human header "Counts" and then JSON
    out = proc.stdout.strip().splitlines()
    # find first line that starts the JSON object and join to the end
    start_idx = None
    for i, line in enumerate(out):
        if line.strip().startswith("{") or line.strip().startswith("["):
            start_idx = i
            break
    if start_idx is None:
        # fallback: last line
        json_part = out[-1]
    else:
        json_part = "\n".join(out[start_idx:])
    data = json.loads(json_part)
    assert "counts" in data and "metadata" in data
    assert data["metadata"]["seed"] == 123

    # now test a circuit that uses 8 qubits and ensure --max-qubits allows it
    mod2 = _write_module(tmp_path, 8)
    cmd2 = [sys.executable, str(Path("bin") / "qx-run"), "--module", str(mod2), "--target", "sim-local", "--shots", "2", "--max-qubits", "8"]
    p2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr
