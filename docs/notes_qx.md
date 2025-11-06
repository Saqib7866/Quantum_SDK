# Notes: QX runtime and simulator

The local state-vector simulator stores the full quantum state in memory. Its memory usage grows roughly as 16 bytes * 2^n per complex amplitude (depends on Python/numpy). Increasing `QX_MAX_QUBITS` can quickly exhaust RAM, so use it with care on resource-constrained machines.

To allow larger circuits for a single run you can either set the environment variable:

```bash
export QX_MAX_QUBITS=20
python my_run_script.py
```

or use the CLI flag:

```bash
./bin/qx-run --module examples/my_circ.py --max-qubits 20
```
