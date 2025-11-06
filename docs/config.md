# qx.config

Use `qx.config` to centrally configure runtime behavior. This module exposes
simple getters and setters so you can programmatically control runtime
settings without scattering env-var writes across the codebase.

API (high level)

- `qx.config.get_max_qubits()` / `qx.config.set_max_qubits(n)`
  - In-memory override takes precedence; if no override is set the
    environment variable `QX_MAX_QUBITS` is read; finally a default is used.
- `qx.config.get_default_shots()` / `set_default_shots(n)` — default shot count used by tools.
- `qx.config.get_runs_dir()` / `set_runs_dir(path)` — control where run artifacts are stored.
- `qx.config.get_logging_level()` / `set_logging_level(level)` — optional runtime logging hint.

Example

```python
from qx.config import set_max_qubits, set_default_shots

set_max_qubits(20)        # prefer this over directly manipulating os.environ
set_default_shots(2048)

from qx import backend, Circuit, run
bk = backend("sim-local")
with Circuit() as qc:
    q = qc.allocate(3)
    qc.h(q[0]); qc.measure(*q)
res = run(qc, bk, shots=get_default_shots()).result()
```

Notes

- For backwards compatibility some older code may still read `QX_MAX_QUBITS` from the environment. Prefer the `qx.config` API in new code.
- `set_runs_dir()` accepts a relative or absolute path and will not create the directory until `get_runs_dir()` is called.
