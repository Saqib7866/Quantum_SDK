# Getting Started

## Install
```bash
pip install -e .

from qx import Circuit, backend, run, draw_text

b = backend("sim-local")
with Circuit() as qc:
    a,bq = qc.allocate(2)
    qc.h(a); qc.cx(a,bq); qc.measure(a,bq)

print(draw_text(qc))
print(run(qc, b, shots=256).result().counts)

## Programmatic configuration with `qx.config`

You can programmatically configure runtime settings without touching environment variables.

Example:

```python
from qx.config import set_max_qubits, set_default_shots, set_runs_dir
from qx import Circuit, backend, run

# prefer programmatic overrides in your scripts/tests
set_max_qubits(12)
set_default_shots(2048)
set_runs_dir('/var/qx_runs')

b = backend('sim-local')
with Circuit() as qc:
    q = qc.allocate(3)
    qc.h(q[0]); qc.measure(*q)

res = run(qc, b, shots=set_default_shots()).result()
print(res.metadata)
```

Environment fallback

If you prefer environment-driven configuration, `qx.config.get_max_qubits()` will still read `QX_MAX_QUBITS` when no programmatic override is set. Additionally, you can control the runs directory via `QX_RUNS_DIR` environment variable (used by `qx.config.get_runs_dir()`):

```bash
export QX_RUNS_DIR=/tmp/qx_runs
export QX_MAX_QUBITS=16
python my_run_script.py
```

For more details see the Configuration page in the docs: `Configuration <./config.md>`_.


### `docs/quickstart.md`
```bash
cat > docs/quickstart.md <<'EOF'
# Quickstart

## Run on sim-local
```python
from qx import Circuit, backend, run
b = backend("sim-local")
with Circuit() as qc:
    a,b = qc.allocate(2)
    qc.h(a); qc.cx(a,b); qc.measure(a,b)
res = run(qc, b, shots=512).result()
print(res.counts)

from qx import Circuit, backend, run
b = backend("targets/zenaquantum-alpha.json")
with Circuit() as qc:
    a,bq = qc.allocate(2)
    qc.h(a); qc.cx(a,bq); qc.measure(a,bq)
print(run(qc, b, shots=256).result().metadata)

from qx import draw_text, plot_counts, estimate_resources
print(draw_text(qc))
plot_counts(res.counts, title="Bell")
print(estimate_resources(qc.program, b["caps"]))


### `docs/targets.md`
```bash
cat > docs/targets.md <<'EOF'
# Targets & Devices

A target profile is JSON:
```json
{
  "name": "zenaquantum-alpha",
  "native_gates": ["h","x","y","z","rx","ry","rz","cx","swap","measure"],
  "n_qubits_max": 5,
  "connectivity": [[0,1],[1,2],[2,3],[3,4]],
  "durations_ns": { "h": 25, "x": 20, "cx": 200, "swap": 250, "measure": 400 },
  "readout_error": 0.02,
  "oneq_error": 0.001,
  "twoq_error": 0.01
}

from qx import backend
b = backend("targets/zenaquantum-alpha.json")


### `docs/cli.md`
```bash
cat > docs/cli.md <<'EOF'
# CLI

## qx-run
```bash
./bin/qx-run --backend sim-local --shots 256
./bin/qx-run --backend targets/zenaquantum-alpha.json --shots 256

./bin/qx-report --list
./bin/qx-report --run <RUN_ID>


### `docs/changelog.md`
```bash
cat > docs/changelog.md <<'EOF'
# Changelog

## Stage 5 – Visualization & Dev Tools
- ASCII & Matplotlib circuit drawers
- Histogram result plotter
- Resource estimator (depth/gates/fidelity)
- MkDocs docs site

## Stage 4 – Runtime & Job Model
- Job API (submit/status/result), artifacts, sweeps, CLI

## Stage 3 – Transpiler & Capabilities
- Target profiles, mapping/routing/opt, reports, noise hooks

## Stage 2 – Core Library
- IR, circuit API, local state-vector sim

## Stage 1 – Environment
- WSL2 dev env, Python/Rust toolchains, tests green
