# cli

## Runtime overrides: QX_MAX_QUBITS and seed

You can control the maximum number of qubits usable by the local simulator and provide a deterministic RNG seed for runs.

- QX_MAX_QUBITS (env var): set this environment variable to change the default maximum number of qubits accepted by the local state-vector simulator. The `backend("sim-local")` registry will also report `caps["n_qubits_max"]` as this value unless a target JSON explicitly sets `n_qubits_max`.

	Example (bash):
	```bash
	export QX_MAX_QUBITS=20
	python my_run_script.py
	```

- Seed: the runtime accepts an integer `seed` which is passed to the backend runner to make RNGs deterministic (both Python `random` and `numpy`). When you pass `--seed` via the CLI or `seed=` to `run(...)`, the value is recorded in `result().metadata["seed"]`.

CLI flags

- `--max-qubits N` — process-level override (sets `QX_MAX_QUBITS` for the run)
- `--seed S` — deterministic RNG seed passed to the backend

Examples

Run a module with a 16-qubit limit and deterministic seed:
```bash
export QX_MAX_QUBITS=16
./bin/qx-run --module examples/my_circ.py --target sim-local --shots 1024 --seed 42
```

Or use the per-invocation flags:
```bash
./bin/qx-run --module examples/my_circ.py --target sim-local --shots 1024 --max-qubits 16 --seed 42
```

Notes

- If you pass a backend JSON that contains `n_qubits_max`, that value will be used for the backend caps instead of the environment default.
- The `seed` is recorded in the run artifact metadata so you can reproduce runs later.
