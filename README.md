# Quantum SDK

A quantum computing framework for circuit execution and analysis.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Saqib7866/Quantum_SDK.git
   cd Quantum_SDK
   ```
2. Install in development mode:

   ```bash
   pip install -e .
   ```

## Usage

### Command Line Interface

Run a quantum circuit:

```bash
qx run test_circuits/bell.json --shots 1000
```

Check job status:

```bash
qx status [JOB_ID]
```

Generate a report for a job:

```bash
qx report JOB_ID
```

### Python API

```python
from qx_ir import Circuit, Op, Program, LocalBackend

# Create a circuit
circuit = Circuit(n_qubits=2)
circuit.add_instruction(Op('h', [0]))
circuit.add_instruction(Op('cx', [0, 1]))

# Create a program
program = Program([circuit], {'shots': 1000})

# Run on local backend
backend = LocalBackend()
job = backend.submit(program)
result = job.result()

print(result)
```
