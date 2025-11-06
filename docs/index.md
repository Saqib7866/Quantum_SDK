# QX-IR Documentation

Welcome to the QX-IR documentation! QX-IR is a quantum computing framework designed for easy quantum circuit design, simulation, and analysis.

## Features

- **Intuitive API**: Design quantum circuits with a clean, Pythonic interface
- **Multiple Backends**: Simulate circuits using different backends
- **Visualization**: Built-in tools for circuit visualization
- **Resource Estimation**: Analyze circuit requirements and estimate errors
- **Extensible**: Easy to extend with new gates, operations, and backends

## Quick Start

```python
from qx_ir import Circuit, Op, StatevectorSimulator

# Create a simple Bell state circuit
circuit = Circuit(n_qubits=2)
circuit.add_op(Op('h', qubits=[0]))
circuit.add_op(Op('cx', qubits=[0, 1]))

# Simulate the circuit
simulator = StatevectorSimulator()
result = simulator.run(circuit, shots=1000)
print(result.get_counts())  # Should show ~50% |00> and 50% |11>
```

## Getting Started

- [Installation](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [Core Concepts](guide/core-concepts.md)

## Examples

- [Basic Quantum Circuits](examples/basic-circuits.md)
- [Quantum Teleportation](examples/teleportation.md)
- [Quantum Fourier Transform](examples/qft.md)

## API Reference

- [Core Module](api/core.md)
- [Simulator](api/simulator.md)
- [Visualization](api/visualization.md)
- [Resource Estimation](api/estimator.md)
