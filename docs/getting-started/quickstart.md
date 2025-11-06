# Quick Start Guide

This guide will walk you through the basics of using QX-IR to create, simulate, and visualize quantum circuits.

## Creating a Simple Circuit

Let's start by creating a simple quantum circuit that creates a Bell state (entangled state):

```python
from qx_ir import Circuit, Op

# Create a quantum circuit with 2 qubits
circuit = Circuit(n_qubits=2)

# Add a Hadamard gate on qubit 0
circuit.add_op(Op('h', qubits=[0]))

# Add a CNOT gate with control on qubit 0 and target on qubit 1
circuit.add_op(Op('cx', qubits=[0, 1]))

# Print the circuit
print("Circuit:")
print(circuit)
```

## Simulating the Circuit

Now, let's simulate this circuit using the statevector simulator:

```python
from qx_ir import StatevectorSimulator

# Create a simulator instance
simulator = StatevectorSimulator()

# Run the simulation with 1000 shots (measurements)
result = simulator.run(circuit, shots=1000)

# Get the measurement counts
counts = result.get_counts()
print("\nMeasurement results:")
for state, count in counts.items():
    print(f"|{state}⟩: {count}")
```

## Visualizing the Circuit

You can visualize the circuit using the built-in visualization tools:

```python
from qx_ir.visualization import CircuitDrawer

# Draw the circuit as text
print("\nText representation:")
print(CircuitDrawer.draw_text(circuit))

# Save the circuit as an image (requires matplotlib)
CircuitDrawer.draw_mpl(circuit, filename="bell_circuit.png", show=False)
print("\nCircuit diagram saved as 'bell_circuit.png'")
```

## Estimating Resources

You can estimate the resources required to run a circuit:

```python
from qx_ir import ResourceEstimator

# Create a resource estimator
estimator = ResourceEstimator()

# Estimate resources for our circuit
estimate = estimator.estimate(circuit)

# Print the estimate
print("\nResource Estimate:")
print(estimate)
```

## Next Steps

Now that you've learned the basics, you can explore more advanced features:

- [Core Concepts](guide/core-concepts.md): Learn about the fundamental concepts in QX-IR
- [Quantum Operations](guide/operations.md): Explore the available quantum operations
- [Visualization](guide/visualization.md): Learn how to create custom visualizations
- [Resource Estimation](guide/resource-estimation.md): Understand how to analyze circuit resources

For more examples, check out the [Examples](../examples/basic-circuits.md) section.
