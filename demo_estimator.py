"""
Demo script for the Quantum Resource Estimator.

This script demonstrates how to use the ResourceEstimator class to analyze
quantum circuits and estimate their resource requirements and error rates.
"""
from qx_ir import Circuit, Op, ResourceEstimator

def create_quantum_teleportation() -> Circuit:
    """Create a quantum teleportation circuit."""
    circuit = Circuit(n_qubits=3)
    
    # Prepare the state to teleport (qubit 0)
    circuit.add_op(Op('h', qubits=[0]))
    circuit.add_op(Op('x', qubits=[0]))
    
    # Create Bell pair (qubits 1 and 2)
    circuit.add_op(Op('h', qubits=[1]))
    circuit.add_op(Op('cx', qubits=[1, 2]))
    
    # Bell measurement
    circuit.add_op(Op('cx', qubits=[0, 1]))
    circuit.add_op(Op('h', qubits=[0]))
    
    # Add measurement operations
    circuit.add_op(Op('measure', qubits=[0]))
    circuit.add_op(Op('measure', qubits=[1]))
    
    # Conditional operations (classical control would be handled by the quantum processor)
    circuit.add_op(Op('x', qubits=[2]))  # X gate on qubit 2
    circuit.add_op(Op('z', qubits=[2]))  # Z gate on qubit 2
    
    return circuit

def create_quantum_fourier_transform(n_qubits: int = 3) -> Circuit:
    """Create a quantum Fourier transform circuit."""
    circuit = Circuit(n_qubits=n_qubits)
    
    for i in range(n_qubits):
        circuit.add_op(Op('h', qubits=[i]))
        for j in range(i + 1, n_qubits):
            # Add controlled phase gates
            angle = 2 * 3.14159 / (2 ** (j - i + 1))
            circuit.add_op(Op('cp', qubits=[j, i], params=[angle]))
    
    # Swap qubits to match QFT definition
    for i in range(n_qubits // 2):
        circuit.add_op(Op('swap', qubits=[i, n_qubits - 1 - i]))
    
    return circuit

def main():
    """Run the resource estimation demo."""
    print("🚀 Quantum Resource Estimator Demo\n")
    
    # Create an estimator with default error rates
    estimator = ResourceEstimator()
    
    # Example 1: Simple Bell state circuit
    print("1️⃣  Bell State Circuit")
    bell_circuit = Circuit(n_qubits=2)
    bell_circuit.add_op(Op('h', qubits=[0]))
    bell_circuit.add_op(Op('cx', qubits=[0, 1]))
    
    # Estimate resources
    bell_estimate = estimator.estimate(bell_circuit)
    print(bell_estimate)
    
    # Example 2: Quantum Teleportation
    print("\n2️⃣  Quantum Teleportation Circuit")
    teleportation = create_quantum_teleportation()
    teleport_estimate = estimator.estimate(teleportation)
    print(teleport_estimate)
    
    # Example 3: Quantum Fourier Transform
    print("\n3️⃣  3-Qubit Quantum Fourier Transform")
    qft_circuit = create_quantum_fourier_transform(3)
    qft_estimate = estimator.estimate(qft_circuit)
    print(qft_estimate)
    
    # Example 4: Custom error rates
    print("\n4️⃣  Custom Error Rates")
    custom_errors = {
        'h': 1e-5,  # More accurate H gates
        'cx': 1e-4,  # More accurate CNOT gates
        'measure': 5e-4  # Less accurate measurements
    }
    custom_estimator = ResourceEstimator(error_rates=custom_errors)
    custom_estimate = custom_estimator.estimate(qft_circuit)
    print("With custom error rates:")
    print(f"  - H gate error: {custom_errors['h']:.1e}")
    print(f"  - CX gate error: {custom_errors['cx']:.1e}")
    print(f"  - Measurement error: {custom_errors['measure']:.1e}")
    print("\nEstimate:")
    print(custom_estimate)
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
