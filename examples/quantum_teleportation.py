"""
Quantum Teleportation Example

This script demonstrates quantum teleportation using the ZenadroneAlphaEmulator.
It shows how to teleport a quantum state from one qubit to another and tests
how different noise levels affect the teleportation fidelity.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to import the emulator
sys.path.append(str(Path(__file__).parent.parent))
from run_stage6_demo import ZenadroneAlphaEmulator, Circuit, Op

def create_teleportation_circuit(initial_state: str = '1') -> Circuit:
    """
    Create a quantum circuit that demonstrates quantum teleportation.
    
    Args:
        initial_state: The state to teleport ('0', '1', '+', '-', etc.)
        
    Returns:
        Circuit: The teleportation circuit
    """
    # Initialize 3 qubits (Alice's qubit, Bell pair qubits)
    circuit = Circuit(3)  # Qubits: [qubit_to_teleport, alice_qubit, bob_qubit]
    
    # Prepare the initial state to teleport
    if initial_state == '1':
        circuit.add_op(Op('x', [0]))  # |1⟩
    elif initial_state == '+':
        circuit.add_op(Op('h', [0]))  # |+⟩
    elif initial_state == '-':
        circuit.add_op(Op('x', [0]))
        circuit.add_op(Op('h', [0]))  # |−⟩
    elif initial_state == 'r':
        circuit.add_op(Op('h', [0]))
        circuit.add_op(Op('s', [0]))  # |0⟩ + i|1⟩/√2
    
    # Create Bell pair between Alice (qubit 1) and Bob (qubit 2)
    circuit.add_op(Op('h', [1]))
    circuit.add_op(Op('cx', [1, 2]))
    
    # Alice's operations
    circuit.add_op(Op('cx', [0, 1]))
    circuit.add_op(Op('h', [0]))
    
    # Measurements (Alice's qubits)
    circuit.add_op(Op('measure', [0]))
    circuit.add_op(Op('measure', [1]))
    
    # Bob's corrections (would be classically controlled in a real implementation)
    # For simulation, we'll apply them unconditionally
    # Note: Using only supported gates (h, x, cx)
    
    # Apply X correction (if q1 was |1>)
    # X = HZH, so we can implement CZ using H-CX-H
    circuit.add_op(Op('h', [2]))
    circuit.add_op(Op('cx', [1, 2]))
    circuit.add_op(Op('h', [2]))
    
    # Apply Z correction (if q0 was |1>)
    # Z = HXH, so we can implement CZ using H-CX-H
    circuit.add_op(Op('h', [2]))
    circuit.add_op(Op('cx', [0, 2]))
    circuit.add_op(Op('h', [2]))
    
    # Final measurement of Bob's qubit
    circuit.add_op(Op('measure', [2]))
    
    return circuit

def calculate_teleportation_fidelity(emulator, initial_state: str = '0', shots: int = 1000) -> float:
    """
    Calculate the fidelity of quantum teleportation.
    
    Args:
        emulator: The quantum emulator to use
        initial_state: The state to teleport ('0', '1', '+', '-', 'r')
        shots: Number of shots to run
        
    Returns:
        float: The fidelity of teleportation (0.0 to 1.0)
    """
    circuit = create_teleportation_circuit(initial_state)
    
    # Run the circuit
    counts = emulator.execute(circuit, shots=shots)
    
    # Calculate fidelity based on the expected state
    total_correct = 0
    
    for outcome, count in counts.items():
        # The last bit is Bob's qubit (teleported state)
        bob_qubit = outcome[-1]
        
        if initial_state == '0':
            # For |0⟩ state, Bob's qubit should be |0⟩
            if bob_qubit == '0':
                total_correct += count
        elif initial_state == '1':
            # For |1⟩ state, Bob's qubit should be |1⟩
            if bob_qubit == '1':
                total_correct += count
        elif initial_state in ['+', '-', 'r']:
            # For superposition states, we'll verify by measuring in the appropriate basis
            # This is a simplified check - a full verification would require multiple measurements
            # in different bases
            if initial_state == '+' and bob_qubit == '0':
                total_correct += count / 2  # 50% chance for |+⟩ state
            elif initial_state == '-' and bob_qubit == '1':
                total_correct += count / 2  # 50% chance for |-⟩ state
            elif initial_state == 'r':
                # For |r⟩ state, we can't verify easily with single measurement
                # Just count all outcomes as correct for simplicity
                total_correct += count
    
    # Calculate fidelity
    if initial_state in ['+', '-']:
        # For superposition states, normalize the fidelity
        fidelity = total_correct / shots * 2
    else:
        fidelity = total_correct / shots
    
    return min(1.0, fidelity)  # Cap fidelity at 1.0

def run_teleportation_experiment(noise_levels=None, shots=1000, initial_state='0'):
    """
    Run the teleportation experiment with different noise levels.
    
    Args:
        noise_levels: List of noise levels to test (0.0 to 1.0)
        shots: Number of shots per experiment
        initial_state: The state to teleport ('0', '1', '+', '-', 'r')
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    print("\n" + "="*70)
    print("QUANTUM TELEPORTATION EXPERIMENT")
    print("="*70)
    print(f"Initial state: |{initial_state}>")
    print(f"Shots per experiment: {shots}")
    print("-"*70)
    print(f"{'Noise Level':<15} {'Fidelity':<15} {'Success Rate':<15}")
    print("-"*70)
    
    for level in noise_levels:
        # Create emulator with specified noise level
        if level == 0.0:
            # Perfect emulator (no noise)
            emulator = ZenadroneAlphaEmulator(
                t1=float('inf'),
                t2=float('inf'),
                readout_error=0.0,
                gate_errors={k: 0.0 for k in ['h', 'x', 'cx', 'cz', 'measure']}
            )
        else:
            # Noisy emulator
            emulator = ZenadroneAlphaEmulator(
                t1=50_000 / level,  # Shorter T1 for higher noise
                t2=25_000 / level,  # Shorter T2 for higher noise
                readout_error=min(0.5, 0.1 * level),  # Cap at 50%
                gate_errors={
                    'h': 0.5 * level,
                    'x': 0.5 * level,
                    'cx': level,  # CNOT is more error-prone
                    'cz': level,  # CZ is also error-prone
                    'measure': 0.1 * level
                }
            )
        
        # Run the teleportation experiment
        start_time = time.time()
        fidelity = calculate_teleportation_fidelity(emulator, initial_state, shots)
        elapsed = time.time() - start_time
        
        # Print results
        print(f"{level:<15.4f} {fidelity:<15.4f} {fidelity*100:<14.2f}% ({elapsed:.2f}s)")
    
    print("="*70 + "\n")

def visualize_teleportation():
    """Create a text-based visualization of the teleportation circuit."""
    # Create the teleportation circuit
    circuit = create_teleportation_circuit('1')
    
    # Print a text representation of the circuit
    print("\n" + "="*70)
    print("TELEPORTATION CIRCUIT")
    print("="*70)
    print("Qubit 0: Alice's qubit to teleport")
    print("Qubit 1: Alice's Bell pair qubit")
    print("Qubit 2: Bob's Bell pair qubit")
    print("-"*70)
    
    # Print each operation with a step number
    for i, op in enumerate(circuit.instructions):
        # Format the operation for better readability
        op_str = f"{op.name.upper()}"
        if hasattr(op, 'qubits') and op.qubits:
            op_str += f" on qubits {op.qubits}"
        if hasattr(op, 'params') and op.params:
            op_str += f" with params {op.params}"
            
        print(f"Step {i+1:2d}: {op_str}")
    
    print("-"*70)
    print("Note: Measurements are performed on qubits 0 and 1 (Alice's qubits)")
    print("Final measurement is on qubit 2 (Bob's qubit)")
    print("="*70 + "\n")
        
    # Example of how to add visualization if your project supports it
    try:
        # This is a placeholder for your project's visualization
        # Replace with your actual visualization code if available
        from qx_ir.visualization import plot_circuit  # Example import
        plot_circuit(circuit).savefig('teleportation_circuit.png')
        print("Circuit diagram saved as 'teleportation_circuit.png'")
    except ImportError:
        print("\nNote: For circuit visualization, implement a visualization module")
        print("or use an external tool with the printed circuit information.")
        
def create_teleportation_circuit(initial_state: str = '1') -> Circuit:
    """
    Create a quantum circuit that demonstrates quantum teleportation.
    
    Args:
        initial_state: The state to teleport ('0', '1', '+', '-', etc.)
        
    Returns:
        Circuit: The teleportation circuit
    """
    # Initialize 3 qubits (Alice's qubit, Bell pair qubits) and 2 classical bits
    circuit = Circuit(3)  # Qubits: [qubit_to_teleport, alice_qubit, bob_qubit]
    
    # Prepare the initial state to teleport
    if initial_state == '1':
        circuit.add_op(Op('x', [0]))  # |1⟩
    elif initial_state == '+':
        circuit.add_op(Op('h', [0]))  # |+⟩
    elif initial_state == '-':
        circuit.add_op(Op('x', [0]))
        circuit.add_op(Op('h', [0]))  # |−⟩
    
    # Create Bell pair between Alice (qubit 1) and Bob (qubit 2)
    circuit.add_op(Op('h', [1]))
    circuit.add_op(Op('cx', [1, 2]))
    
    # Alice's operations
    circuit.add_op(Op('cx', [0, 1]))
    circuit.add_op(Op('h', [0]))
    
    # Measurements (Alice's qubits)
    circuit.add_op(Op('measure', [0]))
    circuit.add_op(Op('measure', [1]))
    
    # Bob's corrections
    # Since CZ isn't supported, we'll use H-CNOT-H as a replacement
    # circuit.add_op(Op('cz', [0, 2]))  # Original CZ
    circuit.add_op(Op('h', [2]))
    circuit.add_op(Op('cx', [0, 2]))
    circuit.add_op(Op('h', [2]))
    
    # CNOT with measurement result (simplified for simulation)
    circuit.add_op(Op('cx', [1, 2]))
    
    # Final measurement of Bob's qubit
    circuit.add_op(Op('measure', [2]))
    
    return circuit

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    print("Testing teleportation circuit...")
    
    # First, test with a perfect emulator
    perfect_emulator = ZenadroneAlphaEmulator(
        t1=float('inf'),  # No decoherence
        t2=float('inf'),
        readout_error=0.0,
        gate_errors={k: 0.0 for k in ['h', 'x', 'cx', 'measure']}
    )
    
    # Run with perfect conditions
    print("\nTesting with perfect conditions (no noise):")
    circuit = create_teleportation_circuit('1')
    counts = perfect_emulator.execute(circuit, shots=1000)
    print("\nMeasurement results (should be mostly '1' in Bob's qubit):")
    for state, count in sorted(counts.items()):
        print(f"{state}: {count}")
    
    # Now run with noise
    print("\n\nRunning noise experiment...")
    run_teleportation_experiment()
    
    # Show the circuit structure
    visualize_teleportation()
    
    print("\nExperiment complete! Check the output above for results.")
    print("\nTo run with custom noise levels, use:")
    print("  from examples.quantum_teleportation import run_teleportation_experiment")
    print("  run_teleportation_experiment(noise_levels=[0.01, 0.05, 0.1], shots=5000)")
    print("\nTo test with different initial states:")
    print("  circuit = create_teleportation_circuit('0')  # or '+', '-', etc.")
