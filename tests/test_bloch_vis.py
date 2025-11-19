"""
Test script for Bloch sphere visualization in the Quantum SDK.

This script demonstrates how to use the Bloch sphere visualization
with different quantum gates.
"""
import sys
import os
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

from qx import Circuit, draw

def test_hadamard():
    """Test Hadamard gate on |0⟩ state."""
    print("Testing Hadamard gate (|+⟩ state):")
    circuit = Circuit()
    q = circuit.allocate(1)[0]  # Allocate one qubit
    circuit.h(q)  # Apply Hadamard gate
    return circuit

def test_phase_gate():
    """Test phase gate on |+⟩ state."""
    print("Testing Phase gate (|+i⟩ state):")
    circuit = Circuit()
    q = circuit.allocate(1)[0]
    circuit.h(q)  # |+⟩
    circuit.s(q)   # Apply S gate (phase gate)
    return circuit

def test_rotation_gates():
    """Test rotation gates."""
    print("Testing Rotation gates (custom rotation):")
    circuit = Circuit()
    q = circuit.allocate(1)[0]
    circuit.rx(q, np.pi/4)  # Rotate π/4 around X-axis
    circuit.ry(q, np.pi/3)  # Rotate π/3 around Y-axis
    return circuit

def main():
    """Run all test cases."""
    # Create test circuits
    circuits = [
        test_hadamard(),
        test_phase_gate(),
        test_rotation_gates()
    ]
    
    # Display each circuit
    for i, circuit in enumerate(circuits):
        print(f"\nCircuit {i+1}:")
        print("Text representation:")
        print(draw(circuit, output='text'))
        
        print("\nBloch sphere visualization (close the window to continue):")
        fig = draw(circuit, output='bloch')
        if hasattr(fig, 'show'):
            fig.show()
        else:
            print("Matplotlib figure could not be displayed. Do you have a display available?")
        
        if i < len(circuits) - 1:
            input("Press Enter to continue to the next test...")

if __name__ == "__main__":
    main()
