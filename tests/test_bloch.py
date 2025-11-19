import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

from python.qx import Circuit, draw

def test_bloch_visualization():
    # Test 1: Basic |+> state
    print("Test 1: Hadamard gate (|+> state)")
    circuit = Circuit()
    q = circuit.allocate(1)[0]  # Allocate one qubit
    circuit.h(q)
    fig = draw(circuit, output='bloch')
    assert fig is not None

    # Test 2: |+i> state
    print("\nTest 2: |+i> state (H then S)")
    circuit = Circuit()
    q = circuit.allocate(1)[0]  # Allocate one qubit
    circuit.h(q)
    circuit.s(q)
    fig = draw(circuit, output='bloch')
    assert fig is not None

    # Test 3: Custom rotation
    print("\nTest 3: Custom rotation (H, RX, RY)")
    circuit = Circuit()
    q = circuit.allocate(1)[0]  # Allocate one qubit
    circuit.h(q)
    circuit.rx(q, np.pi/4)
    circuit.ry(q, np.pi/3)
    fig = draw(circuit, output='bloch')
    assert fig is not None

if __name__ == "__main__":
    test_bloch_visualization()
