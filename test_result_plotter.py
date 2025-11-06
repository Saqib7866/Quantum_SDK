#!/usr/bin/env python3
"""
Demo script for testing the quantum result plotter.
Shows different ways to visualize quantum measurement results.
"""
import random
import numpy as np
from qx_ir.visualization import CircuitDrawer

def generate_random_counts(n_qubits=3, shots=1000):
    """Generate random measurement counts for testing."""
    # Generate random probabilities for each basis state
    probs = np.random.random(2**n_qubits)
    probs = probs / np.sum(probs)  # Normalize
    
    # Generate counts based on probabilities
    states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    counts = {}
    remaining_shots = shots
    
    # Distribute shots according to probabilities
    for i, state in enumerate(states[:-1]):
        count = int(shots * probs[i])
        counts[state] = count
        remaining_shots -= count
    
    # Assign remaining shots to the last state
    counts[states[-1]] = remaining_shots
    
    return counts

def main():
    """Run the result plotter demo."""
    print("🎯 Quantum Result Plotter Demo\n")
    
    # Example 1: Simple 2-qubit results
    print("1️⃣  Simple 2-Qubit Results")
    simple_counts = {
        '00': 523,
        '01': 12,
        '10': 8,
        '11': 457
    }
    CircuitDrawer.plot_results(
        counts=simple_counts,
        title="2-Qubit Measurement Results",
        filename="2qubit_results.png"
    )
    
    # Example 2: Random 3-qubit results
    print("\n2️⃣  Random 3-Qubit Results")
    random_3q_counts = generate_random_counts(n_qubits=3, shots=1000)
    CircuitDrawer.plot_results(
        counts=random_3q_counts,
        title="Random 3-Qubit Measurements (1000 shots)",
        color='#2ca02c',  # Green
        filename="random_3q_results.png"
    )
    
    # Example 3: Integer input (automatically converted to binary)
    print("\n3️⃣  Integer Input (Auto-converted to Binary)")
    int_counts = {
        0: 500,  # Will be shown as '000'
        1: 100,  # '001'
        2: 50,   # '010'
        4: 25,   # '100'
        7: 325   # '111'
    }
    CircuitDrawer.plot_results(
        counts=int_counts,
        title="Integer Input (Shown as Binary)",
        color='#d62728',  # Red
        filename="integer_results.png"
    )
    
    # Example 4: List of measurements (automatically counted)
    print("\n4️⃣  List of Measurements (Auto-counted)")
    # Simulate 1000 measurements of a 4-qubit system
    measurements = [random.randint(0, 15) for _ in range(1000)]
    CircuitDrawer.plot_results(
        counts=measurements,
        title="4-Qubit Measurements (from list)",
        xlabel="Outcome (4-bit binary)",
        color='#9467bd',  # Purple
        figsize=(12, 6),
        filename="list_measurements.png"
    )
    
    print("\n✅ Demo complete!")
    print("💾 Plots have been saved as PNG files in the current directory.")

if __name__ == "__main__":
    main()
