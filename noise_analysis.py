"""
Noise Analysis for Zenadrone Alpha Emulator

This script demonstrates the impact of different noise levels on quantum circuit fidelity.
It runs the same quantum circuit with varying noise parameters and plots the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from run_stage6_demo import (
    ZenadroneAlphaEmulator, 
    create_ghz_circuit,
    Op, Circuit, Program
)

def create_test_circuit(n_qubits: int = 3) -> Circuit:
    """Create a test GHZ circuit with measurements."""
    circuit = create_ghz_circuit(n_qubits)
    for i in range(n_qubits):
        circuit.add_op(Op('measure', [i]))
    return circuit

def calculate_fidelity(counts: Dict[str, int], expected_states: List[str]) -> float:
    """Calculate the fidelity of the measurement results."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    correct = sum(counts.get(state, 0) for state in expected_states)
    return correct / total_shots

def run_noise_experiment(
    noise_level: float,
    shots: int = 1000,
    n_qubits: int = 3
) -> Tuple[float, float]:
    """
    Run a single noise experiment with the given noise level.
    
    Args:
        noise_level: Multiplier for noise parameters (1.0 = default noise)
        shots: Number of shots to run
        n_qubits: Number of qubits in the GHZ state
        
    Returns:
        Tuple of (fidelity, execution_time)
    """
    # Scale noise parameters based on noise_level
    base_t1 = 100_000  # 100µs
    base_t2 = 50_000   # 50µs
    base_readout = 0.01  # 1%
    
    gate_errors = {
        'h': 0.5 * noise_level,
        'x': 0.5 * noise_level,
        'cx': 0.5 * noise_level,
        'rz': 0.5 * noise_level,
        'measure': 0.1 * noise_level
    }
    
    # Create emulator with scaled noise
    emulator = ZenadroneAlphaEmulator(
        t1=base_t1 / noise_level,
        t2=base_t2 / noise_level,
        readout_error=min(base_readout * noise_level, 0.5),  # Cap at 50%
        gate_errors=gate_errors
    )
    
    # Create and run circuit
    circuit = create_test_circuit(n_qubits)
    
    # Time the execution
    import time
    start_time = time.time()
    counts = emulator.execute(circuit, shots=shots)
    execution_time = time.time() - start_time
    
    # Calculate fidelity (for GHZ state, expected states are 000... and 111...)
    expected_state1 = '0' * n_qubits
    expected_state2 = '1' * n_qubits
    fidelity = calculate_fidelity(counts, [expected_state1, expected_state2])
    
    return fidelity, execution_time

def run_noise_sweep(
    min_noise: float = 0.1,
    max_noise: float = 10.0,
    num_points: int = 10,
    shots: int = 500,
    n_qubits: int = 3
) -> Dict[str, List[float]]:
    """
    Run multiple experiments with different noise levels.
    
    Args:
        min_noise: Minimum noise level (as a multiplier of base noise)
        max_noise: Maximum noise level
        num_points: Number of points to test
        shots: Number of shots per experiment
        n_qubits: Number of qubits in the GHZ state
        
    Returns:
        Dictionary with noise levels, fidelities, and execution times
    """
    noise_levels = np.linspace(min_noise, max_noise, num_points)
    fidelities = []
    execution_times = []
    
    print(f"Running noise sweep from {min_noise:.1f}x to {max_noise:.1f}x base noise...")
    print("Noise Level | Fidelity | Time (s)")
    print("-" * 30)
    
    for noise in noise_levels:
        fidelity, exec_time = run_noise_experiment(noise, shots, n_qubits)
        fidelities.append(fidelity)
        execution_times.append(exec_time)
        print(f"{noise:10.2f}x | {fidelity:8.4f} | {exec_time:8.4f}")
    
    return {
        'noise_levels': noise_levels,
        'fidelities': fidelities,
        'execution_times': execution_times
    }

def plot_results(results: Dict[str, List[float]]):
    """Plot the results of the noise sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot fidelity vs noise level
    ax1.plot(results['noise_levels'], results['fidelities'], 'b-o')
    ax1.set_xlabel('Noise Level (relative to base)')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Circuit Fidelity vs Noise Level')
    ax1.grid(True, alpha=0.3)
    
    # Plot execution time vs noise level
    ax2.plot(results['noise_levels'], results['execution_times'], 'r-o')
    ax2.set_xlabel('Noise Level (relative to base)')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Execution Time vs Noise Level')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Run the noise sweep
    results = run_noise_sweep(
        min_noise=0.1,    # 0.1x base noise
        max_noise=10.0,   # 10x base noise
        num_points=8,     # Number of points to test
        shots=500,        # Shots per experiment
        n_qubits=3        # Number of qubits
    )
    
    # Plot the results
    plot_results(results)
    
    # Print a summary
    print("\nNoise Analysis Summary:")
    print("-" * 30)
    print(f"Minimum Fidelity: {min(results['fidelities']):.4f} at {results['noise_levels'][np.argmin(results['fidelities'])]:.2f}x noise")
    print(f"Maximum Fidelity: {max(results['fidelities']):.4f} at {results['noise_levels'][np.argmax(results['fidelities'])]:.2f}x noise")
    print(f"Average Execution Time: {np.mean(results['execution_times']):.4f} seconds")

if __name__ == "__main__":
    main()
