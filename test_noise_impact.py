"""
Test the impact of noise on quantum circuit execution.

This script demonstrates how increasing noise levels affect the fidelity of quantum operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from run_stage6_demo import ZenadroneAlphaEmulator, create_ghz_circuit, Op, Circuit

def run_experiment(noise_level: float, shots: int = 1000) -> float:
    """
    Run a single experiment with a given noise level.
    
    Args:
        noise_level: Multiplier for noise parameters (1.0 = default noise)
        shots: Number of shots to run
        
    Returns:
        Fidelity of the results (fraction of correct measurements)
    """
    # Create emulator with scaled noise
    emulator = ZenadroneAlphaEmulator(
        t1=100_000 / noise_level,  # Shorter T1 = more noise
        t2=50_000 / noise_level,   # Shorter T2 = more noise
        readout_error=min(0.5, 0.01 * noise_level),  # Cap at 50%
        gate_errors={
            'h': 0.5 * noise_level,
            'x': 0.5 * noise_level,
            'cx': 0.5 * noise_level,
            'rz': 0.5 * noise_level,
            'measure': 0.1 * noise_level
        }
    )
    
    # Create a 3-qubit GHZ state
    circuit = create_ghz_circuit(3)
    for i in range(3):
        circuit.add_op(Op('measure', [i]))
    
    # Run the circuit
    try:
        counts = emulator.execute(circuit, shots=shots)
        
        # Calculate fidelity (should be either '000' or '111' for perfect GHZ)
        correct = counts.get('000', 0) + counts.get('111', 0)
        fidelity = correct / shots
        
        print(f"Noise level: {noise_level:.2f}x - Fidelity: {fidelity:.4f}")
        return fidelity
    except Exception as e:
        print(f"Error with noise level {noise_level:.2f}x: {str(e)}")
        return 0.0

def main():
    # Test different noise levels
    noise_levels = np.linspace(0.1, 10.0, 10)  # From 0.1x to 10x noise
    fidelities = []
    
    print("Testing noise impact on GHZ state fidelity...")
    print("Noise Level | Fidelity")
    print("-" * 30)
    
    for level in noise_levels:
        fidelity = run_experiment(level, shots=1000)
        fidelities.append(fidelity)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, fidelities, 'o-', label='Fidelity')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
    plt.xlabel('Noise Level (relative to base)')
    plt.ylabel('Fidelity')
    plt.title('Quantum Circuit Fidelity vs. Noise Level')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig('noise_impact.png')
    print("\nPlot saved as 'noise_impact.png'")
    
    # Print summary
    print("\nNoise Impact Summary:")
    print("-" * 30)
    for level, fid in zip(noise_levels, fidelities):
        print(f"{level:5.2f}x  |  {fid:.4f}")

if __name__ == "__main__":
    main()
