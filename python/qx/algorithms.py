"""
Quantum algorithms and utilities.

This module provides implementations of common quantum algorithms and utilities.
"""
from typing import List, Callable, Dict, Any, Optional, Union
import numpy as np
from .circuit import Circuit
from .noise import NoiseModel

class QuantumAlgorithms:
    """Collection of quantum algorithms."""
    
    @staticmethod
    def create_bell_pair() -> Circuit:
        """Create a Bell pair (maximally entangled state).
        
        Returns:
            A circuit that creates a Bell pair on qubits 0 and 1.
        """
        qc = Circuit()
        q = qc.allocate(2)
        qc.h(q[0])
        qc.cx(q[0], q[1])
        return qc
    
    @staticmethod
    def qft(circuit: Circuit, qubits: List[int], inverse: bool = False):
        """Apply Quantum Fourier Transform (QFT) to a set of qubits.
        
        Args:
            circuit: The quantum circuit.
            qubits: List of qubit indices to apply QFT to.
            inverse: If True, apply inverse QFT.
        """
        n = len(qubits)
        
        if not inverse:
            # Forward QFT
            for j in range(n):
                circuit.h(qubits[j])
                for k in range(j + 1, n):
                    angle = np.pi / (2 ** (k - j))
                    circuit.cp(angle, qubits[k], qubits[j])
            
            # Swap qubits
            for i in range(n // 2):
                circuit.swap(qubits[i], qubits[n - 1 - i])
        else:
            # Inverse QFT
            for i in range(n // 2):
                circuit.swap(qubits[i], qubits[n - 1 - i])
            
            for j in reversed(range(n)):
                for k in reversed(range(j + 1, n)):
                    angle = -np.pi / (2 ** (k - j))
                    circuit.cp(angle, qubits[k], qubits[j])
                circuit.h(qubits[j])
    
    @staticmethod
    def grover_search(oracle: Callable[[Circuit, List[int]], None], 
                      n_qubits: int, 
                      iterations: Optional[int] = None) -> Circuit:
        """Grover's search algorithm.
        
        Args:
            oracle: A function that marks the solution state.
            n_qubits: Number of qubits in the search space.
            iterations: Number of Grover iterations. If None, uses optimal number.
            
        Returns:
            A circuit implementing Grover's algorithm.
        """
        if iterations is None:
            # Optimal number of iterations is approximately π/4 * sqrt(2^n)
            iterations = int(np.pi/4 * np.sqrt(2 ** n_qubits))
        
        qc = Circuit()
        qubits = qc.allocate(n_qubits)
        target = qc.allocate(1)[0]  # For phase kickback
        
        # Initialize uniform superposition
        for q in qubits:
            qc.h(q)
        
        # Initialize target qubit in |->
        qc.x(target)
        qc.h(target)
        
        # Grover iterations
        for _ in range(iterations):
            # Apply oracle
            oracle(qc, qubits)
            
            # Apply diffusion operator
            for q in qubits:
                qc.h(q)
            for q in qubits:
                qc.x(q)
            
            # Multi-controlled Z
            qc.h(qubits[-1])
            qc.mct(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
            
            for q in qubits:
                qc.x(q)
            for q in qubits:
                qc.h(q)
        
        return qc
    
    @staticmethod
    def quantum_phase_estimation(unitary: Callable[[Circuit, int, int], None], 
                                precision: int,
                                target_qubit: int,
                                control_qubits: List[int]) -> Circuit:
        """Quantum Phase Estimation algorithm.
        
        Args:
            unitary: Function that applies the unitary operator to be estimated.
            precision: Number of bits of precision for the phase estimate.
            target_qubit: Qubit that the unitary operates on.
            control_qubits: Qubits to use for the phase estimation.
            
        Returns:
            A circuit implementing QPE.
        """
        qc = Circuit()
        
        # Allocate qubits: precision qubits + 1 target qubit
        qc.allocate(len(control_qubits) + 1)
        
        # Initialize target qubit in |1>
        qc.x(target_qubit)
        
        # Apply Hadamard to control qubits
        for q in control_qubits:
            qc.h(q)
        
        # Apply controlled unitaries
        for i, q in enumerate(control_qubits):
            for _ in range(2 ** i):
                unitary(qc, q, target_qubit)
        
        # Apply inverse QFT
        QuantumAlgorithms.qft(qc, control_qubits, inverse=True)
        
        return qc

# Alias for convenience
algorithms = QuantumAlgorithms()
