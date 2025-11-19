"""
Noise models for quantum circuit simulation.

This module provides classes and functions for modeling noise in quantum circuits.
"""
from typing import Dict, List, Optional, Union, Callable, Tuple
import numpy as np
from .errors import QXError

class QuantumError:
    """A quantum error channel."""
    
    def __init__(self, kraus_ops: List[np.ndarray], name: str = None):
        """Initialize a quantum error channel.
        
        Args:
            kraus_ops: List of Kraus operators defining the error channel.
            name: Optional name for the error channel.
        """
        self.kraus_ops = [op.astype(np.complex128) for op in kraus_ops]
        self.name = name or "quantum_error"
        self._validate_kraus_ops()
    
    def _validate_kraus_ops(self):
        """Validate that Kraus operators form a valid quantum channel."""
        if not self.kraus_ops:
            raise QXError("At least one Kraus operator is required.")
            
        d = self.kraus_ops[0].shape[0]
        if not all(op.shape == (d, d) for op in self.kraus_ops):
            raise QXError("All Kraus operators must have the same square shape.")
            
        # Check completeness relation
        sum_kdk = np.zeros((d, d), dtype=np.complex128)
        for k in self.kraus_ops:
            sum_kdk += k.conj().T @ k
        
        if not np.allclose(sum_kdk, np.eye(d), atol=1e-10):
            raise QXError("Kraus operators do not satisfy the completeness relation.")


class NoiseModel:
    """A noise model for quantum circuits."""
    
    def __init__(self):
        """Initialize an empty noise model."""
        self.noise_instructions = {}
        self.default_quantum_errors = {}
        self.x90_gates = []
        
    def add_all_qubit_quantum_error(self, error: QuantumError, instructions: List[str]):
        """Add a quantum error to all qubits for the given instructions.
        
        Args:
            error: The quantum error to add.
            instructions: List of instruction names to apply the error to.
        """
        for instr in instructions:
            if instr not in self.noise_instructions:
                self.noise_instructions[instr] = []
            self.noise_instructions[instr].append((None, error))
    
    def add_quantum_error(self, error: QuantumError, instructions: List[str], qubits: List[int]):
        """Add a quantum error to specific qubits for the given instructions.
        
        Args:
            error: The quantum error to add.
            instructions: List of instruction names to apply the error to.
            qubits: List of qubit indices to apply the error to.
        """
        for instr in instructions:
            if instr not in self.noise_instructions:
                self.noise_instructions[instr] = []
            self.noise_instructions[instr].extend([(q, error) for q in qubits])
    
    def add_readout_error(self, probabilities: np.ndarray, qubits: List[int]):
        """Add readout error to specific qubits.
        
        Args:
            probabilities: A 2x2 array where probabilities[i][j] is the probability
                          of measuring j when the true state is i.
            qubits: List of qubit indices to apply the readout error to.
        """
        if probabilities.shape != (2, 2):
            raise QXError("Readout error probabilities must be a 2x2 array.")
            
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
            raise QXError("Readout error probabilities must sum to 1 for each true state.")
            
        error = QuantumError("readout", {"probabilities": probabilities})
        self.add_quantum_error(error, ["measure"], qubits)
    
    def get_quantum_errors(self, instruction: str, qubit: int) -> List[QuantumError]:
        """Get all quantum errors for a given instruction and qubit.
        
        Args:
            instruction: The instruction name.
            qubit: The qubit index.
            
        Returns:
            List of quantum errors that apply to the given instruction and qubit.
        """
        errors = []
        if instruction in self.noise_instructions:
            for q, error in self.noise_instructions[instruction]:
                if q is None or q == qubit:
                    errors.append(error)
        return errors


def depolarizing_error(param: float, num_qubits: int = 1) -> QuantumError:
    """Return a depolarizing quantum error channel.
    
    Args:
        param: The depolarizing error parameter (0 <= p <= 1).
        num_qubits: The number of qubits (1 or 2).
        
    Returns:
        A QuantumError representing the depolarizing channel.
    """
    if not 0 <= param <= 1:
        raise QXError("Depolarizing parameter must be between 0 and 1.")
        
    if num_qubits == 1:
        # Single-qubit depolarizing channel
        p = 4 * param / 3
        if p > 1.0:
            p = 1.0
            
        # Kraus operators
        k0 = np.sqrt(1 - 3*p/4) * np.eye(2)
        k1 = np.sqrt(p/4) * np.array([[0, 1], [1, 0]], dtype=complex)  # X
        k2 = np.sqrt(p/4) * np.array([[0, -1j], [1j, 0]], dtype=complex)  # Y
        k3 = np.sqrt(p/4) * np.array([[1, 0], [0, -1]], dtype=complex)  # Z
        
        return QuantumError([k0, k1, k2, k3], f"depolarizing_{param}")
        
    elif num_qubits == 2:
        # Two-qubit depolarizing channel
        p = 16 * param / 15
        if p > 1.0:
            p = 1.0
            
        # Single-qubit Pauli matrices
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Two-qubit Pauli basis
        paulis = [I, X, Y, Z]
        basis = [np.kron(p1, p2) for p1 in paulis for p2 in paulis]
        
        # Kraus operators
        k0 = np.sqrt(1 - 15*p/16) * np.eye(4)
        kraus_ops = [k0] + [np.sqrt(p/16) * op for op in basis[1:]]
        
        return QuantumError(kraus_ops, f"depolarizing_2q_{param}")
    else:
        raise QXError("Only 1 and 2-qubit depolarizing channels are supported.")
