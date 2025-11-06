"""
Resource estimator for quantum circuits.

This module provides functionality to estimate resource requirements and expected fidelity
for quantum circuits, including qubit counts, gate counts, and error estimates.
"""
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import math

from .core import Circuit, Op

@dataclass
class ResourceEstimate:
    """Container for resource estimation results."""
    # Basic resources
    num_qubits: int
    gate_counts: Dict[str, int]
    depth: int
    
    # Error metrics
    expected_error: float
    expected_fidelity: float
    
    # Memory usage (in bytes, if applicable)
    memory_estimate: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert the estimate to a dictionary."""
        return {
            'num_qubits': self.num_qubits,
            'gate_counts': self.gate_counts,
            'depth': self.depth,
            'expected_error': self.expected_error,
            'expected_fidelity': self.expected_fidelity,
            'memory_estimate': self.memory_estimate
        }
    
    def __str__(self) -> str:
        """String representation of the resource estimate."""
        lines = [
            "Resource Estimate:",
            "-----------------",
            f"Number of qubits: {self.num_qubits}",
            f"Circuit depth: {self.depth}",
            "\nGate counts:",
        ]
        
        # Group gates by type (single-qubit, two-qubit, etc.)
        gate_groups = defaultdict(list)
        for gate, count in sorted(self.gate_counts.items()):
            if count > 0:
                gate_groups[gate[0] if len(gate) == 1 else gate].append(f"{gate}: {count}")
        
        # Add gate counts to output
        for group in gate_groups.values():
            lines.append("  " + ", ".join(group))
        
        # Add error metrics
        lines.extend([
            "\nError Metrics:",
            f"Expected error rate: {self.expected_error:.2e}",
            f"Expected fidelity: {self.expected_fidelity:.2%}"
        ])
        
        if self.memory_estimate is not None:
            lines.append(f"\nMemory estimate: {self._format_bytes(self.memory_estimate)}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_bytes(size: int) -> str:
        """Format bytes in a human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"

class ResourceEstimator:
    """
    Estimates resource requirements and error rates for quantum circuits.
    
    This class provides methods to analyze quantum circuits and estimate their
    resource requirements including qubit counts, gate counts, circuit depth,
    and expected error rates.
    """
    
    # Default error rates (per operation)
    DEFAULT_ERROR_RATES = {
        # Single-qubit gates
        'h': 1e-4,
        'x': 1e-4,
        'y': 1e-4,
        'z': 1e-4,
        's': 1e-4,
        't': 1e-4,
        'rx': 1e-4,
        'ry': 1e-4,
        'rz': 1e-4,
        'u1': 1e-4,
        'u2': 1e-4,
        'u3': 1e-4,
        
        # Two-qubit gates
        'cx': 1e-3,
        'cz': 1e-3,
        'swap': 1e-3,
        'crx': 1e-3,
        'cry': 1e-3,
        'crz': 1e-3,
        'cu1': 1e-3,
        'cu2': 1e-3,
        'cu3': 1e-3,
        'cp': 1e-3,
        'ch': 1e-3,
        
        # Three+ qubit gates
        'ccx': 1e-2,  # Toffoli
        'cswap': 1e-2,  # Fredkin
        
        # Measurement and reset
        'measure': 1e-3,
        'reset': 1e-4,
    }
    
    def __init__(self, error_rates: Optional[Dict[str, float]] = None):
        """
        Initialize the resource estimator.
        
        Args:
            error_rates: Optional dictionary of error rates for different gate types.
                        If not provided, default error rates will be used.
        """
        self.error_rates = self.DEFAULT_ERROR_RATES.copy()
        if error_rates:
            self.error_rates.update(error_rates)
    
    def estimate(self, circuit: Circuit) -> ResourceEstimate:
        """
        Estimate resources and errors for a quantum circuit.
        
        Args:
            circuit: The quantum circuit to analyze.
            
        Returns:
            ResourceEstimate object containing the estimation results.
        """
        # Count gates and calculate depth
        gate_counts = self._count_gates(circuit)
        depth = self._calculate_depth(circuit)
        
        # Calculate error metrics
        total_error = self._calculate_total_error(gate_counts)
        fidelity = math.exp(-total_error)  # Simple exponential decay model
        
        # Estimate memory usage (very rough estimate)
        memory_estimate = self._estimate_memory(circuit.n_qubits, depth)
        
        return ResourceEstimate(
            num_qubits=circuit.n_qubits,
            gate_counts=gate_counts,
            depth=depth,
            expected_error=total_error,
            expected_fidelity=fidelity,
            memory_estimate=memory_estimate
        )
    
    def _count_gates(self, circuit: Circuit) -> Dict[str, int]:
        """Count the number of each type of gate in the circuit."""
        counts = defaultdict(int)
        for op in circuit.instructions:
            counts[op.name] += 1
        return dict(counts)
    
    def _calculate_depth(self, circuit: Circuit) -> int:
        """
        Calculate the depth of the circuit.
        
        This is a simplified version that doesn't account for parallel execution.
        A more sophisticated implementation would consider which gates can be
        executed in parallel.
        """
        # For simplicity, we'll just count the number of operations
        # A more accurate implementation would consider qubit dependencies
        return len(circuit.instructions)
    
    def _calculate_total_error(self, gate_counts: Dict[str, int]) -> float:
        """Calculate the total expected error based on gate counts and error rates."""
        total_error = 0.0
        for gate, count in gate_counts.items():
            error_rate = self.error_rates.get(gate, 1e-2)  # Default to 1% error for unknown gates
            total_error += count * error_rate
        return total_error
    
    def _estimate_memory(self, num_qubits: int, depth: int) -> int:
        """
        Estimate the memory required to simulate the circuit.
        
        This is a very rough estimate based on state vector simulation.
        """
        # For state vector simulation, we need to store 2^num_qubits complex numbers
        # Each complex number is typically 16 bytes (2 * 8-byte floats)
        state_vector_size = (2 ** num_qubits) * 16
        
        # For a more accurate simulation, we might need to store multiple states
        # or intermediate results. This is a simple estimate.
        return state_vector_size * 2  # Some overhead for operations
    
    @classmethod
    def get_gate_errors(cls) -> Dict[str, float]:
        """Get the default gate error rates."""
        return cls.DEFAULT_ERROR_RATES.copy()
