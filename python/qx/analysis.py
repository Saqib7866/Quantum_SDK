"""
Circuit analysis and utility functions.

This module provides tools for analyzing quantum circuits, including depth calculation,
gate counting, and other circuit properties.
"""
from typing import Dict, List, Tuple, Set, Optional, TYPE_CHECKING
from collections import defaultdict
from .ir import Op

# Use string literals for type hints to avoid circular imports
if TYPE_CHECKING:
    from .circuit import Circuit

class CircuitAnalysis:
    """Class for analyzing quantum circuits."""
    
    def __init__(self, circuit: 'Circuit'):
        """Initialize with a quantum circuit.
        
        Args:
            circuit: The quantum circuit to analyze.
        """
        self.circuit = circuit
        self.qubit_ops = self._get_qubit_operations()
    
    def _get_qubit_operations(self) -> Dict[int, List[Tuple[int, Op]]]:
        """Get operations for each qubit with their positions.
        
        Returns:
            Dictionary mapping qubit indices to lists of (position, operation) tuples.
        """
        qubit_ops = defaultdict(list)
        for pos, op in enumerate(self.circuit.program.ops):
            for q in op.qubits:
                qubit_ops[q].append((pos, op))
        return dict(qubit_ops)
    
    def depth(self) -> int:
        """Calculate the circuit depth (longest path).
        
        Returns:
            The circuit depth as an integer.
        """
        if not self.qubit_ops:
            return 0
            
        # Find the last operation across all qubits
        last_positions = [ops[-1][0] for ops in self.qubit_ops.values() if ops]
        return max(last_positions, default=0) + 1
    
    def count_ops(self) -> Dict[str, int]:
        """Count the number of each type of operation in the circuit.
        
        Returns:
            Dictionary mapping operation names to counts.
        """
        counts = defaultdict(int)
        for op in self.circuit.program.ops:
            counts[op.name] += 1
        return dict(counts)
    
    def get_qubit_usage(self) -> Dict[int, int]:
        """Get the number of operations per qubit.
        
        Returns:
            Dictionary mapping qubit indices to operation counts.
        """
        return {q: len(ops) for q, ops in self.qubit_ops.items()}
    
    def get_critical_path(self) -> List[Op]:
        """Find the critical path through the circuit.
        
        Returns:
            List of operations in the critical path.
        """
        # This is a simplified implementation
        # A more complete version would consider gate dependencies
        max_qubit = max(self.qubit_ops.keys()) if self.qubit_ops else -1
        critical_path = []
        
        for q in range(max_qubit + 1):
            if q in self.qubit_ops and self.qubit_ops[q]:
                critical_path.extend(op for _, op in self.qubit_ops[q])
                
        return critical_path

def depth(circuit: 'Circuit') -> int:
    """Calculate the depth of a quantum circuit.
    
    Args:
        circuit: The quantum circuit.
        
    Returns:
        The circuit depth as an integer.
    """
    return CircuitAnalysis(circuit).depth()

def count_ops(circuit: 'Circuit') -> Dict[str, int]:
    """Count operations in a quantum circuit.
    
    Args:
        circuit: The quantum circuit.
        
    Returns:
        Dictionary mapping operation names to counts.
    """
    return CircuitAnalysis(circuit).count_ops()

def get_qubit_usage(circuit: 'Circuit') -> Dict[int, int]:
    """Get operation counts per qubit.
    
    Args:
        circuit: The quantum circuit.
        
    Returns:
        Dictionary mapping qubit indices to operation counts.
    """
    return CircuitAnalysis(circuit).get_qubit_usage()