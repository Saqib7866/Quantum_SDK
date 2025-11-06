# Core Module

The `qx_ir.core` module contains the fundamental building blocks of quantum circuits in QX-IR.

## Circuit

```python
class Circuit:
    """A quantum circuit containing quantum operations.
    
    Args:
        n_qubits: Number of qubits in the circuit
        name: Optional name for the circuit
    """
    
    def add_op(self, op: 'Op') -> None:
        """Add an operation to the circuit.
        
        Args:
            op: The operation to add
        """
        pass
    
    def depth(self) -> int:
        """Calculate the depth of the circuit."""
        pass
    
    def __str__(self) -> str:
        """String representation of the circuit."""
        pass
```

## Operation (Op)

```python
class Op:
    """A quantum operation or gate.
    
    Args:
        name: Name of the operation (e.g., 'h', 'x', 'cx')
        qubits: List of qubit indices the operation acts on
        params: Optional parameters for parameterized gates
        **kwargs: Additional operation-specific arguments
    """
    
    def __init__(self, name: str, qubits: List[int], 
                 params: Optional[List[float]] = None, **kwargs):
        pass
    
    def __str__(self) -> str:
        """String representation of the operation."""
        pass
```

## Program

```python
class Program:
    """A quantum program containing one or more circuits.
    
    Args:
        circuits: List of quantum circuits
        config: Configuration dictionary
    """
    
    def __init__(self, circuits: List[Circuit], 
                 config: Optional[Dict[str, Any]] = None):
        pass
    
    def add_circuit(self, circuit: Circuit) -> None:
        """Add a circuit to the program."""
        pass
    
    def run(self, backend=None, **kwargs):
        """Run the program on a backend."""
        pass
```

## Example Usage

```python
from qx_ir import Circuit, Op, Program

# Create a circuit with 2 qubits
circuit = Circuit(n_qubits=2)

# Add operations
circuit.add_op(Op('h', qubits=[0]))
circuit.add_op(Op('cx', qubits=[0, 1]))

# Create a program with the circuit
program = Program(circuits=[circuit], config={'shots': 1000})

# Run the program (requires a backend)
# result = program.run()
```

## Related

- [Simulator API](simulator.md) - For simulating quantum circuits
- [Visualization API](visualization.md) - For visualizing circuits
- [Resource Estimation API](estimator.md) - For analyzing circuit resources
