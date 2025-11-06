from typing import List, Optional, Dict, Any

class Op:
    """
    Represents a single quantum operation.

    Attributes:
        name (str): Operation or gate name.
        qubits (List[int]): Target qubit indices.
        params (List[float]): Optional numeric parameters.

    Methods:
        __repr__(): Returns a short string describing the operation.

    Example:
        >>> op = Op("rz", [0], [3.1415])
        >>> print(op)
        Op(name='rz', qubits=[0], params=[3.1415])
    """

    def __init__(self, name: str, qubits: List[int], params: Optional[List[float]] = None):
        self.name = name
        self.qubits = qubits
        self.params = params if params is not None else []

    def __repr__(self) -> str:
        return f"Op(name='{self.name}', qubits={self.qubits}, params={self.params})"

class Circuit:
    """
    Represents a quantum circuit as a list of operations on qubits.

    Attributes:
        n_qubits (int): Number of qubits in the circuit.
        instructions (List[Op]): Ordered quantum operations.

    Methods:
        add_op(op): Add an operation, validating qubit indices.
        __repr__(): Return a short summary of the circuit.

    Example:
        >>> circuit = Circuit(3)
        >>> circuit.add_op(Op("x", [0]))
        >>> circuit.add_op(Op("cx", [0, 1]))
        >>> print(circuit)
        Circuit(n_qubits=3, instructions=2 ops)
    """
    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")
        self.n_qubits = n_qubits
        self.instructions: List[Op] = []

    def add_op(self, op: Op):
        """Add a quantum operation(apply gates to changes the state) after validating qubit indices."""
        for qubit_idx in op.qubits:
            if qubit_idx >= self.n_qubits:
                raise ValueError(f"Qubit index {qubit_idx} is out of bounds for a {self.n_qubits}-qubit circuit.")
        self.instructions.append(op)

    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self.n_qubits}, instructions={len(self.instructions)} ops)"

class Program:
    """
    Represents a complete QX-IR program combining multiple circuits and configuration data.

    Attributes:
        circuits (List[Circuit]): List of Circuit objects included in the program.
        config (Dict[str, Any]): Optional configuration parameters (e.g., backend, runtime, or metadata).

    Methods:
        __init__(circuits, config=None): Initializes the program with given circuits and configuration.
        __repr__(): Returns a summary string showing the number of circuits and available config keys.
    """
    def __init__(self, circuits: List[Circuit], config: Optional[Dict[str, Any]] = None):
        self.circuits = circuits
        self.config = config if config is not None else {}

    def __repr__(self) -> str:
        return f"Program(circuits={len(self.circuits)}, config_keys={list(self.config.keys())})"
