"""
Quantum Circuit Module

This module provides a high-level interface for constructing and manipulating quantum circuits.
It supports common quantum gates, qubit allocation, and measurement operations, enabling the
creation of complex quantum algorithms and experiments.

The Circuit class is the main interface for building quantum programs. It provides methods to:
- Allocate and manage quantum and classical bits
- Apply quantum gates and operations
- Perform measurements
- Compose and manipulate circuits

Example:
    Basic Bell State Creation:
    >>> from qx import Circuit, backend, run
    >>> 
    >>> # Create a Bell state circuit
    >>> with Circuit() as qc:
    ...     # Allocate 2 qubits (returns qubit indices)
    ...     a, b = qc.allocate(2)
    ...     # Apply quantum gates
    ...     qc.h(a)        # Hadamard gate on qubit a
    ...     qc.cx(a, b)    # CNOT with control a and target b
    ...     # Measure both qubits
    ...     m1, m2 = qc.measure(a, b)
    ... 
    >>> # Execute the circuit
    >>> job = run(qc, backend("sim-local"), shots=1000)
    >>> result = job.result()
    >>> print("Measurement counts:", result.counts)
    >>> print("Result metadata:", result.metadata)

Advanced Usage:
    The Circuit class supports more advanced features like:
    - Parameterized gates (rx, ry, rz with angle parameters)
    - Qubit measurement and classical control
    - Circuit composition and concatenation
    - Circuit visualization (when visualization tools are available)
"""

from typing import Tuple, Union, List, Optional, Dict, Any, Sequence, overload, Callable
from dataclasses import dataclass
from copy import deepcopy
from .ir import Program, Op
from .errors import (
    CircuitError,
    QubitOutOfRangeError,
    MeasurementError,
    GateError
)
from .analysis import CircuitAnalysis
from .visualization import draw

class Circuit:
    """
    A quantum circuit that supports various quantum operations.
    
    The Circuit class provides methods to build quantum circuits by applying
    quantum gates, allocating qubits, and performing measurements. It uses
    an intermediate representation (IR) to store the quantum program.
    
    Attributes:
        _prog (Program): The internal representation of the quantum program.
    """
    
    def __init__(self, n_qubits_or_name=None, name=None):
        """Initialize a new quantum circuit.

        Args:
            n_qubits_or_name: Number of qubits to allocate or name for the circuit.
            name: Optional name for the circuit (if first arg is n_qubits).
        """
        if isinstance(n_qubits_or_name, int):
            n_qubits = n_qubits_or_name
            self._prog = Program(n_qubits=n_qubits, ops=[], n_clbits=0, metadata={})
            self.name = name or f"circuit_{id(self)}"
        else:
            self._prog = Program(n_qubits=0, ops=[], n_clbits=0, metadata={})
            self.name = n_qubits_or_name or name or f"circuit_{id(self)}"
        self._analysis = None

    def allocate(self, n: int) -> Tuple[int, ...]:
        """
        Allocate one or more qubits in the circuit.
        
        Args:
            n: Number of qubits to allocate.
            
        Returns:
            A tuple of qubit indices (start, start+1, ..., start+n-1).
            
        Raises:
            ValueError: If n is not a positive integer.
            
        Example:
            >>> qc = Circuit()
            >>> q1, q2 = qc.allocate(2)  # Allocates qubits 0 and 1
            >>> q3, = qc.allocate(1)     # Allocates qubit 2
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"Number of qubits must be a positive integer, got {n}")
            
        start = self._prog.n_qubits
        self._prog.n_qubits += n
        return tuple(range(start, start + n))

    def _validate_qubit(self, q: int) -> None:
        """Validate that a qubit index is within bounds.

        If the qubit index is beyond the current number of qubits, automatically allocate more qubits.

        Args:
            q: Qubit index to validate.

        Raises:
            QubitOutOfRangeError: If the qubit index is negative.
        """
        if q < 0:
            raise QubitOutOfRangeError(
                f"Qubit index {q} out of range [0, {self._prog.n_qubits - 1}]"
            )
        if q >= self._prog.n_qubits:
            self.allocate(q - self._prog.n_qubits + 1)

    def _validate_qubits(self, *qubits: int) -> None:
        """Validate multiple qubit indices.
        
        Args:
            *qubits: Qubit indices to validate.
            
        Raises:
            QubitOutOfRangeError: If any qubit index is out of range.
        """
        for q in qubits:
            self._validate_qubit(q)

    def _validate_angle(self, angle: float, name: str = "angle") -> None:
        """Validate that an angle is a valid floating-point number.
        
        Args:
            angle: The angle to validate.
            name: Name of the angle parameter for error messages.
            
        Raises:
            ValueError: If the angle is not a valid number.
        """
        if not isinstance(angle, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(angle).__name__}")

    def _normalize_qubit(self, q: Union[int, Tuple[int, ...], List[int]]) -> int:
        """Normalize qubit input to a single integer index.
        
        Args:
            q: Qubit index or single-element sequence.
            
        Returns:
            Normalized qubit index as an integer.
            
        Raises:
            ValueError: If the input is not a valid qubit identifier.
        """
        if isinstance(q, (tuple, list)):
            if len(q) != 1:
                raise ValueError(
                    f"Expected single qubit identifier, got sequence of length {len(q)}"
                )
            return q[0]
        if not isinstance(q, int):
            raise ValueError(
                f"Qubit identifier must be an integer or single-element sequence, got {type(q).__name__}"
            )
        return q

    # 1-qubit gates
    def h(self, q: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply a Hadamard gate to a qubit.
        
        Args:
            q: Qubit index to apply the gate to.
            
        Example:
            >>> qc = Circuit()
            >>> q = qc.allocate(1)[0]
            >>> qc.h(q)  # Apply Hadamard to qubit 0
        """
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("h", (q,), condition=condition))

    def x(self, q: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply a Pauli-X (NOT) gate to a qubit."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("x", (q,), condition=condition))

    def sx(self, q: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply a square-root of X gate (√X)."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("sx", (q,), condition=condition))

    def sxdg(self, q: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply the adjoint square-root of X gate (√X†)."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("sxdg", (q,), condition=condition))

    def y(self, q: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply a Pauli-Y gate to a qubit."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("y", (q,), condition=condition))

    def z(self, q: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply a Pauli-Z gate to a qubit."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("z", (q,), condition=condition))

    def rx(self, q: Union[int, Tuple[int, ...], List[int]], theta: float, condition: Optional[int] = None) -> None:
        """Apply a rotation around the X-axis by the given angle.
        
        Args:
            q: Qubit to apply the rotation to.
            theta: Rotation angle in radians.
            
        Example:
            >>> qc = Circuit()
            >>> q = qc.allocate(1)[0]
            >>> qc.rx(q, 3.14159)  # Rotate π radians around X-axis
        """
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._validate_angle(theta, "theta")
        self._prog.add(Op("rx", (q,), (theta,), condition=condition))

    def ry(self, q: Union[int, Tuple[int, ...], List[int]], theta: float, condition: Optional[int] = None) -> None:
        """Apply a rotation around the Y-axis by the given angle."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._validate_angle(theta, "theta")
        self._prog.add(Op("ry", (q,), (theta,), condition=condition))

    def rz(self, q: Union[int, Tuple[int, ...], List[int]], theta: float, condition: Optional[int] = None) -> None:
        """Apply a rotation around the Z-axis by the given angle."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._validate_angle(theta, "theta")
        self._prog.add(Op("rz", (q,), (theta,), condition=condition))

    # 2-qubit
    def cx(self, control: Union[int, Tuple[int, ...], List[int]], 
            target: Union[int, Tuple[int, ...], List[int]], condition: Optional[int] = None) -> None:
        """Apply a controlled-X (CNOT) gate.
        
        The CNOT gate flips the target qubit if the control qubit is |1⟩.
        
        Args:
            control: Control qubit index.
            target: Target qubit index.
            
        Raises:
            QubitOutOfRangeError: If either qubit index is out of range.
            ValueError: If control and target qubits are the same.
            
        Example:
            >>> qc = Circuit()
            >>> c, t = qc.allocate(2)
            >>> qc.h(c)     # Put control qubit in superposition
            >>> qc.cx(c, t) # Create Bell pair
        """
        c = self._normalize_qubit(control)
        t = self._normalize_qubit(target)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError(f"Control and target qubits must be different, got {c}")
        self._prog.add(Op("cx", (c, t), condition=condition))

    def ccx(self, c1: int, c2: int, t: int, condition: Optional[int] = None) -> None:
        """Apply a Toffoli (CCX) gate."""
        c1 = self._normalize_qubit(c1)
        c2 = self._normalize_qubit(c2)
        t = self._normalize_qubit(t)
        self._validate_qubits(c1, c2, t)
        if len({c1, c2, t}) != 3:
            raise ValueError("Control and target qubits must be unique")
        self._prog.add(Op("ccx", (c1, c2, t), condition=condition))

    def ccz(self, c1: int, c2: int, t: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-controlled-Z gate."""
        c1 = self._normalize_qubit(c1)
        c2 = self._normalize_qubit(c2)
        t = self._normalize_qubit(t)
        self._validate_qubits(c1, c2, t)
        if len({c1, c2, t}) != 3:
            raise ValueError("All qubits must be unique")
        self._prog.add(Op("ccz", (c1, c2, t), condition=condition))

    def swap(self, q1: int, q2: int, condition: Optional[int] = None) -> None:
        """Apply a SWAP gate."""
        q1 = self._normalize_qubit(q1)
        q2 = self._normalize_qubit(q2)
        self._validate_qubits(q1, q2)
        if q1 == q2:
            raise ValueError("SWAP qubits must be different")
        self._prog.add(Op("swap", (q1, q2), condition=condition))

    def iswap(self, q1: int, q2: int, condition: Optional[int] = None) -> None:
        """Apply an iSWAP gate."""
        q1 = self._normalize_qubit(q1)
        q2 = self._normalize_qubit(q2)
        self._validate_qubits(q1, q2)
        if q1 == q2:
            raise ValueError("iSWAP qubits must be different")
        self._prog.add(Op("iswap", (q1, q2), condition=condition))

    def cu1(self, lam: float, c: int, t: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-U1 gate."""
        self._validate_angle(lam, "lambda")
        c = self._normalize_qubit(c)
        t = self._normalize_qubit(t)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("cu1", (c, t), (lam,), condition=condition))

    def cp(self, lam: float, control: int, target: int, condition: Optional[int] = None) -> None:
        """Apply a controlled phase gate (alias of CU1)."""
        self.cu1(lam, control, target, condition=condition)

    def cz(self, control: int, target: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-Z gate."""
        c = self._normalize_qubit(control)
        t = self._normalize_qubit(target)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("cz", (c, t), condition=condition))

    def cy(self, control: int, target: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-Y gate."""
        c = self._normalize_qubit(control)
        t = self._normalize_qubit(target)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("cy", (c, t), condition=condition))

    def csx(self, control: int, target: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-√X gate."""
        c = self._normalize_qubit(control)
        t = self._normalize_qubit(target)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("csx", (c, t), condition=condition))
        
    # Phase gates
    def s(self, q: int, condition: Optional[int] = None) -> None:
        """Apply an S gate (√Z)."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("s", (q,), condition=condition))
        
    def sdg(self, q: int, condition: Optional[int] = None) -> None:
        """Apply an S† gate (inverse of S)."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("sdg", (q,), condition=condition))
        
    def t(self, q: int, condition: Optional[int] = None) -> None:
        """Apply a T gate (4th root of Z)."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("t", (q,), condition=condition))
        
    def tdg(self, q: int, condition: Optional[int] = None) -> None:
        """Apply a T† gate (inverse of T)."""
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("tdg", (q,), condition=condition))
        
    # Controlled rotation gates
    def crx(self, theta: float, c: int, t: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-RX gate."""
        self._validate_angle(theta, "theta")
        c = self._normalize_qubit(c)
        t = self._normalize_qubit(t)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("crx", (c, t), (theta,), condition=condition))
        
    def cry(self, theta: float, c: int, t: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-RY gate."""
        self._validate_angle(theta, "theta")
        c = self._normalize_qubit(c)
        t = self._normalize_qubit(t)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("cry", (c, t), (theta,), condition=condition))
        
    def crz(self, theta: float, c: int, t: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-RZ gate."""
        self._validate_angle(theta, "theta")
        c = self._normalize_qubit(c)
        t = self._normalize_qubit(t)
        self._validate_qubits(c, t)
        if c == t:
            raise ValueError("Control and target qubits must be different")
        self._prog.add(Op("crz", (c, t), (theta,), condition=condition))
        
    # Multi-qubit gates
    def cswap(self, c: int, a: int, b: int, condition: Optional[int] = None) -> None:
        """Apply a controlled-SWAP (Fredkin) gate."""
        c = self._normalize_qubit(c)
        a = self._normalize_qubit(a)
        b = self._normalize_qubit(b)
        self._validate_qubits(c, a, b)
        if len({c, a, b}) != 3:
            raise ValueError("All qubits must be unique")
        self._prog.add(Op("cswap", (c, a, b), condition=condition))
        
    def rxx(self, theta: float, q1: int, q2: int, condition: Optional[int] = None) -> None:
        """Apply an Ising XX coupling gate."""
        self._validate_angle(theta, "theta")
        q1 = self._normalize_qubit(q1)
        q2 = self._normalize_qubit(q2)
        self._validate_qubits(q1, q2)
        if q1 == q2:
            raise ValueError("Qubits must be different")
        self._prog.add(Op("rxx", (q1, q2), (theta,), condition=condition))
        
    def ryy(self, theta: float, q1: int, q2: int, condition: Optional[int] = None) -> None:
        """Apply an Ising YY coupling gate."""
        self._validate_angle(theta, "theta")
        q1 = self._normalize_qubit(q1)
        q2 = self._normalize_qubit(q2)
        self._validate_qubits(q1, q2)
        if q1 == q2:
            raise ValueError("Qubits must be different")
        self._prog.add(Op("ryy", (q1, q2), (theta,), condition=condition))
        
    def rzz(self, theta: float, q1: int, q2: int, condition: Optional[int] = None) -> None:
        """Apply an Ising ZZ coupling gate."""
        self._validate_angle(theta, "theta")
        q1 = self._normalize_qubit(q1)
        q2 = self._normalize_qubit(q2)
        self._validate_qubits(q1, q2)
        if q1 == q2:
            raise ValueError("Qubits must be different")
        self._prog.add(Op("rzz", (q1, q2), (theta,), condition=condition))
        
    # General unitary gates
    def u1(self, lam: float, q: int, condition: Optional[int] = None) -> None:
        """Apply a U1 gate (phase shift)."""
        self._validate_angle(lam, "lambda")
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("u1", (q,), (lam,), condition=condition))

    def p(self, lam: float, q: int, condition: Optional[int] = None) -> None:
        """Apply a phase shift gate (alias of U1)."""
        self.u1(lam, q, condition=condition)
        
    def u2(self, phi: float, lam: float, q: int, condition: Optional[int] = None) -> None:
        """Apply a U2 gate."""
        self._validate_angle(phi, "phi")
        self._validate_angle(lam, "lambda")
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("u2", (q,), (phi, lam), condition=condition))
        
    def u3(self, theta: float, phi: float, lam: float, q: int, condition: Optional[int] = None) -> None:
        """Apply a U3 gate (general single-qubit unitary)."""
        self._validate_angle(theta, "theta")
        self._validate_angle(phi, "phi")
        self._validate_angle(lam, "lambda")
        q = self._normalize_qubit(q)
        self._validate_qubit(q)
        self._prog.add(Op("u3", (q,), (theta, phi, lam), condition=condition))

    # measure
    def measure(self, *qs: int):
        # accept ints or single-element tuples/lists and flatten
        flat = []
        for q in qs:
            if isinstance(q, (tuple, list)):
                flat.extend(list(q))
            else:
                flat.append(q)
        start = self._prog.n_clbits
        self._prog.n_clbits += len(flat)
        for i, q in enumerate(flat):
            self._prog.add(Op("measure", (q,), (), (start + i,)))
        return tuple(range(start, start + len(flat)))

    @property
    def program(self) -> Program:
        return self._prog
        
    @property
    def analysis(self) -> 'CircuitAnalysis':
        """Get the circuit analysis object."""
        if self._analysis is None:
            self._analysis = CircuitAnalysis(self)
        return self._analysis
        
    @property
    def qubits(self):
        """Return a tuple of allocated qubit indices."""
        return tuple(range(self._prog.n_qubits))

    def depth(self) -> int:
        """Return the circuit depth."""
        return self.analysis.depth()
        
    def count_ops(self) -> Dict[str, int]:
        """Count operations in the circuit."""
        return self.analysis.count_ops()
        
    def draw(self, output: str = 'text', **kwargs):
        """Draw the circuit.
        
        Args:
            output: Output format ('text', 'latex', 'mpl').
            **kwargs: Additional drawing options.
            
        Returns:
            The circuit drawing in the specified format.
        """
        return draw(self, output, **kwargs)
        
    def compose(self, other: 'Circuit', qubits: Optional[List[int]] = None) -> 'Circuit':
        """Compose with another circuit.
        
        Args:
            other: The other circuit to compose with.
            qubits: Qubit mapping. If None, qubits are mapped 1:1.
            
        Returns:
            A new circuit with the composition.
        """
        if qubits is not None and len(qubits) != other.program.n_qubits:
            raise ValueError("Number of qubits in mapping must match the other circuit")
            
        # Create a copy of this circuit
        new_circuit = deepcopy(self)
        
        # Add qubits if needed
        if qubits is None:
            # Simple 1:1 mapping
            if self.program.n_qubits < other.program.n_qubits:
                new_circuit.allocate(other.program.n_qubits - self.program.n_qubits)
            qubit_map = list(range(other.program.n_qubits))
        else:
            # Use provided mapping
            qubit_map = qubits
            max_qubit = max(qubits)
            if self.program.n_qubits <= max_qubit:
                new_circuit.allocate(max_qubit - self.program.n_qubits + 1)
        
        # Add operations from the other circuit
        for op in other.program.ops:
            # Map qubits
            mapped_qubits = tuple(qubit_map[q] for q in op.qubits)
            
            # Create new operation
            new_op = Op(
                name=op.name,
                qubits=mapped_qubits,
                params=op.params,
                clbits=op.clbits,
                condition=op.condition
            )
            new_circuit.program.add(new_op)
        
        # Update classical bits
        new_circuit.program.n_clbits = max(
            self.program.n_clbits,
            other.program.n_clbits
        )
        
        return new_circuit
        
    def to_gate(self, name: Optional[str] = None) -> 'Gate':
        """Convert the circuit to a gate.
        
        Args:
            name: Optional name for the gate.
            
        Returns:
            A Gate object representing this circuit.
        """
        return Gate(self, name=name or f"{self.name}_gate")
        
    def inverse(self) -> 'Circuit':
        """Return the inverse of the circuit."""
        # Create a new circuit with the same number of qubits
        inv_circuit = Circuit(name=f"{self.name}_inverse")
        if self.program.n_qubits > 0:
            inv_circuit.allocate(self.program.n_qubits)
        
        # Add operations in reverse order with inverted gates
        for op in reversed(self.program.ops):
            if op.name in ['h', 'x', 'y', 'z', 'sx', 'sxdg']:
                # Self-inverse gates
                inv_circuit.program.add(op)
            elif op.name == 's':
                inv_circuit.program.add(Op('sdg', op.qubits, op.params, op.clbits, op.condition))
            elif op.name == 'sdg':
                inv_circuit.program.add(Op('s', op.qubits, op.params, op.clbits, op.condition))
            elif op.name == 't':
                inv_circuit.program.add(Op('tdg', op.qubits, op.params, op.clbits, op.condition))
            elif op.name == 'tdg':
                inv_circuit.program.add(Op('t', op.qubits, op.params, op.clbits, op.condition))
            elif op.name == 'rx':
                inv_circuit.program.add(Op('rx', op.qubits, (-op.params[0],), op.clbits, op.condition))
            elif op.name == 'ry':
                inv_circuit.program.add(Op('ry', op.qubits, (-op.params[0],), op.clbits, op.condition))
            elif op.name == 'rz':
                inv_circuit.program.add(Op('rz', op.qubits, (-op.params[0],), op.clbits, op.condition))
            elif op.name in ['cx', 'cy', 'cz', 'swap']:
                inv_circuit.program.add(op)  # Self-inverse two-qubit gates
            else:
                # For other gates, just add them as is (may not be correct for all cases)
                inv_circuit.program.add(op)
        
        return inv_circuit

def __enter__(self): 
    return self
    
def __exit__(self, exc_type, exc, tb): 
    return False
    
# Add methods to Circuit class
Circuit.__enter__ = __enter__
Circuit.__exit__ = __exit__

# Add gate class
class Gate:
    """A gate that can be applied to a circuit."""
    
    def __init__(self, circuit: Circuit, name: Optional[str] = None):
        """Initialize a gate from a circuit.
        
        Args:
            circuit: The circuit to convert to a gate.
            name: Optional name for the gate.
        """
        self.circuit = circuit
        self.name = name or f"gate_{id(self)}"
        self.num_qubits = circuit.program.n_qubits
    
    def __call__(self, *qubits: int) -> None:
        """Apply the gate to the given qubits."""
        if len(qubits) != self.num_qubits:
            raise ValueError(f"Gate requires {self.num_qubits} qubits, got {len(qubits)}")
        
        # Create a copy of the circuit with the qubit mapping
        mapped_circuit = Circuit()
        qubit_map = {i: q for i, q in enumerate(qubits)}
        
        # Add operations with mapped qubits
        for op in self.circuit.program.ops:
            mapped_qubits = tuple(qubit_map[q] for q in op.qubits)
            mapped_op = Op(
                name=op.name,
                qubits=mapped_qubits,
                params=op.params,
                clbits=op.clbits,
                condition=op.condition
            )
            mapped_circuit.program.add(mapped_op)
        
        # Add the operations to the current circuit
        current_circuit = Circuit._current_circuit
        if current_circuit is None:
            raise RuntimeError("Gate must be used within a circuit context")
            
        for op in mapped_circuit.program.ops:
            current_circuit.program.add(op)
    
    def to_circuit(self) -> Circuit:
        """Convert the gate back to a circuit."""
        return deepcopy(self.circuit)
    
    def inverse(self) -> 'Gate':
        """Return the inverse gate."""
        return Gate(self.circuit.inverse(), name=f"{self.name}_dg")

# Add circuit context management
Circuit._current_circuit = None
