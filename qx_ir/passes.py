import numpy as np
from .core import Circuit, Op
from .transpiler import Pass
from .target import Target

class RemoveRedundantH(Pass):
    """
    Removes consecutive Hadamard (H) gates on the same qubit.

    Methods:
        run(circuit, target): Returns a new circuit with redundant H–H pairs removed.
    """
    def run(self, circuit: Circuit, target: Target) -> Circuit:
        """Removes any occurrence of two consecutive H gates on the same qubit."""
        new_circuit = Circuit(n_qubits=circuit.n_qubits)
        # Using a copy to safely remove items while iterating
        instructions_to_process = circuit.instructions[:]
        
        skip_next = False
        for i, op in enumerate(instructions_to_process):
            if skip_next:
                skip_next = False
                continue

            if op.name == 'h' and i + 1 < len(instructions_to_process):
                next_op = instructions_to_process[i + 1]
                if next_op.name == 'h' and op.qubits == next_op.qubits:
                    # Found a pair of H-H gates, so we skip both
                    skip_next = True
                    continue
            
            new_circuit.add_op(op)
            
        return new_circuit


class DecomposeUnsupportedGates(Pass):
    """
    Decomposes unsupported gates into the target’s supported basis gates.

    Methods:
        run(circuit, target): Rewrites unsupported gates using known decompositions.
        _get_decomposition(op): Returns a decomposition rule if available.
        _decompose_h(op): Decomposes an H gate into Rz and SX gates.
        _decompose_ccx(op): Decomposes a CCX (Toffoli) gate into standard gates.
    """

    def run(self, circuit: Circuit, target: Target) -> Circuit:
        """Decomposes unsupported gates into the target's basis gates."""
        new_circuit = Circuit(n_qubits=circuit.n_qubits)
        for op in circuit.instructions:
            if op.name in target.basis_gates:
                new_circuit.add_op(op)
            else:
                # Try to decompose the gate
                decomposition = self._get_decomposition(op)
                if decomposition is None:
                    raise ValueError(
                        f"Gate '{op.name}' is not supported by the target and has no decomposition rule."
                    )
                
                for decomposed_op in decomposition:
                    # Here, we assume the decomposed ops are supported.
                    # A more robust transpiler would recursively decompose.
                    new_circuit.add_op(decomposed_op)
        return new_circuit

    def _get_decomposition(self, op: Op) -> list[Op] | None:
        """Returns a list of decomposed operations or None if no rule exists."""
        if op.name == 'ccx':
            return self._decompose_ccx(op)
        elif op.name == 'h':
            return self._decompose_h(op)
        # Add other decomposition rules here
        return None

    def _decompose_h(self, op: Op) -> list[Op]:
        """Decomposes an H gate into Rz and SX gates."""
        q = op.qubits[0]
        return [
            Op(name='rz', qubits=[q], params=[np.pi / 2]),
            Op(name='sx', qubits=[q]),
            Op(name='rz', qubits=[q], params=[np.pi / 2]),
        ]

    def _decompose_ccx(self, op: Op) -> list[Op]:
        """Decomposes a CCX (Toffoli) gate into H, T, Tdg, and CX gates."""
        c1, c2, t = op.qubits
        return [
            Op(name='h', qubits=[t]),
            Op(name='cx', qubits=[c2, t]),
            Op(name='tdg', qubits=[t]),
            Op(name='cx', qubits=[c1, t]),
            Op(name='t', qubits=[t]),
            Op(name='cx', qubits=[c2, t]),
            Op(name='tdg', qubits=[t]),
            Op(name='cx', qubits=[c1, t]),
            Op(name='t', qubits=[c2]),
            Op(name='t', qubits=[t]),
            Op(name='cx', qubits=[c1, c2]),
            Op(name='h', qubits=[t]),
            Op(name='t', qubits=[c1]),
            Op(name='tdg', qubits=[c2]),
            Op(name='cx', qubits=[c1, c2]),
        ]


class CheckQubitMapping(Pass):
    """
    Validates qubit usage and connectivity against the target device.

    Methods:
        run(circuit, target): Ensures qubit count and coupling map match the target.
    """

    def run(self, circuit: Circuit, target: Target) -> Circuit:
        """Validates the circuit's qubit count and two-qubit gate connectivity."""
        # 1. Check if the circuit has too many qubits
        if circuit.n_qubits > target.n_qubits:
            raise ValueError(
                f"Circuit requires {circuit.n_qubits} qubits, but the target only has {target.n_qubits}."
            )

        # 2. Check the connectivity of two-qubit gates
        for op in circuit.instructions:
            if len(op.qubits) == 2:
                q1, q2 = op.qubits
                # Check both (q1, q2) and (q2, q1) as the coupling map may not be symmetric
                if (q1, q2) not in target.coupling_map and (q2, q1) not in target.coupling_map:
                    raise ValueError(
                        f"Gate '{op.name}' on qubits ({q1}, {q2}) is not supported by the target's coupling map."
                    )

        # If all checks pass, return the original circuit
        return circuit

