from .core import Circuit
from .target import Target

def get_circuit_depth(circuit: Circuit) -> int:
    """
    Calculate the depth of a circuit based on sequential gate layers.

    Assumes single-qubit gates can run in parallel and adjusts depth
    for two-qubit gates acting on connected qubits.
    """
    # This is a simple implementation that assumes gates on different qubits can be parallelized.
    # A more accurate implementation would need to consider the target's connectivity.
    qubit_layers = [0] * circuit.n_qubits
    for op in circuit.instructions:
        if len(op.qubits) == 1:
            q = op.qubits[0]
            qubit_layers[q] += 1
        elif len(op.qubits) == 2:
            q1, q2 = op.qubits
            # The new layer for both qubits is the max of their current layers, plus one
            new_layer = max(qubit_layers[q1], qubit_layers[q2]) + 1
            qubit_layers[q1] = new_layer
            qubit_layers[q2] = new_layer
    return max(qubit_layers)

def get_gate_counts(circuit: Circuit) -> dict[str, int]:
    """
    Count single-qubit and two-qubit gates in a circuit.

    Returns a dict with '1q' and '2q' gate counts.
    """
    counts = {
        '1q': 0,
        '2q': 0,
    }
    for op in circuit.instructions:
        if len(op.qubits) == 1:
            counts['1q'] += 1
        elif len(op.qubits) == 2:
            counts['2q'] += 1
    return counts

def estimate_circuit_fidelity(circuit: Circuit, target: Target) -> float:
    """
    Estimate circuit fidelity using target gate fidelities.

    Multiplies the fidelity of each gate; returns 0.0 if any gate
    is missing from the target’s fidelity data.
    """
    fidelity = 1.0
    for op in circuit.instructions:
        gate_fidelity = target.gate_fidelities.get(op.name)
        if gate_fidelity is None:
            # If a gate is not in the fidelity list, we can't calculate the fidelity.
            # A more robust implementation might raise a warning or an error.
            return 0.0
        fidelity *= gate_fidelity
    return fidelity
