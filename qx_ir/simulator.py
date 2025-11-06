import numpy as np
from collections import Counter
from .core import Circuit, Program

class StatevectorSimulator:
    """
    Minimal state-vector simulator for QX-IR circuits.

    Uses dense NumPy state evolution for a single circuit and samples
    measurement outcomes from the final state.

    Attributes:
        GATE_MAP (dict[str, np.ndarray]): Built-in 1-qubit gate matrices
            (Hadamard `h`, Pauli-X `x`, phase `t`, and `tdg`).

    Methods:
        run(program, noise_model=None):
            Execute the first circuit in `program`, simulate its final
            statevector, and return shot counts (default shots from
            `program.config['shots']`, 1024 if missing).
        _simulate_statevector(circuit):
            Apply gates in order to produce the final statevector; supports
            1-qubit gates in `GATE_MAP`, `cx`, and ignores `measure` (sampling
            is done in `run`).
        _construct_operator(gate_matrix, target_qubit, n_qubits):
            Build the n-qubit operator for a single-qubit gate via Kronecker
            products (identity elsewhere).
        _construct_cnot(control_qubit, target_qubit, n_qubits):
            Build the full n-qubit CNOT matrix by mapping basis states.
    """
    GATE_MAP = {
        'h': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        'x': np.array([[0, 1], [1, 0]]),
        't': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
        'tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]),
    }

    def run(self, program: Program, noise_model=None):
        """Run a program and return measurement counts."""
        # For now, we assume a single circuit in the program
        if not program.circuits:
            return {}
        circuit = program.circuits[0]
        shots = program.config.get('shots', 1024)

        # Get the final state vector
        final_state = self._simulate_statevector(circuit)

        # Calculate probabilities
        probabilities = np.abs(final_state)**2

        # Generate basis states labels (e.g., '00', '01', '10', '11')
        basis_states = [format(i, f'0{circuit.n_qubits}b') for i in range(2**circuit.n_qubits)]

        # Sample from the distribution
        measured_outcomes = np.random.choice(basis_states, size=shots, p=probabilities)

        # Tally the results
        counts = Counter(measured_outcomes)

        if noise_model:
            # In the future, a noise model could be applied here to modify the counts
            pass

        return dict(counts)

    def _simulate_statevector(self, circuit: Circuit):
        """Simulate the circuit and return the final state vector."""
        n_qubits = circuit.n_qubits
        # Initialize state to |0...0>
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1

        # Har gate (operation) ko ek-ek karke read karta hai.
        for op in circuit.instructions:
            # Agar gate supported hai (h, x, t, tdg), uska matrix le leta hai:
            if op.name in self.GATE_MAP:
                gate_matrix = self.GATE_MAP[op.name]
                target_qubit = op.qubits[0]
                # Construct the full operator for the n-qubit system
                op_matrix = self._construct_operator(gate_matrix, target_qubit, n_qubits)
                state = op_matrix @ state
            elif op.name == 'cx':
                control_qubit, target_qubit = op.qubits
                # Construct the CNOT operator
                op_matrix = self._construct_cnot(control_qubit, target_qubit, n_qubits)
                state = op_matrix @ state
            elif op.name == 'measure':
                # Measurement is handled by sampling after the state vector is computed
                pass
            else:
                print(f"Warning: Operation '{op.name}' is not supported by the simulator and will be ignored.")

        return state

    def _construct_operator(self, gate_matrix, target_qubit, n_qubits):
        """Construct the n-qubit operator for a single-qubit gate."""
        # Start with an identity matrix of the correct size
        op_matrix = np.identity(1)
        for qubit in range(n_qubits):
            if qubit == target_qubit:
                op_matrix = np.kron(op_matrix, gate_matrix)
            else:
                op_matrix = np.kron(op_matrix, np.identity(2))
        return op_matrix

    def _construct_cnot(self, control_qubit, target_qubit, n_qubits):
        """Construct the n-qubit CNOT operator."""
        cnot = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        for i in range(2**n_qubits):
            binary_i = format(i, f'0{n_qubits}b')
            if binary_i[control_qubit] == '1':
                # Flip the target qubit
                binary_j_list = list(binary_i)
                binary_j_list[target_qubit] = '0' if binary_i[target_qubit] == '1' else '1'
                j = int("".join(binary_j_list), 2)
                cnot[j, i] = 1
            else:
                # Do nothing
                cnot[i, i] = 1
        return cnot
