import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qx_ir.core import Circuit, Op
from qx_ir.target import Target
from qx_ir.reports import get_circuit_depth, get_gate_counts, estimate_circuit_fidelity

class TestReports(unittest.TestCase):
    """Unit tests for the reporting functions."""

    def test_circuit_depth(self):
        """Test that the circuit depth is calculated correctly."""
        circuit = Circuit(n_qubits=2)
        circuit.add_op(Op(name='h', qubits=[0]))
        circuit.add_op(Op(name='h', qubits=[1]))
        circuit.add_op(Op(name='cx', qubits=[0, 1]))
        circuit.add_op(Op(name='h', qubits=[0]))

        # Expected depth:
        # Layer 1: H(0), H(1)
        # Layer 2: CX(0, 1)
        # Layer 3: H(0)
        # Depth should be 3
        self.assertEqual(get_circuit_depth(circuit), 3)

    def test_gate_counts(self):
        """Test that the gate counts are calculated correctly."""
        circuit = Circuit(n_qubits=2)
        circuit.add_op(Op(name='h', qubits=[0]))
        circuit.add_op(Op(name='h', qubits=[1]))
        circuit.add_op(Op(name='cx', qubits=[0, 1]))

        counts = get_gate_counts(circuit)
        self.assertEqual(counts['1q'], 2)
        self.assertEqual(counts['2q'], 1)

    def test_fidelity_estimation(self):
        """Test that the circuit fidelity is estimated correctly."""
        # Load the target profile
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        profile_path = os.path.join(project_root, 'qxir_v1.json')
        target = Target.from_file(profile_path)

        # Create a circuit
        circuit = Circuit(n_qubits=2)
        circuit.add_op(Op(name='sx', qubits=[0]))
        circuit.add_op(Op(name='cx', qubits=[0, 1]))

        # Calculate the expected fidelity: 0.998 (sx) * 0.99 (cx)
        expected_fidelity = 0.998 * 0.99

        # Estimate the fidelity
        estimated_fidelity = estimate_circuit_fidelity(circuit, target)

        self.assertAlmostEqual(estimated_fidelity, expected_fidelity)

if __name__ == '__main__':
    unittest.main()
