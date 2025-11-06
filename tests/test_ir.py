import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qx_ir.core import Op, Circuit, Program

class TestCoreIR(unittest.TestCase):
    """Unit tests for the core IR classes."""

    def test_op_creation(self):
        """Test the creation of an Op object."""
        op = Op(name='h', qubits=[0], params=[3.14])
        self.assertEqual(op.name, 'h')
        self.assertEqual(op.qubits, [0])
        self.assertEqual(op.params, [3.14])

    def test_circuit_creation(self):
        """Test the creation of a Circuit object."""
        circuit = Circuit(n_qubits=2)
        self.assertEqual(circuit.n_qubits, 2)
        self.assertEqual(len(circuit.instructions), 0)

    def test_add_op_to_circuit(self):
        """Test adding a valid operation to a circuit."""
        circuit = Circuit(n_qubits=2)
        op = Op(name='cx', qubits=[0, 1])
        circuit.add_op(op)
        self.assertEqual(len(circuit.instructions), 1)
        self.assertIs(circuit.instructions[0], op)

    def test_add_invalid_op_to_circuit(self):
        """Test that adding an op with out-of-bounds qubits raises an error."""
        circuit = Circuit(n_qubits=1)
        op = Op(name='cx', qubits=[0, 1]) # Qubit 1 is out of bounds
        with self.assertRaises(ValueError):
            circuit.add_op(op)

    def test_program_creation(self):
        """Test the creation of a Program object."""
        c1 = Circuit(n_qubits=1)
        c2 = Circuit(n_qubits=2)
        program = Program(circuits=[c1, c2], config={'shots': 1024})
        self.assertEqual(len(program.circuits), 2)
        self.assertEqual(program.config['shots'], 1024)

if __name__ == '__main__':
    unittest.main()
