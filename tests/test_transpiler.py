import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qx_ir.core import Circuit, Op
from qx_ir.target import Target
from qx_ir.transpiler import PassManager
from qx_ir.passes import DecomposeUnsupportedGates, CheckQubitMapping

class TestTranspiler(unittest.TestCase):
    """Unit tests for the transpiler passes."""

    def test_decompose_unsupported_gates(self):
        """Test that a CCX gate is decomposed into the target's basis gates."""
        # Define a target that supports H, T, Tdg, and CX, but not CCX
        target_data = {
            "n_qubits": 3,
            "basis_gates": ["h", "t", "tdg", "cx"],
            "coupling_map": [[0, 1], [1, 2]]
        }
        target = Target(target_data)

        # Create a circuit with a CCX gate
        circuit = Circuit(n_qubits=3)
        circuit.add_op(Op(name='ccx', qubits=[0, 1, 2]))

        # Set up the pass manager
        pass_manager = PassManager(passes=[DecomposeUnsupportedGates()])

        # Run the transpiler
        decomposed_circuit = pass_manager.run(circuit, target)

        # Check that the CCX gate is gone
        self.assertNotIn('ccx', [op.name for op in decomposed_circuit.instructions])

        # Check that the number of operations is correct for the decomposition
        self.assertEqual(len(decomposed_circuit.instructions), 15)

    def test_invalid_qubit_mapping(self):
        """Test that an error is raised for a circuit with invalid connectivity."""
        # Define a target with a linear coupling map: 0-1-2
        target_data = {
            "n_qubits": 3,
            "basis_gates": ["cx"],
            "coupling_map": [[0, 1], [1, 2]]
        }
        target = Target(target_data)

        # Create a circuit with a CX gate between non-connected qubits (0 and 2)
        circuit = Circuit(n_qubits=3)
        circuit.add_op(Op(name='cx', qubits=[0, 2]))

        # Set up the pass manager
        pass_manager = PassManager(passes=[CheckQubitMapping()])

        # Assert that a ValueError is raised when running the transpiler
        with self.assertRaises(ValueError):
            pass_manager.run(circuit, target)

if __name__ == '__main__':
    unittest.main()
