import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qx_ir.core import Circuit, Op, Program
from qx_ir.simulator import StatevectorSimulator

class TestSimulator(unittest.TestCase):
    """Unit tests for the StatevectorSimulator."""

    def setUp(self):
        """Set up the simulator for each test."""
        self.simulator = StatevectorSimulator()

    def test_bell_state_counts(self):
        """Test that the simulator produces the correct counts for a Bell state."""
        # Create a Bell state circuit
        bell_circuit = Circuit(n_qubits=2)
        bell_circuit.add_op(Op(name='h', qubits=[0]))
        bell_circuit.add_op(Op(name='cx', qubits=[0, 1]))

        # Create a program with 1000 shots
        program = Program(circuits=[bell_circuit], config={'shots': 1000})

        # Run the simulator
        counts = self.simulator.run(program)

        # Check the results
        self.assertEqual(sum(counts.values()), 1000)
        # The keys should be '00' and '11' for a Bell state
        self.assertIn('00', counts)
        self.assertIn('11', counts)
        # Check that the counts for '00' and '11' are roughly equal
        self.assertAlmostEqual(counts['00'] / 1000, 0.5, delta=0.1)
        self.assertAlmostEqual(counts['11'] / 1000, 0.5, delta=0.1)

    def test_ghz_state_counts(self):
        """Test that the simulator produces the correct counts for a GHZ state."""
        # Create a 3-qubit GHZ state circuit
        ghz_circuit = Circuit(n_qubits=3)
        ghz_circuit.add_op(Op(name='h', qubits=[0]))
        ghz_circuit.add_op(Op(name='cx', qubits=[0, 1]))
        ghz_circuit.add_op(Op(name='cx', qubits=[0, 2]))

        # Create a program with 2000 shots
        program = Program(circuits=[ghz_circuit], config={'shots': 2000})

        # Run the simulator
        counts = self.simulator.run(program)

        # Check the results
        self.assertEqual(sum(counts.values()), 2000)
        self.assertIn('000', counts)
        self.assertIn('111', counts)
        self.assertAlmostEqual(counts['000'] / 2000, 0.5, delta=0.1)
        self.assertAlmostEqual(counts['111'] / 2000, 0.5, delta=0.1)

if __name__ == '__main__':
    unittest.main()
