import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qx_ir.target import Target

class TestTarget(unittest.TestCase):
    """Unit tests for the Target class."""

    def test_load_from_file(self):
        """Test that a target profile is correctly loaded from a JSON file."""
        # The test assumes 'qxir_v1.json' is in the project root
        # We need to construct the path relative to this test file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        profile_path = os.path.join(project_root, 'qxir_v1.json')

        target = Target.from_file(profile_path)

        self.assertEqual(target.name, 'spinq5')
        self.assertEqual(target.n_qubits, 5)
        self.assertIn('cx', target.basis_gates)
        self.assertIn((1, 2), target.coupling_map)

    def test_file_not_found(self):
        """Test that a FileNotFoundError is raised for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            Target.from_file('non_existent_file.json')

if __name__ == '__main__':
    unittest.main()
