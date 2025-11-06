import sys
import os
import numpy as np
import threading
import time

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Create a minimal Program class
class Program:
    def __init__(self, n_qubits, n_clbits, ops):
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.ops = ops
        
    def validate(self):
        """Validate the program."""
        # Simple validation: check that all qubit and clbit indices are in range
        for op in self.ops:
            for q in op.qubits:
                if not 0 <= q < self.n_qubits:
                    raise ValueError(f"Qubit index {q} out of range 0..{self.n_qubits-1}")
            for c in op.clbits:
                if not 0 <= c < self.n_clbits:
                    raise ValueError(f"Classical bit index {c} out of range 0..{self.n_clbits-1}")

# Create a minimal Op class
class Op:
    def __init__(self, name, qubits, params=(), clbits=(), condition=None):
        self.name = name
        self.qubits = qubits
        self.params = params
        self.clbits = clbits
        self.condition = condition

def test_zena_quantum():
    print("Testing ZenaQuantumAlphaSimulator")
    print("=" * 50)
    
    # Create a simple program (Bell pair)
    print("Creating a Bell pair circuit...")
    prog = Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])
    
    try:
        # Import the simulator directly
        print("Importing ZenaQuantumAlphaSimulator...")
        from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
        
        # Initialize the simulator
        print("Initializing ZenaQuantumAlphaSimulator...")
        simulator = ZenaQuantumAlphaSimulator()
        
        # Run the simulation
        print("Running simulation...")
        counts, meta = simulator.execute(prog, shots=1000)
        
        # Print results
        print("\nResults:")
        print(f"Counts: {counts}")
        print(f"Metadata: {meta}")
        
        # Basic validation
        if isinstance(counts, dict):
            total = sum(counts.values())
            print(f"Total shots: {total}")
            
            # Check for expected results
            if any(k in counts for k in ['00', '11']):
                print("✓ Found expected measurements (00 or 11)")
            if any(k in counts for k in ['01', '10']):
                print("Note: Found some unexpected measurements (01 or 10), which could be due to noise")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zena_quantum()
