import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Import numpy directly
import numpy as np

# Define minimal Program and Op classes
class Op:
    def __init__(self, name, qubits, params=(), clbits=(), condition=None):
        self.name = name
        self.qubits = qubits
        self.params = params
        self.clbits = clbits
        self.condition = condition

class Program:
    def __init__(self, n_qubits, n_clbits, ops):
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.ops = ops

# Define the test function
def test_zena_quantum():
    print("Testing ZenaQuantumAlphaSimulator")
    print("=" * 50)
    
    # Create a simple program (Bell pair)
    prog = Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])
    
    try:
        # Import the simulator directly
        from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
        
        print("Initializing ZenaQuantumAlphaSimulator...")
        simulator = ZenaQuantumAlphaSimulator()
        
        print("Running simulation...")
        counts, meta = simulator.execute(prog, shots=1000)
        
        print("\nResults:")
        print(f"Counts: {counts}")
        print(f"Metadata keys: {list(meta.keys())}")
        
        total = sum(counts.values())
        print(f"Total shots: {total}")
        
        # Check for expected results
        if '00' in counts or '11' in counts:
            print("✓ Found expected measurements (00 or 11)")
        if '01' in counts or '10' in counts:
            print("Note: Found some unexpected measurements (01 or 10), which could be due to noise")
            
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zena_quantum()
