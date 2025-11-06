import sys
import os
import importlib.util

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

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

def test_simulator():
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
        # Dynamically import the simulator to avoid importing the entire qx package
        spec = importlib.util.spec_from_file_location(
            "zenaquantum_alpha", 
            os.path.join("python", "qx", "sim", "zenaquantum_alpha.py")
        )
        zena_alpha = importlib.util.module_from_spec(spec)
        sys.modules["zenaquantum_alpha"] = zena_alpha
        spec.loader.exec_module(zena_alpha)
        
        # Import LocalSimulator for the parent class
        from qx.sim.local import LocalSimulator
        
        print("Initializing ZenaQuantumAlphaSimulator...")
        simulator = zena_alpha.ZenaQuantumAlphaSimulator()
        
        print("Running simulation...")
        result = simulator.execute(prog, shots=1000)
        
        # Handle different return types
        if isinstance(result, tuple) and len(result) == 2:
            counts, meta = result
        elif hasattr(result, 'counts') and hasattr(result, 'metadata'):
            counts = result.counts
            meta = result.metadata
        else:
            counts = result
            meta = {}
        
        print("\nResults:")
        print(f"Counts: {counts}")
        print(f"Metadata: {meta}")
        
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
    test_simulator()
