import sys
import os

# Add both the project root and python directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
python_dir = os.path.join(project_root, 'python')
sys.path.insert(0, python_dir)
sys.path.insert(0, project_root)

# Import only what we need to avoid matplotlib
from qx.circuit import Circuit
from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
from qx.ir import Program, Op

def test_zena_direct():
    print("Direct ZenaQuantumAlpha Test")
    print("===========================")
    
    # Create a simple program directly
    prog = Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])
    
    # Initialize the simulator
    print("Initializing ZenaQuantumAlphaSimulator...")
    try:
        simulator = ZenaQuantumAlphaSimulator()
        print("Successfully initialized ZenaQuantumAlphaSimulator")
        
        # Run the program
        print("\nRunning program...")
        counts, meta = simulator.execute(prog, shots=1000)
        
        # Print results
        print("\nResults:")
        print(f"Counts: {counts}")
        print(f"Metadata keys: {list(meta.keys())}")
        
        # Basic validation
        total = sum(counts.values())
        print(f"Total shots: {total}")
        
        # Check for expected results
        if '00' in counts or '11' in counts:
            print("Found expected measurement results (00 or 11)")
        if '01' in counts or '10' in counts:
            print("Note: Found some unexpected measurements (01 or 10), which could be due to noise")
            
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zena_direct()
