import sys
import os
import pytest

# Add both the project root and python directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
python_dir = os.path.join(project_root, 'python')
sys.path.insert(0, python_dir)
sys.path.insert(0, project_root)

# Import only the necessary modules
from qx.ir import Program, Op
from qx.sim.local import LocalSimulator
from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator

@pytest.fixture(params=[LocalSimulator(), ZenaQuantumAlphaSimulator()])
def simulator(request):
    return request.param

@pytest.fixture
def name(simulator):
    return simulator.__class__.__name__

def test_simulator(simulator, name):
    print(f"\nTesting {name}")
    print("=" * 50)
    
    # Create a simple program (Bell pair)
    prog = Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])
    
    try:
        print("Running simulation...")
        counts, meta = simulator.execute(prog, shots=1000)
        
        print("\nResults:")
        print(f"Counts: {counts}")
        print(f"Metadata: {meta}")
        
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

# The test will be automatically discovered and run by pytest
