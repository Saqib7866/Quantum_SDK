import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Create dummy modules to prevent matplotlib import
import sys
import types

class DummyModule(types.ModuleType):
    def __getattr__(self, name):
        return None
    
# Replace matplotlib with a dummy
sys.modules['matplotlib'] = DummyModule('matplotlib')
sys.modules['matplotlib.pyplot'] = DummyModule('matplotlib.pyplot')

try:
    # Import our modules
    from qx.ir import Program, Op
    from qx.sim.local import LocalSimulator
    from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
    
    print("[OK] Successfully imported required modules")
    
    # Test 1: Create a simple circuit with LocalSimulator
    print("\n[TEST] LocalSimulator with Bell pair")
    
    # Create a program directly to avoid Circuit class
    prog = Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])
    
    # Use LocalSimulator
    local_sim = LocalSimulator()
    print("Running simulation with LocalSimulator...")
    counts, meta = local_sim.execute(prog, shots=1000)
    
    print("Results:")
    print(f"Counts: {counts}")
    print(f"Metadata keys: {list(meta.keys())}")
    
    # Basic verification
    total_shots = sum(counts.values())
    print(f"Total shots: {total_shots}")
    assert abs(total_shots - 1000) <= 10, f"Expected ~1000 shots, got {total_shots}"
    
    # For Bell state, should have either 00 or 11
    for key in counts:
        assert key in ['00', '11'], f"Unexpected measurement result: {key}"
    
    print("[PASS] LocalSimulator test passed!")
    
    # Test 2: Test ZenaQuantumAlphaSimulator with noise
    print("\n[TEST] ZenaQuantumAlphaSimulator with noise")
    noise_params = {
        'readout_error': 0.1,
        'oneq_error': 0.01,
        'twoq_error': 0.05
    }
    
    zena_sim = ZenaQuantumAlphaSimulator(noise=noise_params)
    print("Running simulation with ZenaQuantumAlphaSimulator...")
    counts, meta = zena_sim.execute(prog, shots=1000)
    
    print("Results:")
    print(f"Counts: {counts}")
    print(f"Metadata keys: {list(meta.keys())}")
    
    # Basic verification
    total_shots = sum(counts.values())
    print(f"Total shots: {total_shots}")
    assert abs(total_shots - 1000) <= 10, f"Expected ~1000 shots, got {total_shots}"
    
    print("[PASS] ZenaQuantumAlphaSimulator test completed!")
    print("\n[SUCCESS] All tests passed!")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
