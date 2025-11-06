import sys
import os
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Import only the necessary modules
try:
    from qx.circuit import Circuit
    from qx.sim.local import LocalSimulator
    from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
    from qx.ir import Program, Op
    
    print("✓ Successfully imported all required modules")
    
    # Test 1: Create a simple circuit with LocalSimulator
    print("\n--- Testing LocalSimulator with Bell pair ---")
    qc = Circuit()
    q0, q1 = qc.allocate(2)
    qc.h(q0)
    qc.cx(q0, q1)
    qc.measure(q0, q1)
    
    # Use LocalSimulator
    local_sim = LocalSimulator()
    print("Running simulation with LocalSimulator...")
    counts, meta = local_sim.execute(qc.program, shots=1000)
    
    print("LocalSimulator results:")
    print(f"Counts: {counts}")
    print(f"Metadata: {list(meta.keys())}")
    
    # Basic verification
    assert sum(counts.values()) == 1000, f"Expected 1000 shots, got {sum(counts.values())}"
    print("✓ LocalSimulator test passed!")
    
    # Test 2: Test ZenaQuantumAlphaSimulator
    print("\n--- Testing ZenaQuantumAlphaSimulator with Bell pair ---")
    zena_sim = ZenaQuantumAlphaSimulator()
    print("Running simulation with ZenaQuantumAlphaSimulator...")
    counts, meta = zena_sim.execute(qc.program, shots=1000)
    
    print("ZenaQuantumAlphaSimulator results:")
    print(f"Counts: {counts}")
    print(f"Metadata: {list(meta.keys())}")
    
    # Basic verification
    assert sum(counts.values()) == 1000, f"Expected 1000 shots, got {sum(counts.values())}"
    print("✓ ZenaQuantumAlphaSimulator test passed!")
    
    print("\n🎉 All tests completed successfully!")
    
except Exception as e:
    print(f"\n❌ Test failed with error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
