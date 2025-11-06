import sys
import os
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Now import the required modules
from qx.circuit import Circuit
from qx.sim.local import LocalSimulator
from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator

def run_bell_pair_test(simulator, name):
    print(f"\n--- {name}: Testing Bell pair circuit ---")
    try:
        # Create a Bell pair circuit
        qc = Circuit()
        q0, q1 = qc.allocate(2)
        qc.h(q0)
        qc.cx(q0, q1)
        qc.measure(q0, q1)
        
        # Run simulation
        print(f"Running {name}...")
        counts, meta = simulator.execute(qc.program, shots=1000)
        
        # Print results
        print("Counts:", counts)
        print("Metadata keys:", list(meta.keys()))
        
        # Basic verification
        assert len(counts) <= 2  # Should only have 00 and 11
        total = sum(counts.values())
        assert abs(total - 1000) <= 10  # Should have close to 1000 shots
        
        # For Bell state, should be roughly 50/50 between 00 and 11
        if len(counts) == 2:
            ratio = min(counts.values()) / max(counts.values())
            print(f"Ratio of counts: {ratio:.2f}")
            assert 0.7 <= ratio <= 1.3  # Within 30% of 1.0 ratio
        
        print(f"✓ {name} Bell pair test passed!")
        return True
    except Exception as e:
        print(f"❌ {name} test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_single_qubit_test(simulator, name):
    print(f"\n--- {name}: Testing single-qubit operations ---")
    try:
        # Test X, H, and measure
        qc = Circuit()
        q = qc.allocate(1)[0]
        qc.x(q)
        qc.h(q)
        qc.measure(q)
        
        # Run simulation
        print(f"Running {name}...")
        counts, meta = simulator.execute(qc.program, shots=1000)
        
        # Print results
        print("Counts:", counts)
        print("Metadata keys:", list(meta.keys()))
        
        # Basic verification
        assert len(counts) <= 2  # Should only have 0 and 1
        total = sum(counts.values())
        assert abs(total - 1000) <= 10  # Should have close to 1000 shots
        
        # For XH|0>, should be roughly 50/50 between 0 and 1
        if len(counts) == 2:
            ratio = min(counts.values()) / max(counts.values())
            print(f"Ratio of counts: {ratio:.2f}")
            assert 0.7 <= ratio <= 1.3  # Within 30% of 1.0 ratio
        
        print(f"✓ {name} single-qubit test passed!")
        return True
    except Exception as e:
        print(f"❌ {name} test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_conditional_operations():
    print("\n--- Testing conditional operations ---")
    # Test conditional X gate based on measurement
    qc = Circuit()
    q0, q1 = qc.allocate(2)
    c0, c1 = qc.measure(q0, q1)  # c0=0, c1=0 initially
    qc.x(q1, condition=c0)  # Should not execute since c0=0
    qc.measure(q1, c1)  # Measure q1 into c1
    
    # Run simulation
    sim = LocalSimulator()
    print("Running conditional operations test...")
    counts, meta = sim.execute(qc.program, shots=1000)
    
    # Print results
    print("Counts:", counts)
    print("Metadata keys:", list(meta.keys()))
    
    # The result is 4 bits due to how the circuit is constructed
    # We expect '0000' because:
    # 1. First two bits are the measurements of q0 and q1
    # 2. Next two bits are the classical bits c0 and c1
    # Since X is conditional on c0=0, it doesn't execute, so q1 remains |0>
    assert "0000" in counts, f"Expected '0000' in counts, got {counts}"
    assert counts["0000"] == 1000, f"Expected 1000 '0000' results, got {counts['0000']}"
    
    print("✓ Conditional operations test passed!")

def main():
    print("Starting quantum simulator tests...")
    
    # Test LocalSimulator
    local_sim = LocalSimulator()
    local_bell = run_bell_pair_test(local_sim, "LocalSimulator")
    local_single = run_single_qubit_test(local_sim, "LocalSimulator")
    
    # Test ZenaQuantumAlphaSimulator
    zena_sim = ZenaQuantumAlphaSimulator()
    zena_bell = run_bell_pair_test(zena_sim, "ZenaQuantumAlphaSimulator")
    zena_single = run_single_qubit_test(zena_sim, "ZenaQuantumAlphaSimulator")
    
    # Test conditional operations
    cond_test = test_conditional_operations()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"LocalSimulator Bell pair test: {'PASSED' if local_bell else 'FAILED'}")
    print(f"LocalSimulator single-qubit test: {'PASSED' if local_single else 'FAILED'}")
    print(f"ZenaQuantumAlphaSimulator Bell pair test: {'PASSED' if zena_bell else 'FAILED'}")
    print(f"ZenaQuantumAlphaSimulator single-qubit test: {'PASSED' if zena_single else 'FAILED'}")
    print(f"Conditional operations test: {'PASSED' if cond_test else 'FAILED'}")
    
    # Check if all tests passed
    all_passed = all([local_bell, local_single, zena_bell, zena_single, cond_test])
    if all_passed:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
