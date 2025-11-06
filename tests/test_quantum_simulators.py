import sys
import os
import numpy as np

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

def run_test(test_name, circuit_func, expected_states=None, shots=1000, noise_params=None):
    print(f"\n[TEST] {test_name}")
    print("-" * 60)
    
    # Import inside function to ensure clean state
    from qx.ir import Program, Op
    from qx.sim.local import LocalSimulator
    from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
    
    # Create the circuit
    prog = circuit_func()
    
    # Test LocalSimulator
    print("\n1. Testing LocalSimulator (ideal simulation):")
    local_sim = LocalSimulator()
    counts, meta = local_sim.execute(prog, shots=shots)
    
    print(f"Results: {counts}")
    print(f"Metadata: {list(meta.keys())}")
    
    # Basic validation
    total = sum(counts.values())
    assert abs(total - shots) <= 10, f"Expected {shots} shots, got {total}"
    
    if expected_states:
        for state in counts:
            assert state in expected_states, f"Unexpected state {state} in results"
    
    # Test ZenaQuantumAlphaSimulator
    print("\n2. Testing ZenaQuantumAlphaSimulator (with noise):")
    noise = noise_params or {
        'readout_error': 0.1,
        'oneq_error': 0.01,
        'twoq_error': 0.05
    }
    
    zena_sim = ZenaQuantumAlphaSimulator(noise=noise)
    counts, meta = zena_sim.execute(prog, shots=shots)
    
    print(f"Results: {counts}")
    print(f"Metadata: {list(meta.keys())}")
    
    # Basic validation
    total = sum(counts.values())
    assert abs(total - shots) <= 10, f"Expected {shots} shots, got {total}"
    
    print(f"\n[PASS] {test_name} completed successfully!")

def bell_pair_circuit():
    """Create a Bell pair circuit: H on q0, CNOT q0->q1, measure both"""
    from qx.ir import Program, Op
    return Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])

def single_qubit_ops():
    """Test various single-qubit gates"""
    from qx.ir import Program, Op
    return Program(n_qubits=1, n_clbits=1, ops=[
        Op("h", (0,)),
        Op("s", (0,)),
        Op("t", (0,)),
        Op("x", (0,)),
        Op("measure", (0,), (), (0,))
    ])

def multi_qubit_ops():
    """Test multi-qubit operations"""
    from qx.ir import Program, Op
    return Program(n_qubits=3, n_clbits=3, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("ccx", (0, 1, 2)),  # Toffoli gate
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,)),
        Op("measure", (2,), (), (2,))
    ])

def parameterized_gates():
    """Test parameterized gates"""
    from qx.ir import Program, Op
    import math
    return Program(n_qubits=2, n_clbits=1, ops=[
        Op("rx", (0,), (math.pi/2,)),
        Op("ry", (1,), (math.pi/4,)),
        Op("rz", (0,), (math.pi/8,)),
        Op("crz", (0, 1), (math.pi/4,)),
        Op("measure", (0,), (), (0,))
    ])

if __name__ == "__main__":
    print("=== Quantum Simulator Test Suite ===\n")
    
    # Run all test cases
    test_cases = [
        ("Bell Pair Circuit", bell_pair_circuit, ['00', '11']),
        ("Single Qubit Operations", single_qubit_ops, None),
        ("Multi-Qubit Operations", multi_qubit_ops, None),
        ("Parameterized Gates", parameterized_gates, None)
    ]
    
    for name, func, expected in test_cases:
        run_test(name, func, expected_states=expected, shots=1000)
    
    print("\n=== All tests completed successfully! ===")
