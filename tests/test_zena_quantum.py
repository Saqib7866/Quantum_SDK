import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

from qx import Circuit, backend, run

def test_zena_quantum():
    print("Testing ZenaQuantumAlpha Backend")
    print("==============================")
    
    # Create a simple circuit
    print("\nCreating a Bell pair circuit...")
    qc = Circuit()
    q0, q1 = qc.allocate(2)  # Allocate 2 qubits
    qc.h(q0)                 # Hadamard on q0
    qc.cx(q0, q1)            # CNOT with q0 as control and q1 as target
    qc.measure(q0, q1)       # Measure both qubits
    
    # Get the ZenaQuantumAlpha backend
    print("Initializing ZenaQuantumAlpha backend...")
    try:
        zena_backend = backend("zenaquantum-alpha")
        print("Successfully initialized ZenaQuantumAlpha backend")
    except Exception as e:
        print(f"Error initializing backend: {e}")
        return
    
    # Run the circuit
    print("\nRunning circuit on ZenaQuantumAlpha...")
    try:
        job = run(qc, zena_backend, shots=1000)
        result = job.result()
        print("\nResults:")
        print(f"Counts: {result.counts}")
        print(f"Metadata: {result.metadata}")
        
        # Basic validation
        total_shots = sum(result.counts.values())
        print(f"\nTotal shots: {total_shots}")
        
        # For a Bell pair, we expect mostly 00 and 11
        if '01' in result.counts or '10' in result.counts:
            print("Warning: Got unexpected measurement results (01 or 10) in Bell pair test")
        else:
            print("Bell pair test passed!")
            
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zena_quantum()
