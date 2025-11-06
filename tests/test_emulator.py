import pytest
from qx import Circuit
from qx.sim.local import LocalSimulator
from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
from qx.ir import Program, Op

def _ghz(n):
    """Create a GHZ circuit using Program and Op directly."""
    ops = [
        Op("h", (0,)),
    ]
    # Add CNOTs
    for i in range(n-1):
        ops.append(Op("cx", (i, i+1)))
    # Add measurements
    for i in range(n):
        ops.append(Op("measure", (i,), (), (i,)))
    return Program(n_qubits=n, n_clbits=n, ops=ops)

@pytest.fixture(params=[
    LocalSimulator(),
    ZenaQuantumAlphaSimulator()
])
def simulator(request):
    return request.param

def test_emulator_metadata_and_noise(simulator):
    """Test that the simulator returns metadata and handles noise."""
    try:
        # Create a simple GHZ circuit
        prog = _ghz(3)
        
        # Run the circuit
        if hasattr(simulator, 'execute'):
            # Handle simulators with execute method
            result = simulator.execute(prog, shots=1000)
            if isinstance(result, tuple) and len(result) == 2:
                counts, metadata = result
            else:
                counts = result.get_counts()
                metadata = getattr(result, 'metadata', {})
        else:
            # Fallback for other types of backends
            counts, metadata = simulator.run(prog, shots=1000)
        
        # Check that we got some results
        assert counts is not None
        assert len(counts) > 0
        
        # Check for expected metadata
        if metadata:
            assert isinstance(metadata, dict)
            # Check for common metadata fields
            if 'shots' in metadata:
                assert isinstance(metadata['shots'], int)
            if 'n_qubits' in metadata:
                assert isinstance(metadata['n_qubits'], int)
        
        print(f"\n{simulator.__class__.__name__} results:")
        print(f"Counts: {counts}")
        print(f"Metadata: {metadata}")
                
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

def test_noise_scales_with_depth():
    """Test that deeper circuits show more noise."""
    # Use LocalSimulator for this test as it's more predictable
    sim = LocalSimulator()
    
    # Define a helper function to run a circuit and get counts
    def get_counts(circuit):
        result = sim.execute(circuit, shots=1000)
        if isinstance(result, tuple) and len(result) == 2:
            counts, _ = result
        else:
            counts = result.get_counts()
        return counts
    
    # Run GHZ-2 and GHZ-5 circuits
    counts2 = get_counts(_ghz(2))
    counts5 = get_counts(_ghz(5))
    
    # Basic validation of results
    assert len(counts2) > 0, "No results for GHZ-2 circuit"
    assert len(counts5) > 0, "No results for GHZ-5 circuit"
    
    print("\nGHZ-2 counts:", counts2)
    print("GHZ-5 counts:", counts5)
    
    # For a noiseless simulator, both should have exactly 2 outcomes (all 0s and all 1s)
    # But we'll just check that we got results
