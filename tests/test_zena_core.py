import sys
import os
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Create a mock for the visualization modules
class MockVis:
    def draw_text(*args, **kwargs):
        pass
    
    def draw_matplotlib(*args, **kwargs):
        pass

sys.modules['qx.vis.draw'] = MockVis()
sys.modules['qx.vis.plot'] = MockVis()

# Now import the simulator
try:
    print("Importing ZenaQuantumAlphaSimulator...")
    from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
    from qx.ir import Program, Op
    
    print("Successfully imported ZenaQuantumAlphaSimulator")
    
    # Create a simple program (Bell pair)
    print("\nCreating a Bell pair circuit...")
    prog = Program(n_qubits=2, n_clbits=2, ops=[
        Op("h", (0,)),
        Op("cx", (0, 1)),
        Op("measure", (0,), (), (0,)),
        Op("measure", (1,), (), (1,))
    ])
    
    # Initialize the simulator
    print("Initializing ZenaQuantumAlphaSimulator...")
    simulator = ZenaQuantumAlphaSimulator()
    
    # Run the simulation
    print("Running simulation...")
    counts, meta = simulator.execute(prog, shots=1000)
    
    # Print results
    print("\nResults:")
    print(f"Counts: {counts}")
    print(f"Metadata: {meta}")
    
    # Basic validation
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
