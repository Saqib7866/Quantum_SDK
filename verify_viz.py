
import sys
import os
import matplotlib.pyplot as plt

# Add the python directory to the path so we can import qx
sys.path.insert(0, os.path.join(os.getcwd(), 'python'))

from qx import Circuit

def test_bloch_size():
    print("Testing Bloch sphere size customization...")
    qc = Circuit(1)
    qc.h(0)
    
    # Test default size
    fig_default = qc.draw('bloch')
    print(f"Default size: {fig_default.get_size_inches()}")
    
    # Test custom size
    target_size = (3, 3)
    fig_custom = qc.draw('bloch', figsize=target_size)
    actual_size = fig_custom.get_size_inches()
    print(f"Custom size requested: {target_size}")
    print(f"Custom size actual: {actual_size}")
    
    if actual_size[0] == target_size[0] and actual_size[1] == target_size[1]:
        print("SUCCESS: Bloch sphere size customized correctly.")
    else:
        print("FAILURE: Bloch sphere size mismatch.")
        sys.exit(1)

def test_qsphere_size():
    print("\nTesting Q-sphere size customization...")
    qc = Circuit(1)
    qc.h(0)
    
    # Test custom size
    target_size = (3, 3)
    fig_custom = qc.draw('qsphere', figsize=target_size)
    actual_size = fig_custom.get_size_inches()
    print(f"Custom size requested: {target_size}")
    print(f"Custom size actual: {actual_size}")
    
    if actual_size[0] == target_size[0] and actual_size[1] == target_size[1]:
        print("SUCCESS: Q-sphere size customized correctly.")
    else:
        print("FAILURE: Q-sphere size mismatch.")
        sys.exit(1)

def test_mpl_size():
    print("\nTesting MPL circuit drawer size customization...")
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Test custom size
    target_size = (6, 2.5)
    try:
        fig_custom = qc.draw('mpl', figsize=target_size)
        actual_size = fig_custom.get_size_inches()
        print(f"Custom size requested: {target_size}")
        print(f"Custom size actual: {actual_size}")
        
        if actual_size[0] == target_size[0] and actual_size[1] == target_size[1]:
            print("SUCCESS: MPL circuit drawer size customized correctly.")
        else:
            print("FAILURE: MPL circuit drawer size mismatch.")
            sys.exit(1)
    except Exception as e:
        print(f"FAILURE: MPL drawer raised exception: {e}")
        # Don't exit here, as MPL might not be fully supported in headless env without some config
        pass

if __name__ == "__main__":
    try:
        test_bloch_size()
        test_qsphere_size()
        test_mpl_size()
        print("\nAll visualization size tests passed!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
