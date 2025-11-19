"""
Run the Bloch sphere visualization test with the correct Python path.
"""
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

# Now import and run the test
from test_bloch import test_bloch_visualization

if __name__ == "__main__":
    test_bloch_visualization()
