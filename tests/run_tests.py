import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Now import and run the tests
from test_simulators import main

if __name__ == "__main__":
    main()
