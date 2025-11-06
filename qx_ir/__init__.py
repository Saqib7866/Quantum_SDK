"""
Quantum SDK - A quantum computing framework
"""

# Core classes
from .core import Circuit, Op, Program
from .target import Target
from .job import Job, JobStatus
from .backend import LocalBackend
from .simulator import StatevectorSimulator
from .storage import JobMetadata
from .batch_processor import BatchProcessor
from .visualization import CircuitDrawer
from .transpiler import PassManager
from .reports import get_circuit_depth, get_gate_counts, estimate_circuit_fidelity
from .estimator import ResourceEstimator, ResourceEstimate

# Version
__version__ = "0.5.0"

# Import CLI if available
try:
    from .cli.main import qx as cli
except ImportError:
    qx = None

# Define what gets imported with 'from qx_ir import *'
__all__ = [
    # Core components
    'Circuit', 'Op', 'Program',
    'Target',
    'Job', 'JobStatus',
    'LocalBackend',
    'StatevectorSimulator',
    'JobMetadata',
    'BatchProcessor',
    'CircuitDrawer',  # Visualization
    'PassManager',
    # Utility functions
    'get_circuit_depth', 'get_gate_counts', 'estimate_circuit_fidelity',
    'ResourceEstimator', 'ResourceEstimate',
    'qx'  # CLI command group if available
]

# Conditional CLI import
try:
    from .cli.main import qx as cli
    __all__.append('cli')
except ImportError:
    # Click not installed, CLI will not be available
    pass

# This makes the package executable via python -m qx_ir
if __name__ == '__main__':
    if 'cli' in globals():
        cli()
    else:
        print("Error: Click is required to use the command line interface.\n"
              "Install it with: pip install click")
        sys.exit(1)
