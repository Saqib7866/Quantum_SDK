from typing import List
from .core import Circuit
from .target import Target

class Pass:
    """
    Abstract base class for a transpiler pass.

    Each pass transforms a circuit for a given target and returns
    a new, optimized or modified circuit.

    Methods:
        run(circuit, target): Apply the pass to the given circuit.
    """
    def run(self, circuit: Circuit, target: Target) -> Circuit:

        raise NotImplementedError

class PassManager:
    """
    Executes a sequence of transpiler passes on a circuit.

    Attributes:
        passes (List[Pass]): Ordered list of transpiler passes.

    Methods:
        run(circuit, target): Apply all passes in order and return
        the final transformed circuit.
    """
    def __init__(self, passes: List[Pass]):
        self.passes = passes

    def run(self, circuit: Circuit, target: Target) -> Circuit:
        
        for p in self.passes:
            circuit = p.run(circuit, target)
        return circuit
