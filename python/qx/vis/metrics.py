"""
qx.vis.metrics — basic resource estimation utilities.
Compute gate counts, depth, and expected fidelity.
"""

from typing import Dict, Any
from ..ir import Program

def estimate_resources(prog: Program, caps: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Estimate circuit resources.
    Returns dict with:
      - depth
      - 1Q / 2Q gate counts
      - total gates
      - estimated fidelity (if caps provided)
    """
    if not hasattr(prog, "ops"):
        raise ValueError("estimate_resources expects a Program object")

    oneq_gates = {"h","x","y","z","rx","ry","rz"}
    twoq_gates = {"cx","cz","swap"}

    c1 = c2 = 0
    depth = 0
    last_used = {}

    for op in prog.ops:
        qset = set(op.qubits)
        if len(qset) == 1 and op.name in oneq_gates:
            c1 += 1
        elif len(qset) == 2 and op.name in twoq_gates:
            c2 += 1

        # simple depth estimation: each qubit timeline
        step = 1 + max((last_used.get(q,0) for q in qset), default=0)
        for q in qset:
            last_used[q] = step
        depth = max(depth, step)

    total = c1 + c2

    fidelity = None
    if caps:
        e1 = caps.get("oneq_error", 0.0)
        e2 = caps.get("twoq_error", 0.0)
        ro = caps.get("readout_error", 0.0)
        fidelity = (1-e1)**c1 * (1-e2)**c2 * (1-ro)

    return {
        "depth": depth,
        "gates_1q": c1,
        "gates_2q": c2,
        "total_gates": total,
        "estimated_fidelity": fidelity,
    }
