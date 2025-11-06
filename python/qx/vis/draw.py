"""
qx.vis.draw — circuit visualizer (text + optional matplotlib)
"""

from typing import Optional
from ..circuit import Circuit


def draw_text(circ: Circuit, max_width: int = 80) -> str:
    """Render circuit as ASCII art."""
    if not hasattr(circ, "program"):
        raise ValueError("draw_text expects a qx.Circuit object")

    prog = circ.program
    n = prog.n_qubits
    lines = [f"q{i}: ─" for i in range(n)]

    for op in prog.ops:
        if op.name == "measure":
            for q in op.qubits:
                lines[q] += "●──M"
        elif len(op.qubits) == 1:
            q = op.qubits[0]
            gate = op.name.upper()
            lines[q] += f"─[{gate}]─"
        elif len(op.qubits) == 2:
            c, t = op.qubits
            if op.name == "cx":
                lines[c] += "─■─"
                lines[t] += "─X─"
            else:
                lines[c] += f"─[{op.name}]─"
                lines[t] += f"─[{op.name}]─"
        else:
            for q in op.qubits:
                lines[q] += f"─({op.name})─"

    return "\n".join(lines)

# ---- optional matplotlib backend ----
def draw_matplotlib(circ: Circuit, ax: Optional["matplotlib.axes.Axes"] = None, save_path: Optional[str] = None):
    """Draw circuit diagram using matplotlib.
    - If display not available (e.g. WSL), saves as PNG automatically.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib not installed. run: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.7 * circ.program.n_qubits))
    ax.set_axis_off()

    y_positions = list(range(circ.program.n_qubits))
    for y in y_positions:
        ax.hlines(y, 0, len(circ.program.ops) + 1, color="black", lw=1)

    x = 1
    for op in circ.program.ops:
        if len(op.qubits) == 1:
            y = op.qubits[0]
            ax.text(x, y, op.name.upper(), ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        elif len(op.qubits) == 2:
            c, t = op.qubits
            ax.plot([x, x], [c, t], color="black", lw=1)
            ax.plot(x, c, "ko")
            ax.plot(x, t, "kx")
        x += 1

    ax.set_ylim(-1, circ.program.n_qubits)
    ax.set_xlim(0, x + 1)
    plt.tight_layout()

    # Decide save path or show
    if save_path is None:
        save_path = "circuit_diagram.png"

    try:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Circuit diagram saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Could not display or save plot: {e}")
    """Draw circuit diagram using matplotlib (if available)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib not installed. run: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.7 * circ.program.n_qubits))
    ax.set_axis_off()

    y_positions = list(range(circ.program.n_qubits))
    for y in y_positions:
        ax.hlines(y, 0, len(circ.program.ops) + 1, color="black", lw=1)

    x = 1
    for op in circ.program.ops:
        if len(op.qubits) == 1:
            y = op.qubits[0]
            ax.text(x, y, op.name.upper(), ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        elif len(op.qubits) == 2:
            c, t = op.qubits
            ax.plot([x, x], [c, t], color="black", lw=1)
            ax.plot(x, c, "ko")
            ax.plot(x, t, "kx")
        x += 1

    ax.set_ylim(-1, circ.program.n_qubits)
    ax.set_xlim(0, x + 1)
    plt.tight_layout()
    plt.show()
