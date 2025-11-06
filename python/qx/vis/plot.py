"""
qx.vis.plot — result visualization utilities (histogram for counts)
"""

from typing import Dict, Optional
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_counts(counts: Dict[str, int],
                title: str = "Measurement Results",
                save_path: Optional[str] = None,
                show: bool = True,
                normalize: bool = False,
                n_qubits: Optional[int] = None,
                theme: str = "light"):
    """Plot histogram of measurement counts or probabilities."""
    if not counts:
        import logging
        logging.getLogger("qx.vis").warning("No counts to plot.")
        return
    # If caller provides n_qubits, ensure we include all possible bitstrings
    if n_qubits is not None:
        # generate all bitstrings of length n_qubits in lexicographic order
        from itertools import product
        all_labels = ["".join(bits) for bits in ("".join(p) for p in product('01', repeat=n_qubits))]
        # product returns tuples; the above double-join is awkward; rebuild properly
        all_labels = ["".join(p) for p in product('01', repeat=n_qubits)]
        labels = all_labels
        values = [counts.get(lbl, 0) for lbl in labels]
    else:
        labels, values = zip(*sorted(counts.items()))
    total = sum(values)
    data = np.array(values, dtype=float)
    if normalize:
        data /= total

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, data,
                  color="steelblue", edgecolor="black", width=0.8)

    # annotate bars
    for bar, val in zip(bars, data):
        txt = f"{val:.2%}" if normalize else f"{int(val)}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                txt, ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Bitstring Outcome")
    ax.set_ylabel("Probability" if normalize else "Counts")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = "result_histogram.png"
    try:
        plt.savefig(save_path, bbox_inches="tight")
        import logging
        logging.getLogger("qx.vis").info("Saved histogram → %s", os.path.abspath(save_path))
    except Exception as e:
        import logging
        logging.getLogger("qx.vis").warning("Could not save histogram: %s", e)
    if theme=="dark":
        ax.set_facecolor("#111")
        fig.patch.set_facecolor("#111")
        for spine in ax.spines.values(): spine.set_color("#ccc")
        ax.tick_params(colors="#ccc"); ax.title.set_color("#ccc"); ax.yaxis.label.set_color("#ccc"); ax.xaxis.label.set_color("#ccc")
    if show:
        plt.show()
    else:
        plt.close(fig)
