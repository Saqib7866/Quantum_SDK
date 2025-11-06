"""Bell pair example: create an EPR pair and verify correlations."""
from qx import Circuit, backend, run, draw_text
from qx.vis import draw_matplotlib, plot_counts


def build_bell():
    with Circuit() as qc:
        a, b = qc.allocate(2)
        qc.h(a)
        qc.cx(a, b)
        qc.measure(a, b)
    return qc


def main():
    qc = build_bell()
    print("Circuit:")
    print(draw_text(qc))
    try:
        draw_matplotlib(qc, save_path="bell_circuit.png")
    except Exception as e:
        print("Could not draw circuit:", e)

    b = backend("sim-local")
    job = run(qc, b, shots=512, seed=2025)
    res = job.result()
    print("Counts:", res.counts)
    try:
        plot_counts(res.counts, title="Bell counts", save_path="bell_counts.png", show=False, n_qubits=2)
        print("Saved histogram bell_counts.png")
    except Exception as e:
        print("Could not save histogram:", e)


if __name__ == '__main__':
    main()
