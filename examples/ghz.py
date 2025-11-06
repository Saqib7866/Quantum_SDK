"""GHZ state example: prepare an n-qubit GHZ, run and visualize results."""
from qx import Circuit, backend, run, draw_text
from qx.vis import draw_matplotlib, plot_counts


def build_ghz(n=4):
    with Circuit() as qc:
        q = qc.allocate(n)
        qc.h(q[0])
        for i in range(n - 1):
            qc.cx(q[i], q[i + 1])
        qc.measure(*q)
    return qc


def main():
    qc = build_ghz(4)
    b = backend("sim-local")

    print("Circuit (ASCII):")
    print(draw_text(qc))
    try:
        draw_matplotlib(qc, save_path="ghz_circuit.png")
    except Exception as e:
        print("Could not draw matplotlib circuit:", e)

    job = run(qc, b, shots=256, seed=123)
    res = job.result()
    print("Counts:", res.counts)
    try:
        plot_counts(res.counts, title="GHZ counts", save_path="ghz_counts.png", show=False, n_qubits=4)
        print("Saved histogram to ghz_counts.png")
    except Exception as e:
        print("Could not save histogram:", e)


if __name__ == '__main__':
    main()
