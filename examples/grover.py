"""Simple Grover search example for 2-qubit database.

Searches for the marked element '11' using one Grover iteration.
"""
from qx import Circuit, backend, run, draw_text
from qx.vis import draw_matplotlib, plot_counts


def build_oracle_mark_11(qc, qubits):
    # phase flip on |11> via CZ implemented as H target, CX, H
    a, b = qubits
    qc.h(b)
    qc.cx(a, b)
    qc.h(b)


def diffusion(qc, qubits):
    # diffusion: H, X, multi-controlled Z, X, H
    a, b = qubits
    for q in qubits:
        qc.h(q)
        qc.x(q)

    # multi-controlled Z for 2 qubits -> use H on target and CX
    qc.h(b)
    qc.cx(a, b)
    qc.h(b)

    for q in qubits:
        qc.x(q)
        qc.h(q)


def main():
    with Circuit() as qc:
        q = qc.allocate(2)
        # prepare uniform superposition
        qc.h(q[0]); qc.h(q[1])

        # oracle marking '11'
        build_oracle_mark_11(qc, q)

        # diffusion (inversion about mean)
        diffusion(qc, q)

        qc.measure(*q)

    print("Circuit:")
    print(draw_text(qc))
    try:
        draw_matplotlib(qc, save_path="grover_circuit.png")
    except Exception as e:
        print("Could not draw circuit:", e)

    backend_desc = backend("sim-local")
    job = run(qc, backend_desc, shots=512, seed=7)
    res = job.result()
    print("Counts:", res.counts)
    try:
        plot_counts(res.counts, title="Grover (marked=11)", save_path="grover_counts.png", show=False, n_qubits=2)
        print("Saved histogram grover_counts.png")
    except Exception as e:
        print("Could not save histogram:", e)


if __name__ == '__main__':
    main()
