"""Quantum Fourier Transform example (n=3).

Builds the QFT circuit for n qubits, runs and visualizes results on a basis state.
"""
from math import pi
from qx import Circuit, backend, run, draw_text
from qx.vis import draw_matplotlib, plot_counts


def controlled_rz(qc, control, target, theta):
    # implement controlled-Rz via CX - Rz - CX (works up to global phase)
    qc.cx(control, target)
    qc.rz(target, theta)
    qc.cx(control, target)


def qft_circuit(n):
    with Circuit() as qc:
        q = qc.allocate(n)
        # assume input is prepared by caller; here we prepare |1> on LSB for demo
        qc.x(q[0])
        # QFT
        for j in range(n):
            qc.h(q[j])
            for k in range(1, n - j):
                theta = pi / (2 ** k)
                controlled_rz(qc, q[j + k], q[j], theta)
        # optional: swap qubits to reverse order (implemented with CX sequence)
        for i in range(n // 2):
            a = q[i]
            b = q[n - 1 - i]
            qc.cx(a, b)
            qc.cx(b, a)
            qc.cx(a, b)
        qc.measure(*q)
    return qc


def main():
    qc = qft_circuit(3)
    print("Circuit:")
    print(draw_text(qc))
    try:
        draw_matplotlib(qc, save_path="qft_circuit.png")
    except Exception as e:
        print("Could not draw circuit:", e)

    b = backend("sim-local")
    job = run(qc, b, shots=256, seed=11)
    res = job.result()
    print("Counts:", res.counts)
    try:
        plot_counts(res.counts, title="QFT counts", save_path="qft_counts.png", show=False, n_qubits=3)
        print("Saved histogram qft_counts.png")
    except Exception as e:
        print("Could not save histogram:", e)


if __name__ == '__main__':
    main()
