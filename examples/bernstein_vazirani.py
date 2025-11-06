"""Bernstein-Vazirani example using qx.Circuit.

This builds an n-bit BV oracle for a secret string `s`, runs it on the
local simulator, saves a circuit diagram and a histogram of results.
"""
from qx import Circuit, backend, run, draw_text
from qx.vis import draw_matplotlib, plot_counts


def build_bv_circuit(secret: str):
    n = len(secret)
    with Circuit() as qc:
        inputs = qc.allocate(n)
        anc = qc.allocate(1)

        # prepare inputs in |+> and ancilla in |->
        for q in inputs:
            qc.h(q)
        qc.x(anc[0])
        qc.h(anc[0])

        # oracle: for each bit of secret, CX from input to ancilla if bit==1
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                qc.cx(inputs[i], anc[0])

        # final Hadamards on inputs
        for q in inputs:
            qc.h(q)

        # measure inputs
        qc.measure(*inputs)

    return qc


def main():
    secret = "1011"
    qc = build_bv_circuit(secret)
    b = backend("sim-local")

    # save ASCII and matplotlib circuit diagram
    print("Circuit (ASCII):")
    print(draw_text(qc))
    try:
        draw_matplotlib(qc, save_path="bv_circuit.png")
    except Exception as e:
        print("Could not draw matplotlib circuit:", e)

    # run the circuit deterministically
    job = run(qc, b, shots=64, seed=42)
    res = job.result()
    print("Metadata:", res.metadata)
    print("Counts:", res.counts)

    # save histogram
    try:
        plot_counts(res.counts, title=f"BV results (secret={secret})", save_path="bv_counts.png", show=False, normalize=False, n_qubits=4)
        print("Saved histogram to bv_counts.png")
    except Exception as e:
        print("Could not save histogram:", e)


if __name__ == '__main__':
    main()
