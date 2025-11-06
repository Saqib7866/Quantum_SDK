from qx import Circuit, backend, run

with Circuit() as qc:
    a, b = qc.allocate(2)
    qc.h(a); qc.cx(a, b)
    qc.measure(a, b)

job = run(qc, backend("sim-local"), shots=2000)
res = job.result()
print("Counts:", res.counts)
print("Metadata:", res.metadata)
