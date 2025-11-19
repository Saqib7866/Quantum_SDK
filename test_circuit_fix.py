from python.qx import Circuit

# Test Circuit()
qc1 = Circuit()
print("Circuit() created:", type(qc1), hasattr(qc1, 'h'))

# Test Circuit(1)
qc2 = Circuit(1)
print("Circuit(1) created:", type(qc2), qc2._prog.n_qubits)

# Test allocate
qc1.allocate(1)
print("After allocate(1):", qc1._prog.n_qubits)

# Test h
qc1.h(0)
print("After h(0):", qc1._prog.n_qubits)