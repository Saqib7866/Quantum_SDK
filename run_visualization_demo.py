import matplotlib.pyplot as plt
from qx_ir import Circuit, Op, CircuitDrawer

def create_quantum_teleportation() -> Circuit:
    """Create a quantum teleportation circuit."""
    circuit = Circuit(n_qubits=3)
    
    # Prepare the state to teleport (qubit 0)
    circuit.add_op(Op('h', qubits=[0]))
    circuit.add_op(Op('x', qubits=[0]))
    
    # Create Bell pair (qubits 1 and 2)
    circuit.add_op(Op('h', qubits=[1]))
    circuit.add_op(Op('cx', qubits=[1, 2]))
    
    # Bell measurement
    circuit.add_op(Op('cx', qubits=[0, 1]))
    circuit.add_op(Op('h', qubits=[0]))
    
    # Add measurement operations
    circuit.add_op(Op('measure', qubits=[0]))
    circuit.add_op(Op('measure', qubits=[1]))
    
    # Note: The actual conditional operations would be handled by the quantum processor
    # For visualization purposes, we'll add the operations without conditions
    circuit.add_op(Op('x', qubits=[2]))  # X gate on qubit 2
    circuit.add_op(Op('z', qubits=[2]))  # Z gate on qubit 2
    
    return circuit

def create_quantum_fourier_transform(n_qubits: int = 3) -> Circuit:
    """Create a quantum Fourier transform circuit."""
    circuit = Circuit(n_qubits=n_qubits)
    
    for i in range(n_qubits):
        circuit.add_op(Op('h', qubits=[i]))
        for j in range(i + 1, n_qubits):
            # Add controlled phase gates
            angle = 2 * 3.14159 / (2 ** (j - i + 1))
            circuit.add_op(Op('cp', qubits=[j, i], params=[angle]))
    
    # Swap qubits to match QFT definition
    for i in range(n_qubits // 2):
        circuit.add_op(Op('swap', qubits=[i, n_qubits - 1 - i]))
    
    return circuit

def main():
    """Run the visualization demo."""
    # Set up matplotlib to use a non-interactive backend if needed
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    print("🚀 Quantum Circuit Visualization Demo\n")
    # print("🔍 Note: Running in non-interactive mode. Visualizations will be saved as PNG files.\n")
    
    # Create some example circuits
    print("1️⃣  Simple Bell State Circuit")
    bell_circuit = Circuit(n_qubits=2)
    bell_circuit.add_op(Op('h', qubits=[0]))
    bell_circuit.add_op(Op('cx', qubits=[0, 1]))
    
    # Text visualization
    # print("\nText Representation:")
    # print(CircuitDrawer.draw_text(bell_circuit))
    
    # Save Bell state circuit visualization
    bell_png = "bell_circuit.png"
    CircuitDrawer.draw_mpl(bell_circuit, filename=bell_png, show=False)
    
    # Simulate the Bell state circuit and get measurement results
    from qx_ir.simulator import StatevectorSimulator
    from qx_ir import Program
    
    # Create a simulator and program
    simulator = StatevectorSimulator()
    program = Program(circuits=[bell_circuit], config={'shots': 1000})
    
    # Run the simulation
    bell_results = simulator.run(program)
    
    # Plot the results
    CircuitDrawer.plot_results(
        counts=bell_results,
        title="Bell State Measurement Results (Simulated)",
        xlabel="Quantum State",
        ylabel="Counts",
        color='#4b6cb7',
        filename="bell_state_results.png"
    )
    CircuitDrawer.draw_mpl(bell_circuit, filename=bell_png, show=False)
    
    # Quantum Teleportation
    # print("\n2️⃣  Quantum Teleportation Circuit")
    # teleportation = create_quantum_teleportation()
    
    # # print("\nText Representation:")
    # # print(CircuitDrawer.draw_text(teleportation))
    
    # # Save to file with custom style
    # teleport_png = "teleportation_circuit.png"
    # style = {
    #     'fig_width': 12,
    #     'fig_height': 6,
    #     'gate_color': '#4b6cb7',  # Blue
    #     'target_color': '#dd5e89',  # Pink
    #     'measure_color': '#56ab2f',  # Green
    #     'background_color': '#f8f9fa',
    #     'font_size': 10
    # }
    # # print(f"\nSaving teleportation circuit to '{teleport_png}'")
    # CircuitDrawer.draw_mpl(
    #     teleportation, 
    #     filename=teleport_png,
    #     style=style,
    #     show=False
    # )
    
    # Quantum Fourier Transform
    print("\n3️⃣  Quantum Fourier Transform (3-qubit)")
    qft_circuit = create_quantum_fourier_transform(3)
    
    # print("\nText Representation:")
    # print(CircuitDrawer.draw_text(qft_circuit))
    
    # Custom style for QFT
    qft_png = "qft_circuit.png"
    qft_style = {
        'fig_width': 14,
        'fig_height': 6,
        'gate_color': '#8e44ad',  # Purple
        'target_color': '#e74c3c',  # Red
        'control_color': '#3498db',  # Blue
        'background_color': '#f8f9fa',
        'font_size': 10
    }
    
    # Save QFT circuit visualization
    print(f"\nSaving QFT circuit to '{qft_png}'")
    CircuitDrawer.draw_mpl(
        qft_circuit,
        filename=qft_png,
        style=qft_style,
        show=False
    )
    
    # Simulate the QFT circuit with |100> input
    qft_input = Circuit(n_qubits=3)
    qft_input.add_op(Op('x', qubits=[0]))  # Prepare |100> state
    
    # Create a new circuit and add operations from both circuits
    full_qft_circuit = Circuit(n_qubits=3)
    
    # Add operations from qft_input
    for op in qft_input.instructions:
        full_qft_circuit.add_op(op)
    
    # Add operations from qft_circuit
    for op in qft_circuit.instructions:
        full_qft_circuit.add_op(op)
    
    # Create program with the combined circuit
    qft_program = Program(circuits=[full_qft_circuit], config={'shots': 1000})
    
    # Run the simulation
    qft_results = simulator.run(qft_program)
    
    # Plot the results
    CircuitDrawer.plot_results(
        counts=qft_results,
        title="3-Qubit QFT Measurement Results (Simulated |100> input)",
        xlabel="Quantum State",
        ylabel="Counts",
        color='#8e44ad',
        filename="qft_results.png"
    )
    
    print("\n✅ Demo complete!")
    print("💾 Circuit visualizations and result histograms have been saved as PNG files.")
    print("   - bell_circuit.png: Bell state circuit diagram")
    print("   - bell_state_results.png: Bell state measurement histogram")
    print("   - qft_circuit.png: QFT circuit diagram")
    print("   - qft_results.png: QFT measurement histogram")

if __name__ == "__main__":
    main()
