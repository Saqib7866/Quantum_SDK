from qx_ir.core import Circuit, Op, Program
from qx_ir.target import Target
from qx_ir.transpiler import PassManager
from qx_ir.passes import DecomposeUnsupportedGates, CheckQubitMapping
from qx_ir.simulator import StatevectorSimulator
from qx_ir.reports import get_circuit_depth, get_gate_counts, estimate_circuit_fidelity

def main():
    """Demonstrate the full Stage 3 hardware-aware compilation and execution pipeline."""

    # 1. Load the hardware profile
    try:
        target = Target.from_file('qxir_v1.json')
        print(f"✅ Loaded target profile: {target.name}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Create a circuit with a gate not native to the target (CCX)
    # This circuit creates a GHZ state using a Toffoli gate.
    # Update operations to use only qubits 0 and 1
    circuit = Circuit(n_qubits=2)
    circuit.add_op(Op('h', [0]))
    circuit.add_op(Op('cx', [0, 1]))

    # circuit = Circuit(n_qubits=3)
    # circuit.add_op(Op(name='h', qubits=[1])) # Hadamard gate(qubit ko superposition me dalta hai.)
    # circuit.add_op(Op(name='cx', qubits=[1, 2])) # CNOT (Controlled-X) gate – ek control aur ek target qubit use karta hai.
    # circuit.add_op(Op(name='ccx', qubits=[0, 1, 2])) # Toffoli gate (Controlled-Controlled-X) – do control aur ek target qubit use karta hai.
    print(f"\nOriginal circuit has {len(circuit.instructions)} operations.")

    # 3. Set up the transpilation pipeline
    # step-by-step har pass lagata hai, aur final transformed (clean + valid) circuit return karta hai.
    pass_manager = PassManager(passes=[
        DecomposeUnsupportedGates(),
        CheckQubitMapping(),
    ])
    print("\n🔧 Running transpiler...")

    # 4. Run the circuit through the transpiler
    try:
        transpiled_circuit = pass_manager.run(circuit, target)
        print("✅ Transpilation successful!")
        print(f"Transpiled circuit has {len(transpiled_circuit.instructions)} operations.")
    except ValueError as e:
        print(f"❌ Transpilation failed: {e}")
        return

    # 5. Generate and print reports for the transpiled circuit
    print("\n📊 Generating reports for transpiled circuit...")
    # Depth batata hai kitne steps me circuit chalega — jitni kam depth, utna fast aur accurate circuit hoga.
    # Ya line transpiled circuit ki total sequential gate layers (execution steps) calculate karti hai.(output e.g 13)
    depth = get_circuit_depth(transpiled_circuit)

    #Ya code circuit me 1-qubit aur 2-qubit gates kitni hain wo ginti hai — output deta hai dict jese: {'1q': 10, '2q': 7}.
    counts = get_gate_counts(transpiled_circuit)

    # Ya line har gate ki fidelity (accuracy) le kar sabko multiply karti hai —
    # taake pura circuit kitna accurate chalega wo estimate mile.
    # ➡️ Output ek 0–1 value hoti hai (jaise 0.92 = 92% overall reliability).

    fidelity = estimate_circuit_fidelity(transpiled_circuit, target)
    # In short: accuracy = hardware + circuit quality, shots se nahi badhti.
    print(f"  - Estimated Depth: {depth}")
    print(f"  - Gate Counts: {counts['1q']} (1Q), {counts['2q']} (2Q)")
    print(f"  - Estimated Fidelity: {fidelity:.4f}")

    # 6. Run the transpiled circuit on the simulator
    print("\n🔬 Running simulation...")
    # below main circuit + settings ko ek execution-ready package me convert karta hai taake simulator run kar sake
    program = Program(circuits=[transpiled_circuit], config={'shots': 5000})

    simulator = StatevectorSimulator() # simulator banata hai.
    result_counts = simulator.run(program) # circuit chala kar measurement results (counts) return

    print("\n✅ Simulation complete!")
    print("Measurement Counts:")
    # Sort by counts for readability
    # print("-----------------------------------------------------")
    sorted_counts = sorted(result_counts.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_counts)
    # print("-----------------------------------------------------")
    for outcome, count in sorted_counts:
        print(f"  - |{outcome}>: {count}")

if __name__ == "__main__":
    main()
