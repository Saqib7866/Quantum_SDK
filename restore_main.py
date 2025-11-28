
file_path = r"\\wsl.localhost\Ubuntu\home\saqib\projects\Quantum_SDK\streamlit_app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with Load Example button
cutoff_idx = -1
for i, line in enumerate(lines):
    if 'if st.button("🧪 **Load Example**", use_container_width=True):' in line:
        cutoff_idx = i
        break

if cutoff_idx != -1:

    
    # Keep lines up to cutoff (inclusive)
    new_lines = lines[:cutoff_idx+1]
    
    # Append the rest of main
    rest_of_main = r'''                test_circuit = """# Bell State Example
from qx import Circuit

# Create circuit
qc = Circuit()

# Allocate 2 qubits
qc.allocate(2)

# Create entangled Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
qc.h(0)      # Put qubit 0 in superposition
qc.cx(0, 1)  # Entangle with qubit 1

# Measure both qubits
qc.measure(0, 1)"""
                st.session_state.code = test_circuit
                st.rerun()

        # Results section - Moved here to be directly below actions
        if st.session_state.execution_result:
            st.markdown("---")
            display_results(st.session_state.execution_result)

    with settings_col:
        st.subheader("⚙️ Simulation Settings")

        # Backend selection
        backend_name = st.selectbox(
            "Backend",
            list(config.BACKENDS.keys()),
            index=list(config.BACKENDS.keys()).index(st.session_state.backend_name),
            help="Select the quantum backend simulator"
        )
        st.session_state.backend_name = backend_name

        # Shots
        shots = st.number_input(
            "Shots",
            min_value=1,
            max_value=10000,
            value=st.session_state.shots,
            step=100,
            help="Number of times to execute the circuit"
        )
        st.session_state.shots = shots

        # Noise Configuration
        st.markdown("---")
        st.markdown("**Noise Model**")
        
        # Only show noise options if backend supports it
        if config.BACKENDS[backend_name].get("noise_supported", False):
            col_a, col_b = st.columns(2)
            with col_a:
                ro = st.slider(
                    "Readout Error",
                    0.0, 0.2,
                    float(st.session_state.noise_cfg.get('readout_error', config.DEFAULT_NOISE_LEVEL)),
                    0.01,
                    help="Probability of measurement error"
                )
                e1 = st.slider(
                    "1-Qubit Gate Error",
                    0.0, 0.1,
                    float(st.session_state.noise_cfg.get('oneq_error', config.DEFAULT_NOISE_LEVEL)),
                    0.001,
                    help="Depolarizing error per single-qubit gate"
                )
            with col_b:
                e2 = st.slider(
                    "2-Qubit Gate Error",
                    0.0, 1.0,
                    float(st.session_state.noise_cfg.get('twoq_error', config.DEFAULT_NOISE_LEVEL)),
                    0.01,
                    help="Depolarizing error per two-qubit gate"
                )

            noise_cfg = {
                "readout_error": float(ro),
                "oneq_error": float(e1),
                "twoq_error": float(e2)
            }
            st.session_state.noise_cfg = noise_cfg

            # Show noise summary
            if any(noise_cfg.values()):
                st.info(f"🧪 Noise active: Readout {ro*100:.1f}%, 1Q {e1*100:.1f}%, 2Q {e2*100:.1f}%")
            else:
                st.success("✨ Perfect (noiseless) simulation")
        else:
            st.info("🎯 This backend supports noiseless simulation only")
            st.session_state.noise_cfg = {}

    with ref_col:
        st.subheader("📚 Quick Reference")

        # Getting Started Tips
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            **1.** Create a Circuit: `qc = Circuit(n_qubits)`\n
            **2.** Add gates: `qc.h(0)`, `qc.cx(0,1)`\n
            **3.** Measure: `qc.measure(0,1)`\n
            **4.** Run simulation above!
            """)

        with st.expander(" Single-Qubit Gates", expanded=False):
            st.markdown("""
            - `h(q)`: Hadamard (superposition)
            - `x(q)`: Pauli-X (NOT gate)
            - `y(q)`: Pauli-Y
            - `z(q)`: Pauli-Z
            - `s(q)`: S gate (√Z)
            - `t(q)`: T gate
            """)

        with st.expander("🔵 Two-Qubit Gates", expanded=False):
            st.markdown("""
            - `cx(c,t)`: CNOT gate
            - `cz(c,t)`: Controlled-Z
            - `swap(a,b)`: Swap qubits
            """)

        with st.expander("🟡 Rotations", expanded=False):
            st.markdown("""
            - `rx(θ,q)`: Rotate X-axis
            - `ry(θ,q)`: Rotate Y-axis
            - `rz(θ,q)`: Rotate Z-axis
            """)

        # Interactive Gate Builder
        st.markdown("---")
        with st.expander("🛠️ Interactive Gate Builder", expanded=False):
            gate_type = st.selectbox("Select Gate Type", ["Hadamard", "Pauli-X", "Pauli-Y", "Pauli-Z", "CNOT", "Rotation"])
            qubit = st.slider("Target Qubit", 0, 4, 0)

            if gate_type == "CNOT":
                control_qubit = st.slider("Control Qubit", 0, 4, 0)
                if control_qubit != qubit:
                    gate_code = f"qc.cx({control_qubit}, {qubit})"
                else:
                    gate_code = "# Invalid: Control and target qubits must be different"
            elif gate_type.startswith("Rotation"):
                angle = st.slider("Angle (radians)", 0.0, 2*3.14159, 3.14159/2)
                axis = st.selectbox("Axis", ["X", "Y", "Z"])
                gate_code = f"qc.r{axis.lower()}({angle:.2f}, {qubit})"
            else:
                gate_map = {"Hadamard": "h", "Pauli-X": "x", "Pauli-Y": "y", "Pauli-Z": "z"}
                gate_code = f"qc.{gate_map[gate_type]}({qubit})"

            st.code(gate_code)
            if st.button("➕ Add to Circuit"):
                if not st.session_state.code.endswith("\n"):
                    st.session_state.code += "\n"
                st.session_state.code += gate_code + "\n"
                st.rerun()

        # Tips section
        st.markdown("---")
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        - **Bell States**: Try `h(0); cx(0,1)` for entanglement
        - **Superposition**: Use `h(q)` for equal |0⟩+|1⟩ states
        - **Statistics**: More shots = better precision
        - **Noise**: Add realistic errors to simulate hardware
        """)


# This ensures the main function is called when the streamlit app is run
if __name__ == "__main__":
    main()
'''
    new_lines.append(rest_of_main)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("File restored successfully.")
else:
    print("Could not find cutoff line.")
