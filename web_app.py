import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from qx import Circuit, backend, run
from qx.sim.local import LocalSimulator
from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantum Circuit Visualizer",
    page_icon="🔮",
    layout="wide",
)

# --- Custom Styling ---
st.markdown("""<style>
body { font-family: 'Inter', sans-serif; }
.main { background-color: #f0f2f6; }
.stButton>button { 
    border: 2px solid #4B8BBE; 
    background-color: white; 
    color: #4B8BBE; 
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover { 
    background-color: #4B8BBE; 
    color: white; 
    border-color: #4B8BBE;
}
.stButton>button[kind="primary"] { 
    background-color: #4B8BBE; 
    color: white; 
}
.stButton>button[kind="primary"]:hover { 
    background-color: #3A6A9A; 
    border-color: #3A6A9A;
}
.stTabs [data-baseweb=\"tab-list\"] { gap: 24px; }
.stTabs [data-baseweb=\"tab\"] { 
    padding: 10px 16px; 
    border-radius: 8px; 
    background-color: #f0f2f6;
    border: none;
}
.stTabs [aria-selected=\"true\"] { 
    background-color: #D3E3FD; 
    color: #1967d2;
}
</style>""", unsafe_allow_html=True)

# --- Core Functions ---
def execute_circuit(code, shots, noise_cfg, backend_name):
    """Executes the user-provided quantum circuit code."""
    try:
        local_vars = {'Circuit': Circuit, 'run': run, 'backend': backend}
        exec(code, globals(), local_vars)
        
        circuit = next((v for v in local_vars.values() if isinstance(v, Circuit)), None)
        if not circuit:
            st.error("No circuit object found in the provided code.")
            return None

        # Automatically add measurements if the user hasn't
        if not any(op.name == 'measure' for op in circuit._prog.ops):
            qubit_indices = list(range(circuit._prog.n_qubits))
            if qubit_indices:
                circuit.measure(*qubit_indices)

        # Use local simulator
        try:
            # Create and configure the simulator
            noise_cfg = noise_cfg or {}
            if backend_name == "zenaquantum-alpha":
                simulator = ZenaQuantumAlphaSimulator(noise=noise_cfg)
            else:
                simulator = LocalSimulator(noise=noise_cfg)
            
            # Run the circuit directly using the simulator
            counts, metadata = simulator.execute(circuit._prog, shots=shots)
            
            # Create a result object with the expected interface
            class Result:
                def __init__(self, counts, metadata):
                    self._counts = counts
                    self.metadata = metadata or {}
                    # For backward compatibility
                    self.counts = counts
                def get_counts(self):
                    return self._counts
                    
            # Ensure noise and backend information are present in metadata for the Details tab
            if "noise" not in metadata:
                metadata["noise"] = noise_cfg
            metadata.setdefault("backend", backend_name)
            result = Result(counts, metadata)
            
        except Exception as e:
            st.error(f"An error occurred during simulation: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
            
        result.diagram = generate_circuit_diagram(circuit)
        return result
    except Exception as e:
        st.error(f"Execution Error: {e}")
        return None

def generate_circuit_diagram(qc):
    """Generates a text-based diagram of the quantum circuit."""
    if not hasattr(qc, '_prog') or not qc._prog.ops:
        return "Circuit is empty or not properly initialized."

    n_qubits = qc._prog.n_qubits
    wires = [f"q{i:<2}│" for i in range(n_qubits)]

    for op in qc._prog.ops:
        max_len = max(len(w) for w in wires)
        for i in range(n_qubits):
            if len(wires[i]) < max_len:
                wires[i] += '─' * (max_len - len(wires[i]))

        gate_name = op.name.upper()
        qubits = op.qubits
        gate_symbol = op.name

        box_len = 7
        if len(qubits) == 1:
            q = qubits[0]
            box = f"─[{gate_symbol}]─"
            box_len = len(box)
            wires[q] += box
            for i in range(n_qubits):
                if i != q:
                    wires[i] += '─' * box_len
        elif len(qubits) == 2:
            c, t = sorted(qubits)
            wires[t] += f"─({gate_symbol})─"
            wires[c] += "───●───"
            for i in range(c + 1, t):
                wires[i] += "───│───"
            for i in range(n_qubits):
                if i not in range(c, t + 1):
                    wires[i] += '─' * box_len
        elif len(qubits) >= 3:
            t = max(qubits)
            controls = [q for q in qubits if q != t]
            wires[t] += f"─({gate_symbol})─"
            for c in controls:
                wires[c] += "───●───"
            for i in range(min(qubits) + 1, max(qubits)):
                if i not in qubits:
                    wires[i] += "───│───"
            for i in range(n_qubits):
                if i not in qubits:
                    wires[i] += '─' * box_len

    max_len = max(len(w) for w in wires)
    for i in range(n_qubits):
        if len(wires[i]) < max_len:
            wires[i] += '─' * (max_len - len(wires[i]))

    return "\n".join(wires)

def display_results(result):
    """Displays the results of a quantum circuit execution."""
    st.subheader("Results")
    tab1, tab2, tab3 = st.tabs(["📊 Histogram", "🔍 Circuit Diagram", "📋 Details"])

    counts = result.counts
    total_shots = sum(counts.values())
    probabilities = {s: (c / total_shots * 100) for s, c in counts.items()}
    n_qubits = result.metadata.get('n_qubits', 0)

    with tab1:
        if probabilities and n_qubits > 0:
            all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
            probs = [probabilities.get(s, 0) for s in all_states]
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(all_states, probs, color="#4B8BBE", edgecolor='black')
            ax.set_ylabel("Probability (%)", fontsize=12)
            ax.set_xlabel("State", fontsize=12)
            ax.set_title("Measurement Probabilities", fontsize=14, weight='bold')
            ax.set_ylim(0, 100) # Set fixed y-axis from 0 to 100
            plt.xticks(rotation=90, fontsize=10)
            plt.yticks(fontsize=10)
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 1:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
            plt.close(fig) # Explicitly close the figure to free memory
        else:
            st.warning("No probabilities to display.")

    with tab2:
        diagram = getattr(result, 'diagram', "No diagram generated.")
        if "Error" not in diagram and "empty" not in diagram:
            st.code(diagram, language='text')
            st.download_button("Download Diagram", diagram, "circuit_diagram.txt")
        else:
            st.warning(diagram)

    with tab3:
        st.subheader("Measurement Counts")
        st.table(counts)
        st.subheader("Metadata")
        st.json(result.metadata)

def main():
    """Main function to run the Streamlit app."""
    st.title("🔮 Quantum Circuit Visualizer")

    if 'code' not in st.session_state:
        st.session_state.code = """from qx import Circuit

with Circuit() as qc:
    q = qc.allocate(2)
    qc.h(q[0])
    qc.cx(q[0], q[1])
"""
    if 'execution_result' not in st.session_state:
        st.session_state.execution_result = None

    # --- Layout --- 
    editor_col, ref_col = st.columns([2, 1])

    with editor_col:
        st.subheader("Quantum Circuit Editor")
        code = st.text_area(
            "Quantum Circuit Code",
            st.session_state.code,
            height=350,
            key="code_editor",
        )
        st.session_state.code = code

        with st.expander("⚙️ Simulation Settings", expanded=True):
            backend_name = st.selectbox("Select Backend", list(config.BACKENDS.keys()))
            st.caption(config.BACKENDS[backend_name]['description'])
            shots = st.slider("Number of Shots", 1, 10000, config.DEFAULT_SHOTS, 1)

            # Noise controls
            noise_cfg = {}
            if config.BACKENDS[backend_name]["noise_model"]:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    ro = st.slider("Readout error", 0.0, 1.0, float(config.DEFAULT_NOISE_LEVEL), 0.01, help="Probability a measured bit flips.")
                with col_b:
                    e1 = st.slider("1Q gate error", 0.0, 1.0, float(config.DEFAULT_NOISE_LEVEL), 0.01, help="Per 1-qubit gate depolarizing error.")
                with col_c:
                    e2 = st.slider("2Q gate error", 0.0, 1.0, float(config.DEFAULT_NOISE_LEVEL), 0.01, help="Per 2-qubit gate depolarizing error.")
                noise_cfg = {"readout_error": float(ro), "oneq_error": float(e1), "twoq_error": float(e2)}

        run_btn, clear_btn = st.columns(2)
        if run_btn.button("▶️ Run Circuit", use_container_width=True, type="primary"):
            with st.spinner("Executing circuit..."):
                st.session_state.execution_result = execute_circuit(code, shots, noise_cfg, backend_name)
        
        if clear_btn.button("🗑️ Clear Output", use_container_width=True):
            st.session_state.execution_result = None

        if st.session_state.execution_result:
            display_results(st.session_state.execution_result)

    with ref_col:
        st.subheader("Quick References")
        st.info("💡 Use `qc.` to access circuit methods. Example: `qc.h(0)`")
        gate_sets = {
            "Single-Qubit": "- `h(q)`: Hadamard\n- `x(q)`: Pauli-X (NOT)\n- `y(q)`: Pauli-Y\n- `z(q)`: Pauli-Z",
            "Phase": "- `s(q)`: Phase Gate\n- `sdg(q)`: S-dagger\n- `t(q)`: T Gate\n- `tdg(q)`: T-dagger",
            "Rotation": "- `rx(θ, q)`\n- `ry(θ, q)`\n- `rz(θ, q)`",
            "Two-Qubit": "- `cx(c, t)`: CNOT\n- `cz(c, t)`\n- `swap(q1, q2)`",
            "Three-Qubit": "- `ccx(c1, c2, t)`: Toffoli\n- `cswap(c, t1, t2)`: Fredkin"
        }
        for name, text in gate_sets.items():
            with st.expander(name):
                st.markdown(text)


if __name__ == "__main__":
    main()
