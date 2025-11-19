import streamlit as st
import numpy as np
import sys
import os

# Add the python directory to the path so we can import qx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from python.qx import Circuit, backend, run
from python.qx.sim.local import LocalSimulator
import python.qx as qx
import config

# Import matplotlib with Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set default style for all plots
plt.style.use('seaborn-v0_8-whitegrid')

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantum Circuit Visualizer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
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
def _setup_execution_environment():
    """Set up the execution environment for user code."""
    local_vars = {
        'Circuit': Circuit,
        'run': run,
        'backend': backend,
        'qx': qx,
        'algorithms': None,
        'np': np,
        'print': print
    }

    # Add algorithms if available
    try:
        from python.qx.algorithms import algorithms
        local_vars['algorithms'] = algorithms
    except ImportError:
        pass

    # Add qubit allocation helper
    def allocate_qubits(circuit, num_qubits):
        """Helper to allocate qubits in a way that works with the circuit."""
        try:
            if hasattr(circuit, 'qubits') and circuit.qubits and len(circuit.qubits) >= num_qubits:
                return circuit.qubits[:num_qubits]
            if hasattr(circuit, 'allocate_qubit'):
                return [circuit.allocate_qubit() for _ in range(num_qubits)]
            elif hasattr(circuit, 'qubit'):
                return [circuit.qubit(i) for i in range(num_qubits)]
            return list(range(num_qubits))
        except Exception as e:
            st.warning(f"Warning in qubit allocation: {str(e)}")
            return list(range(num_qubits))

    local_vars['allocate_qubits'] = allocate_qubits
    return local_vars

def _find_circuit_object(local_vars):
    """Find the Circuit object from executed variables."""
    # First try isinstance check with imported Circuit
    for var in local_vars.values():
        if isinstance(var, Circuit):
            return var

    # Fallback: check for objects that look like circuits (but not classes)
    for var_name, var in local_vars.items():
        if var_name.startswith('_'):  # Skip internal variables
            continue
        if not isinstance(var, type) and (hasattr(var, 'program') or hasattr(var, '_prog')):
            # Check if it has circuit-like attributes
            if hasattr(var, 'h') and callable(getattr(var, 'h', None)):
                return var

    return None

def _ensure_measurements(qc):
    """Ensure the circuit has measurements."""
    try:
        ops = qc._prog.ops if hasattr(qc, '_prog') else qc.program
        has_measure = any(op.name == 'measure' for op in ops)

        if not has_measure:
            n_qubits = len(qc.qubits) if hasattr(qc, 'qubits') else (qc._prog.n_qubits if hasattr(qc, '_prog') else 1)
            if n_qubits > 0:
                qc.measure(*range(n_qubits))
    except Exception as e:
        st.warning(f"Could not automatically add measurements: {e}")

def _create_result_object(counts, metadata, qc, statevector, backend_name, noise_cfg):
    """Create an enhanced result object."""
    class Result:
        def __init__(self, counts, metadata, circuit, statevector):
            self._counts = counts
            self.metadata = metadata or {}
            self.circuit = circuit
            self.statevector = statevector
            self.counts = counts
            self.n_qubits = circuit._prog.n_qubits

        def get_counts(self):
            return self._counts

        def get_statevector(self):
            return self.statevector

        def __str__(self):
            return f"Result with {len(self._counts)} outcomes"

        def __repr__(self):
            return f"<Result: {len(self._counts)} outcomes, {self.n_qubits} qubits>"

    # Ensure metadata completeness
    metadata = metadata or {}
    metadata.setdefault("noise", noise_cfg)
    metadata.setdefault("backend", backend_name)
    metadata["n_qubits"] = qc._prog.n_qubits

    result = Result(counts, metadata, qc, statevector)
    result.diagram = generate_circuit_diagram(qc)
    return result

def execute_circuit(code, shots, noise_cfg, backend_name):
    """Executes the user-provided quantum circuit code with enhanced result handling."""
    try:
        # Set up execution environment
        local_vars = _setup_execution_environment()

        # Execute the user's code
        exec(code, globals(), local_vars)

        # Find the Circuit object
        qc = _find_circuit_object(local_vars)
        if qc is None:
            available_vars = [name for name, var in local_vars.items() if not name.startswith('_')]
            st.error(f"No Circuit object found. Available variables: {available_vars}")
            st.info("Make sure to create a Circuit object (e.g., 'qc = Circuit(2)') and assign it to a variable.")
            return None

        # Ensure circuit has qubits and measurements
        if not hasattr(qc, 'qubits') or not qc.qubits:
            if hasattr(qc, 'program') and hasattr(qc.program, 'n_qubits'):
                qc.qubits = list(range(qc.program.n_qubits))
            elif hasattr(qc, 'num_qubits'):
                qc.qubits = list(range(qc.num_qubits))

        _ensure_measurements(qc)

        # Execute simulation
        noise_cfg = noise_cfg or {}
        simulator = LocalSimulator(noise=noise_cfg)
        program = qc.program if hasattr(qc, 'program') else qc

        statevector = simulator.get_statevector(program)
        counts, metadata = simulator.execute(program, shots=shots)

        return _create_result_object(counts, metadata, qc, statevector, backend_name, noise_cfg)

    except Exception as e:
        st.error(f"Execution Error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def generate_circuit_diagram(_qc):
    """Generates a text-based diagram of the quantum circuit.

    This is a fallback implementation that's used when the new draw() method
    is not available. The new Circuit class provides a more sophisticated
    visualization through the draw() method.
    """
    # First try using the new draw() method if available
    if hasattr(_qc, 'draw') and callable(getattr(_qc, 'draw')):
        try:
            return _qc.draw('text')
        except Exception:
            # Fall through to legacy implementation
            pass
            
    # Legacy implementation for backward compatibility
    if not hasattr(_qc, '_prog') or not _qc._prog.ops:
        return "Circuit is empty or not properly initialized."

    n_qubits = _qc._prog.n_qubits
    wires = [f"q{i:<2}│" for i in range(n_qubits)]

    for op in _qc._prog.ops:
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

def _display_execution_summary(result):
    """Display execution summary with key metrics."""
    st.header("🎯 Execution Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Shots", f"{result.metadata.get('shots', 'N/A'):,}")
    with col2:
        st.metric("Qubits", result.metadata.get('n_qubits', 'N/A'))
    with col3:
        st.metric("Backend", result.metadata.get('backend', 'N/A'))
    with col4:
        exec_time = result.metadata.get('execution_time', None)
        if exec_time:
            st.metric("Execution Time", f"{exec_time:.2f}s")
        else:
            total_counts = sum(result.counts.values())
            st.metric("Total Counts", f"{total_counts:,}")

def _get_available_tabs(result):
    """Determine which tabs should be available based on result properties."""
    tab_titles = ["📊 Histogram", "🔍 Circuit Diagram"]

    has_analysis = hasattr(result, 'circuit') and hasattr(result.circuit, 'analysis')
    if has_analysis:
        tab_titles.insert(1, "📈 Analysis")

    n_qubits = result.metadata.get('n_qubits', 0)
    has_single_qubit_viz = (n_qubits == 1)
    if has_single_qubit_viz:
        tab_titles.extend(["🌐 Q-Sphere", "🔄 Phase"])

    tab_titles.append("📋 Details")

    return tab_titles

def _extract_common_data(result):
    """Extract commonly used data from result."""
    counts = getattr(result, 'counts', {}) or getattr(result, 'get_counts', lambda: {})()
    total_shots = sum(counts.values()) if counts else 0
    probabilities = {s: (c / total_shots * 100) for s, c in counts.items()} if total_shots > 0 else {}
    n_qubits = getattr(result, 'n_qubits', 0) or len(next(iter(counts.keys()), '')) if counts else 0
    return counts, total_shots, probabilities, n_qubits

def display_results(result):
    """Displays the results of a quantum circuit execution with enhanced visualization."""
    _display_execution_summary(result)

    tab_titles = _get_available_tabs(result)
    tabs = st.tabs(tab_titles)

    counts, total_shots, probabilities, n_qubits = _extract_common_data(result)

    # Display each tab content
    tab_index = 0

    # Histogram tab (always first)
    with tabs[tab_index]:
        _display_histogram_tab(result, probabilities, n_qubits, total_shots, "🌐 Q-Sphere" in tab_titles)
    tab_index += 1

    # Analysis tab (if available)
    if "📈 Analysis" in tab_titles:
        with tabs[tab_index]:
            _display_analysis_tab(result)
        tab_index += 1

    # Circuit Diagram tab
    with tabs[tab_index]:
        _display_circuit_diagram_tab(result)
    tab_index += 1

    # Q-Sphere tab (for single qubit)
    if "🌐 Q-Sphere" in tab_titles:
        with tabs[tab_index]:
            _display_qsphere_tab(result)
        tab_index += 1

    # Phase tab (for single qubit)
    if "🔄 Phase" in tab_titles:
        with tabs[tab_index]:
            _display_phase_tab(result)
        tab_index += 1

    # Details tab (always last)
    with tabs[tab_index]:
        _display_details_tab(result, counts)

def _display_histogram_tab(result, probabilities, n_qubits, total_shots, has_single_qubit_viz):
    """Display the histogram tab with measurement results."""
    try:
        if not probabilities or n_qubits <= 0:
            st.warning("⚠️ No measurement data available to display.")
            return

        all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
        probs = [probabilities.get(s, 0) for s in all_states]

        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 4))

        colors = [plt.cm.Blues(0.3 + 0.7 * (p / max(probs) if max(probs) > 0 else 0)) for p in probs]
        bars = ax.bar(all_states, probs, color=colors, edgecolor='#2E5C8A', linewidth=1.5, alpha=0.8)

        ax.set_ylabel("Probability (%)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Quantum State", fontsize=12, fontweight='bold')
        ax.set_title("📊 Measurement Results", fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(110, max(probs) * 1.2))

        plt.xticks(rotation=0, fontsize=11, fontweight='bold')
        plt.yticks(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        for bar, prob in zip(bars, probs):
            if prob > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                        f'{prob:.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='#2E5C8A')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Top outcomes table
        st.markdown("### 🎯 Top Measurement Outcomes")
        sorted_outcomes = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)[:8]

        if sorted_outcomes:
            outcome_data = []
            for state, count in sorted_outcomes:
                prob = (count / total_shots) * 100
                outcome_data.append({
                    "State": f"|{state}⟩",
                    "Count": f"{count:,}",
                    "Probability": f"{prob:.2f}%"
                })
            st.table(outcome_data)

        # Bloch sphere for single-qubit
        if n_qubits == 1 and has_single_qubit_viz:
            st.markdown("---")
            try:
                bloch_fig = result.circuit.draw('bloch')
                if bloch_fig is not None:
                    st.subheader("🧲 Bloch Sphere Representation")
                    st.pyplot(bloch_fig)
                    plt.close(bloch_fig)
                    st.caption("*The Bloch sphere shows the quantum state's position in 3D space*")
            except Exception as e:
                st.warning(f"Could not generate Bloch sphere: {e}")

    except Exception as e:
        st.error(f"❌ Error generating histogram: {str(e)}")
        st.info("💡 Make sure matplotlib and numpy are installed: `pip install matplotlib numpy`")
    
    # Show circuit info in a clean way
    if hasattr(result, 'circuit'):
        with st.expander("ℹ️ Circuit Information", expanded=False):
            circuit = result.circuit
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Qubits:** {circuit.program.n_qubits}")
                st.write(f"**Operations:** {len(circuit.program.ops)}")
            with col2:
                st.write(f"**Depth:** {getattr(circuit, 'depth', lambda: 'N/A')()}")
                st.write(f"**Classical Bits:** {circuit.program.n_clbits}")

def _display_qsphere_tab(result):
    """Display the Q-Sphere visualization tab."""
    st.subheader("Qubit State on the Q-Sphere")
    try:
        # Use the built-in Q-sphere visualization
        qsphere_fig = result.circuit.draw('qsphere')
        if qsphere_fig is not None:
            st.pyplot(qsphere_fig)
            plt.close(qsphere_fig)
            st.markdown("""
            ### Understanding the Q-Sphere
            - The Q-sphere shows the quantum state's orientation in 3D space
            - |0⟩ is at the top, |1⟩ is at the bottom
            - The purple point shows the quantum state
            - The line from origin to point shows the state vector
            """)
        else:
            st.warning("Q-sphere visualization not available.")
    except Exception as e:
        st.error(f"Error generating Q-sphere: {str(e)}")
        st.warning("Q-sphere visualization requires matplotlib and mpl_toolkits.")
        st.code("pip install matplotlib numpy")
    
def _display_phase_tab(result):
    """Display the phase visualization tab."""
    st.subheader("Phase Visualization")
    try:
        import numpy as np

        # Get the state vector
        state = None
        if hasattr(result, 'get_statevector'):
            try:
                state = result.get_statevector()
            except Exception as e:
                st.warning(f"Could not get statevector: {str(e)}")

        if state is None:
            state = np.zeros(2**result.n_qubits, dtype=complex)
            state[0] = 1  # Default to |0> if no statevector available

        # Create a figure for the phase visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot amplitude
        ax1.bar([0, 1], np.abs(state) ** 2, color='skyblue')
        ax1.set_title('Probability Amplitudes')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['|0⟩', '|1⟩'])
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Probability')

        # Plot phase
        phases = np.angle(state)
        # Use scatter plot to make phases visible even when 0
        ax2.scatter([0, 1], phases, color='lightcoral', s=100, zorder=5)
        ax2.plot([0, 1], phases, 'o-', color='lightcoral', linewidth=2)
        # Add background bars for reference
        ax2.bar([0, 1], [2*np.pi]*2, bottom=-np.pi, color='lightgray', alpha=0.2, width=0.8)
        ax2.set_title('Phases (radians)')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['|0⟩', '|1⟩'])
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        ax2.set_ylabel('Phase (rad)')
        # Add phase value labels
        for i, phase in enumerate(phases):
            ax2.text(i, phase + 0.1 if phase >= 0 else phase - 0.1,
                    f'{phase:.2f}', ha='center',
                    va='bottom' if phase >= 0 else 'top', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Add some explanation
        st.markdown("""
        ### Understanding Phase Visualization
        - **Left plot**: Shows the probability of measuring |0⟩ and |1⟩
        - **Right plot**: Shows the phase (angle) of each basis state as points on a circle
        - Phase is measured in radians (-π to π)
        - The gray background bars show the full phase range for reference
        - A phase difference between |0⟩ and |1⟩ indicates quantum interference
        """)

    except Exception as e:
        st.error(f"Error generating phase visualization: {str(e)}")
        st.warning("""
        Phase visualization requires:
        - A single-qubit circuit
        - Matplotlib and NumPy installed
        """)
        st.code("pip install matplotlib numpy")
    
def _display_analysis_tab(result):
    """Display the circuit analysis tab."""
    if hasattr(result, 'circuit') and hasattr(result.circuit, 'analysis'):
        analysis = result.circuit.analysis

        # Circuit Depth
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Circuit Depth", analysis.depth())

        # Operation Counts
        with col2:
            op_counts = analysis.count_ops()
            st.metric("Total Operations", sum(op_counts.values()))

        # Operation Distribution
        if op_counts:
            st.subheader("Operation Distribution")
            try:
                # Import required libraries
                import numpy as np
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 2.5))

                n_bars = len(op_counts)
                bar_width = 0.8  # Wider bars with some spacing
                x = np.arange(n_bars)  # Evenly spaced bars

                # Calculate bar positions to center them
                x = x + (1 - bar_width) / 2

                bars = ax.bar(x, op_counts.values(),
                            width=bar_width,
                            color='#4B8BBE',
                            edgecolor='white',
                            linewidth=0.5,
                            align='edge')

                ax.set_ylabel("", fontsize=5)
                ax.set_title("Gate Counts", pad=1, fontsize=7, y=0.95)
                ax.set_xticks(x)
                ax.set_xticklabels(op_counts.keys(), rotation=45, ha='right', fontsize=5)

                # Set y-ticks and other properties
                ax.tick_params(axis='both', which='both', length=1, width=0.5, pad=1)

                # Remove spines
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # Set tighter margins and layout
                ax.margins(x=0.01, y=0.1)
                plt.yticks(fontsize=5)
                plt.tight_layout()

                # Display the plot
                st.pyplot(fig, use_container_width=True)  # Responsive
                plt.close(fig)

            except Exception as e:
                st.error(f"Error generating operation distribution: {str(e)}")
                st.warning("Please ensure matplotlib is properly installed.")
                st.code("pip install matplotlib numpy")

        # Qubit Usage
        qubit_usage = analysis.get_qubit_usage()
        if qubit_usage:
            st.subheader("Qubit Usage")
            fig, ax = plt.subplots(figsize=(8, 2.5))  # Increased height for better visibility
            n_bars = len(qubit_usage)
            # Match histogram style with proper bar spacing
            bar_width = 0.8  # Wider bars with some spacing
            import numpy as np
            x = np.arange(n_bars)  # Evenly spaced bars

            # Calculate bar positions to center them
            x = x + (1 - bar_width) / 2

            bars = ax.bar(x, qubit_usage.values(),
                        width=bar_width,
                        color='#4B8BBE',
                        edgecolor='white',
                        linewidth=0.5,
                        align='edge')

            ax.set_ylabel("", fontsize=5)
            ax.set_title("Qubit Usage", pad=1, fontsize=7, y=0.95)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{q}" for q in qubit_usage.keys()],
                              rotation=45, ha='right', fontsize=5)
            plt.yticks(fontsize=5)
            ax.tick_params(axis='both', which='both', length=1, width=0.5, pad=1)

            # Remove spines and adjust layout
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Set tighter margins and layout
            ax.margins(x=0.01, y=0.1)
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            st.pyplot(fig, use_container_width=True)  # Responsive
            plt.close(fig)
    else:
        st.info("Circuit analysis features not available for this result.")
    
def _display_circuit_diagram_tab(result):
    """Display the circuit diagram tab."""
    st.subheader("🔍 Circuit Diagram")

    # Try to get the diagram from the result or generate it
    diagram = getattr(result, 'diagram', None)
    if diagram is None and hasattr(result, 'circuit'):
        try:
            diagram = result.circuit.draw('text')
        except Exception as e:
            st.warning(f"⚠️ Could not generate circuit diagram: {e}")
            diagram = None

    if diagram and "Error" not in str(diagram) and "empty" not in str(diagram):
        # Display the ASCII diagram
        st.code(diagram, language='text')

        # Download button
        st.download_button(
            label="📥 Download Diagram",
            data=diagram,
            file_name="circuit_diagram.txt",
            mime="text/plain"
        )

        # Try to show matplotlib visualization
        try:
            mpl_fig = result.circuit.draw('mpl')
            if mpl_fig is not None:
                st.markdown("### 🎨 Enhanced Circuit Visualization")
                st.pyplot(mpl_fig)
                plt.close(mpl_fig)
        except Exception as e:
            st.info("ℹ️ Enhanced visualization not available for this circuit type")

    else:
        st.info("ℹ️ Circuit diagram not available. The circuit may be too complex or visualization is not supported.")
    
def _display_details_tab(result, counts):
    """Display the execution details tab."""
    st.subheader("📋 Execution Details")

    # Measurement Results
    if counts:
        st.markdown("### 📊 Raw Measurement Counts")
        # Display as a nice table
        count_data = [{"State": f"|{state}⟩", "Count": f"{count:,}"} for state, count in counts.items()]
        st.table(count_data)

    # Metadata Information
    metadata = getattr(result, 'metadata', {}) or {}
    if metadata:
        st.markdown("### ℹ️ Execution Metadata")
        # Format metadata nicely
        meta_display = {
            "Backend": metadata.get('backend', 'Unknown'),
            "Shots": f"{metadata.get('shots', 'N/A'):,}",
            "Qubits": metadata.get('n_qubits', 'N/A'),
            "Execution Time": "Real-time",
            "Noise Model": "Active" if any(metadata.get('noise', {}).values()) else "Disabled"
        }

        # Show noise details if active
        noise_info = metadata.get('noise', {})
        if any(noise_info.values()):
            meta_display["Readout Error"] = f"{noise_info.get('readout_error', 0)*100:.1f}%"
            meta_display["1Q Gate Error"] = f"{noise_info.get('oneq_error', 0)*100:.1f}%"
            meta_display["2Q Gate Error"] = f"{noise_info.get('twoq_error', 0)*100:.1f}%"

        st.json(meta_display)

    # Circuit Statistics
    if hasattr(result, 'circuit'):
        st.markdown("### 🔧 Circuit Statistics")
        circuit = result.circuit

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Qubits", circuit.program.n_qubits)
            st.metric("Classical Bits", circuit.program.n_clbits)
            st.metric("Circuit Depth", getattr(circuit, 'depth', lambda: 'N/A')())

        with col2:
            st.metric("Total Operations", len(circuit.program.ops))
            st.metric("Gate Types", len(set(op.name for op in circuit.program.ops)))

        # Operation breakdown
        if hasattr(circuit, 'count_ops'):
            ops = circuit.count_ops()
            if ops:
                st.markdown("#### Gate Usage Breakdown")
                # Create a nice visualization of gate counts
                gate_data = [{"Gate": gate, "Count": count} for gate, count in ops.items()]
                st.table(gate_data)

def main():
    """Main function to run the Streamlit app."""
    # Header with title and description
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔮 Quantum Circuit Visualizer")
        st.markdown("*Build, simulate, and visualize quantum circuits with real-time feedback*")
    with col2:
        st.markdown("""
        <div style='text-align: right; padding-top: 20px;'>
            <small style='color: #666;'>Powered by QX SDK</small>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state
    if 'code' not in st.session_state:
        # Default circuit example - single qubit for Q-Sphere testing
        st.session_state.code = """# Simple single-qubit circuit for Q-Sphere testing
from qx import Circuit
import numpy as np

# Create a circuit
qc = Circuit()

# Allocate 1 qubit
qc.allocate(1)

# Apply a Hadamard gate to create superposition
qc.h(0)

# Optional: Uncomment to test different states
# qc.x(0)     # For |1⟩ state
# qc.s(0)     # Add phase
# qc.ry(np.pi/4, 0)  # Custom rotation

# Measure the qubit
qc.measure(0)"""

    if 'execution_result' not in st.session_state:
        st.session_state.execution_result = None

    # --- Main Layout ---
    st.markdown("---")

    # Initialize default values
    if 'backend_name' not in st.session_state:
        st.session_state.backend_name = list(config.BACKENDS.keys())[0]
    if 'shots' not in st.session_state:
        st.session_state.shots = config.DEFAULT_SHOTS
    if 'noise_cfg' not in st.session_state:
        st.session_state.noise_cfg = {}

    # Create main content area with sidebar
    main_container = st.container()
    with main_container:
        # Create three columns: Editor | Settings | Reference
        editor_col, settings_col, ref_col = st.columns([2, 1.5, 1])

    with editor_col:
        st.subheader("📝 Quantum Circuit Editor")

        # Code editor with syntax highlighting hint
        st.markdown("**Write your quantum circuit code below:**")
        code = st.text_area(
            label="Circuit Code",
            value=st.session_state.code,
            height=300,
            key="code_editor",
            help="Create Circuit objects and apply quantum gates. Example: qc = Circuit(2); qc.h(0); qc.cx(0,1); qc.measure(0,1)",
            label_visibility="collapsed"
        )
        st.session_state.code = code

        # Action buttons with better layout
        st.markdown("### Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ **Run Circuit**", use_container_width=True, type="primary"):
                import time
                start_time = time.time()
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    with st.spinner("🔬 Simulating quantum circuit..."):
                        status_text.text("Setting up execution environment...")
                        progress_bar.progress(25)
                        status_text.text("Executing quantum circuit...")
                        progress_bar.progress(50)
                        st.session_state.execution_result = execute_circuit(code, st.session_state.shots, st.session_state.noise_cfg, st.session_state.backend_name)
                        progress_bar.progress(75)
                        status_text.text("Processing results...")
                        progress_bar.progress(100)
                        execution_time = time.time() - start_time
                        if st.session_state.execution_result:
                            st.success(f"✅ Circuit executed successfully in {execution_time:.2f}s!")
                            # Store execution time in result metadata
                            if hasattr(st.session_state.execution_result, 'metadata'):
                                st.session_state.execution_result.metadata['execution_time'] = execution_time
                            status_text.empty()
                            progress_bar.empty()
                        else:
                            st.error("❌ Circuit execution failed. Check your code and try again.")
                            status_text.empty()
                            progress_bar.empty()
                except Exception as e:
                    execution_time = time.time() - start_time
                    st.error(f"❌ Unexpected error during execution ({execution_time:.2f}s): {str(e)}")
                    st.info("💡 Check your circuit code for syntax errors or invalid operations.")
                    if 'progress_bar' in locals():
                        progress_bar.empty()
                    if 'status_text' in locals():
                        status_text.empty()

        with col2:
            if st.button("🗑️ **Clear Results**", use_container_width=True):
                st.session_state.execution_result = None
                st.rerun()

        with col3:
            if st.button("🧪 **Load Example**", use_container_width=True):
                test_circuit = """# Bell State Example
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

    with settings_col:
        st.subheader("⚙️ Simulation Settings")

        # Backend selection
        backend_name = st.selectbox(
            "Backend",
            list(config.BACKENDS.keys()),
            index=list(config.BACKENDS.keys()).index(st.session_state.backend_name),
            help="Choose the quantum simulator backend"
        )
        st.session_state.backend_name = backend_name
        st.caption(f"📋 {config.BACKENDS[backend_name]['description']}")

        # Shots slider with better explanation
        shots = st.slider(
            "Measurement Shots",
            min_value=100,
            max_value=10000,
            value=st.session_state.shots,
            step=100,
            help="Number of times to measure the circuit. More shots = better statistics but slower simulation."
        )
        st.session_state.shots = shots

        st.markdown("### Noise Configuration")
        st.caption("Add realistic noise to simulate quantum hardware imperfections")

        # Noise controls in a more organized way
        noise_cfg = {}
        if config.BACKENDS[backend_name]["noise_model"]:
            # Readout error
            ro = st.slider(
                "Readout Error",
                0.0, 1.0,
                float(st.session_state.noise_cfg.get('readout_error', config.DEFAULT_NOISE_LEVEL)),
                0.01,
                help="Probability that a measurement result is flipped"
            )

            # Gate errors
            col_a, col_b = st.columns(2)
            with col_a:
                e1 = st.slider(
                    "1-Qubit Gate Error",
                    0.0, 1.0,
                    float(st.session_state.noise_cfg.get('oneq_error', config.DEFAULT_NOISE_LEVEL)),
                    0.01,
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

        with st.expander("� Single-Qubit Gates", expanded=False):
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
        st.markdown("### 🛠️ Interactive Gate Builder")
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

    # Results section
    if st.session_state.execution_result:
        results_container = st.container()
        with results_container:
            st.markdown("---")
            display_results(st.session_state.execution_result)

    with ref_col:
        st.subheader("Quick References")
        st.info("💡 Use `qc.` to access circuit methods. Example: `qc.h(0)`")
        gate_sets = {
            "Single-Qubit": "- `h(q)`: Hadamard\n- `x(q)`: Pauli-X (NOT)\n- `sx(q)`: √X\n- `sxdg(q)`: √X†\n- `y(q)`: Pauli-Y\n- `z(q)`: Pauli-Z",
            "Phase": "- `s(q)`: Phase Gate\n- `sdg(q)`: S-dagger\n- `t(q)`: T Gate\n- `tdg(q)`: T-dagger\n- `p(θ, q)`: Phase shift",
            "Rotation": "- `rx(θ, q)`\n- `ry(θ, q)`\n- `rz(θ, q)`",
            "Two-Qubit": "- `cx(c, t)`: CNOT\n- `cy(c, t)`\n- `csx(c, t)`\n- `cp(θ, c, t)`\n- `cz(c, t)`\n- `swap(q1, q2)`\n- `iswap(q1, q2)`",
            "Three-Qubit": "- `ccx(c1, c2, t)`: Toffoli\n- `ccz(c1, c2, t)`\n- `cswap(c, t1, t2)`: Fredkin"
        }
        for name, text in gate_sets.items():
            with st.expander(name):
                st.markdown(text)

# This ensures the main function is called when the streamlit app is run
if __name__ == "__main__":
    main()