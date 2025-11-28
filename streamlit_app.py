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
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { 
    padding: 10px 16px; 
    border-radius: 8px; 
    background-color: #f0f2f6;
    border: none;
}
.stTabs [aria-selected="true"] { 
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
    """Generates a text-based diagram of the quantum circuit."""
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
    tab_titles = ["📊 Counts", "🔍 Circuit"]

    has_analysis = hasattr(result, 'circuit') and hasattr(result.circuit, 'analysis')
    if has_analysis:
        tab_titles.insert(1, "📈 Stats")

    n_qubits = result.metadata.get('n_qubits', 0)
    has_single_qubit_viz = (n_qubits == 1)
    if has_single_qubit_viz:
        tab_titles.extend(["🌐 Q-Sphere", "🔄 Phase"])

    tab_titles.append("📋 Info")

    return tab_titles

def _extract_common_data(result):
    """Extract commonly used data from result."""
    counts = getattr(result, 'counts', {}) or getattr(result, 'get_counts', lambda: {})()
    total_shots = sum(counts.values()) if counts else 0
    probabilities = {s: (c / total_shots) for s, c in counts.items()} if total_shots > 0 else {}
    n_qubits = getattr(result, 'n_qubits', 0) or len(next(iter(counts.keys()), '')) if counts else 0
    return counts, total_shots, probabilities, n_qubits

def _display_histogram_tab(result, probabilities, n_qubits, total_shots, has_single_qubit_viz):
    """Display the histogram tab with measurement results."""
    try:
        if not probabilities or n_qubits <= 0:
            st.warning("⚠️ No measurement data available to display.")
            return

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Measurement Probabilities")
            
            # Create histogram
            fig, ax = plt.subplots(figsize=(4, 2.5))
            
            # Sort states by binary value
            sorted_states = sorted(probabilities.keys(), key=lambda x: int(x, 2))
            probs = [probabilities[s] for s in sorted_states]
            labels = [f"|{s}⟩" for s in sorted_states]
            
            # Create bars with gradient-like colors
            bars = ax.bar(labels, probs, color='#6c5ce7', alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Customize chart
            ax.set_ylabel('Probability')
            ax.set_ylim(0, max(probs) * 1.2)  # Add headroom for labels
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig, use_container_width=False)
            
        with col2:
            if has_single_qubit_viz:
                st.subheader("⚛️ Bloch Sphere")
                try:
                    # Draw Bloch sphere with compact size
                    bloch_fig = result.circuit.draw('bloch', figsize=(2.5, 2.5))
                    st.pyplot(bloch_fig, use_container_width=False)
                except Exception as e:
                    st.error(f"Could not generate Bloch sphere: {str(e)}")
            else:
                st.info("ℹ️ Bloch sphere is available for single-qubit circuits.")
                
    except Exception as e:
        st.error(f"Error displaying histogram: {str(e)}")

def _display_analysis_tab(result):
    """Display circuit analysis metrics."""
    st.subheader("📈 Circuit Analysis")
    
    if not hasattr(result, 'circuit'):
        st.warning("No circuit information available.")
        return

    circuit = result.circuit
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Operation Distribution")
        if hasattr(circuit, 'count_ops'):
            ops = circuit.count_ops()
            if ops:
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.bar(ops.keys(), ops.values(), color='#00b894')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig, use_container_width=False)
            else:
                st.info("No operations to analyze.")
                
    with col2:
        st.markdown("#### Qubit Usage")
        # Simple qubit usage visualization
        n_qubits = circuit.program.n_qubits
        usage = [0] * n_qubits

def _display_circuit_diagram_tab(result):
    """Display the enhanced circuit diagram."""
    st.subheader("🔍 Circuit Diagram")
    if hasattr(result, 'circuit'):
        try:
            # Use the new MPL drawer with compact size
            mpl_fig = result.circuit.draw('mpl', figsize=(6, 2.5))
            st.pyplot(mpl_fig, use_container_width=False)
        except Exception as e:
            st.error(f"Could not generate circuit diagram: {e}")

def _display_qsphere_tab(result):
    """Display Q-Sphere visualization."""
    st.subheader("🌐 Q-Sphere Representation")
    try:
        qsphere_fig = result.circuit.draw('qsphere', figsize=(2.5, 2.5))
        st.pyplot(qsphere_fig, use_container_width=False)
    except Exception as e:
        st.error(f"Could not generate Q-Sphere: {e}")

def _display_phase_tab(result):
    """Display phase visualization."""
    st.subheader("🔄 Phase Visualization")
    try:
        # Check if statevector is available
        if not hasattr(result, 'statevector') or result.statevector is None:
            st.warning("Phase visualization requires statevector data.")
            return

        statevector = result.statevector
        # Ensure statevector is a numpy array
        if not isinstance(statevector, np.ndarray):
            statevector = np.array(statevector)
            
        n_states = len(statevector)
        n_qubits = result.n_qubits
        
        # Calculate amplitude and phase
        amplitudes = np.abs(statevector)
        phases = np.angle(statevector)
        
        # Filter out states with near-zero probability to keep plot clean
        threshold = 1e-6
        active_indices = [i for i, amp in enumerate(amplitudes) if amp > threshold]
        
        if not active_indices:
            st.info("State vector is zero.")
            return
            
        active_amplitudes = amplitudes[active_indices]
        active_phases = phases[active_indices]
        active_labels = [f"|{i:0{n_qubits}b}⟩" for i in active_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
        
        # Plot magnitudes
        x_pos = np.arange(len(active_indices))
        
        ax1.bar(x_pos, active_amplitudes, color='#e17055', alpha=0.8, edgecolor='black')
        ax1.set_title("Magnitude |α|")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(active_labels, rotation=45)
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Plot phases
        ax2.bar(x_pos, active_phases, color='#fdcb6e', alpha=0.8, edgecolor='black')
        ax2.set_title("Phase φ (rad)")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(active_labels, rotation=45)
        ax2.set_ylim(-np.pi, np.pi)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        
    except Exception as e:
        st.error(f"Could not generate phase visualization: {e}")

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
    if "📈 Stats" in tab_titles:
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
