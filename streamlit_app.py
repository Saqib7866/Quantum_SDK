import streamlit as st
import numpy as np

# Add the python directory to the path so we can import qx
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from python.qx import Circuit, backend, run
from python.qx.sim.local import LocalSimulator
from python.qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
import config

# Import matplotlib with Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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
    """Executes the user-provided quantum circuit code with enhanced result handling."""
    try:
        # Debug: Print the code being executed
        print("\n=== Executing Code ===")
        print(code)
        print("=====================\n")
        
        # Create a dictionary to store the local variables after execution
        local_vars = {
            'Circuit': Circuit,
            'run': run,
            'backend': backend,
            'algorithms': None,  # Will be set if available
            'np': np,
            'print': print  # Make print available in the executed code
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
                # First try to get qubits if they already exist
                if hasattr(circuit, 'qubits') and circuit.qubits is not None:
                    if len(circuit.qubits) >= num_qubits:
                        return circuit.qubits[:num_qubits]
                
                # If we can allocate new qubits, do that
                if hasattr(circuit, 'allocate_qubit'):
                    return [circuit.allocate_qubit() for _ in range(num_qubits)]
                elif hasattr(circuit, 'qubit'):
                    return [circuit.qubit(i) for i in range(num_qubits)]
                
                # If we can't allocate, just use indices
                return list(range(num_qubits))
            except Exception as e:
                st.warning(f"Warning in qubit allocation: {str(e)}")
                return list(range(num_qubits))  # Fallback to simple indices
        
        local_vars['allocate_qubits'] = allocate_qubits
        
        # Execute the user's code
        exec(code, globals(), local_vars)
        
        # Get the circuit object from the executed code
        qc = None
        print("=== Checking for Circuit objects ===")
        for name, var in local_vars.items():
            print(f"Variable {name}: type={type(var)}")
            # Check if it's a Circuit object by looking for the _prog attribute
            if hasattr(var, '_prog') and hasattr(var, 'program'):
                print(f"Found Circuit object in variable: {name}")
                print(f"Type: {type(var)}")
                print(f"Circuit program: {var.program}")
                print(f"Number of qubits: {var.program.n_qubits}")
                qc = var
                break
        print("===================================")
                
        if qc is None:
            print("Warning: No Circuit object found in the executed code")
            print("Available variables:", list(local_vars.keys()))
            st.error("No circuit object found in the provided code. Make sure to create a Circuit object (e.g., 'qc = Circuit(1)').")
            return None
            
        # Debug: Print circuit information
        print("\n=== Circuit Info ===")
        print(f"Circuit type: {type(qc)}")
        print(f"Circuit attributes: {[a for a in dir(qc) if not a.startswith('_')]}")
        if hasattr(qc, 'qubits'):
            print(f"Qubits: {qc.qubits}")
        if hasattr(qc, 'n_qubits'):
            print(f"Number of qubits (n_qubits): {qc.n_qubits}")
        if hasattr(qc, '_prog'):
            print(f"_prog type: {type(qc._prog)}")
            print(f"_prog attributes: {[a for a in dir(qc._prog) if not a.startswith('_')]}")
        print("==================\n")

        # Ensure the circuit has qubits
        if not hasattr(qc, 'qubits') or not qc.qubits:
            try:
                # Try to get the number of qubits from the circuit
                if hasattr(qc, '_prog') and hasattr(qc._prog, 'n_qubits'):
                    n_qubits = qc._prog.n_qubits
                    if n_qubits > 0:
                        qc.qubits = list(range(n_qubits))
                elif hasattr(qc, 'num_qubits'):
                    qc.qubits = list(range(qc.num_qubits))
            except:
                pass

        # Automatically add measurements if the user hasn't
        if hasattr(qc, '_prog') and hasattr(qc._prog, 'ops'):
            if not any(op.name == 'measure' for op in qc._prog.ops):
                try:
                    qubit_indices = list(range(len(qc.qubits) if hasattr(qc, 'qubits') else qc._prog.n_qubits))
                    if qubit_indices:
                        qc.measure(*qubit_indices)
                except Exception as e:
                    st.warning(f"Could not automatically add measurements: {e}")
        else:
            # Fallback for circuits without _prog.ops
            try:
                if not any(gate[0].name == 'measure' for gate in qc.program if hasattr(gate[0], 'name')):
                    qubit_indices = list(range(len(qc.qubits) if hasattr(qc, 'qubits') else 1))
                    if qubit_indices:
                        qc.measure(*qubit_indices)
            except Exception as e:
                st.warning(f"Could not check/apply measurements: {e}")

        # Use local simulator
        try:
            # Create and configure the simulator
            noise_cfg = noise_cfg or {}
            if backend_name == "zenaquantum-alpha":
                simulator = ZenaQuantumAlphaSimulator(noise=noise_cfg)
            else:
                simulator = LocalSimulator(noise=noise_cfg)
            
            # Debug: Print circuit info before execution
            print("\n=== Before Execution ===")
            print(f"Circuit type: {type(qc)}")
            print(f"Has _prog: {hasattr(qc, '_prog')}")
            if hasattr(qc, '_prog'):
                print(f"_prog type: {type(qc._prog)}")
                if hasattr(qc._prog, 'n_qubits'):
                    print(f"Number of qubits in _prog: {qc._prog.n_qubits}")
            print("======================\n")
            
            # Get the program to execute
            program = qc._prog if hasattr(qc, '_prog') else qc
            
            # Get the statevector before measurements
            statevector = simulator.get_statevector(program)

            # Execute the circuit
            counts, metadata = simulator.execute(program, shots=shots)

            # Create an enhanced result object
            class Result:
                def __init__(self, counts, metadata, circuit, statevector):
                    self._counts = counts
                    self.metadata = metadata or {}
                    self.circuit = circuit
                    self.statevector = statevector
                    # For backward compatibility
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
            
            # Ensure noise and backend information are present in metadata
            if "noise" not in metadata:
                metadata["noise"] = noise_cfg
            metadata.setdefault("backend", backend_name)
            metadata["n_qubits"] = qc._prog.n_qubits
            
            # Create the result object with the circuit and statevector attached
            result = Result(counts, metadata, qc, statevector)

            # Generate and attach the diagram
            result.diagram = generate_circuit_diagram(qc)
            
            return result
            
        except Exception as e:
            st.error(f"An error occurred during simulation: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
            
    except Exception as e:
        st.error(f"Execution Error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def generate_circuit_diagram(qc):
    """Generates a text-based diagram of the quantum circuit.
    
    This is a fallback implementation that's used when the new draw() method
    is not available. The new Circuit class provides a more sophisticated
    visualization through the draw() method.
    """
    # First try using the new draw() method if available
    if hasattr(qc, 'draw') and callable(getattr(qc, 'draw')):
        try:
            return qc.draw('text')
        except Exception:
            # Fall through to legacy implementation
            pass
            
    # Legacy implementation for backward compatibility
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
    """Displays the results of a quantum circuit execution with enhanced visualization."""
    st.subheader("Results")
    
    # Use tabs for different views
    tab_titles = ["📊 Histogram", "🔍 Circuit Diagram", "📋 Details"]
    
    # Add Analysis tab if we have the new CircuitAnalysis features
    if hasattr(result, 'circuit') and hasattr(result.circuit, 'analysis'):
        tab_titles.insert(1, "📈 Analysis")
    
    # Add visualization tabs for single-qubit circuits
    if hasattr(result, 'circuit'):
        # Debug: Show circuit attributes
        st.sidebar.subheader("Circuit Debug Info")
        st.sidebar.write("Circuit attributes:", [attr for attr in dir(result.circuit) if not attr.startswith('_')])
        
        # Get number of qubits using different possible attributes
        n_qubits = None
        if hasattr(result.circuit, 'n_qubits'):
            n_qubits = result.circuit.n_qubits
            st.sidebar.write("Using n_qubits:", n_qubits)
        elif hasattr(result.circuit, 'num_qubits'):
            n_qubits = result.circuit.num_qubits
            st.sidebar.write("Using num_qubits:", n_qubits)
        elif hasattr(result.circuit, 'qubits'):
            n_qubits = len(result.circuit.qubits)
            st.sidebar.write("Using len(qubits):", n_qubits)
        else:
            st.sidebar.warning("Could not determine number of qubits from circuit")
        
        # If we found the number of qubits and it's 1, add visualization tabs
        if n_qubits == 1:
            tab_titles.extend(["🌐 Q-Sphere", "🔄 Phase"])
            st.sidebar.success("Single qubit circuit detected - adding visualization tabs")
        else:
            st.sidebar.warning(f"Expected 1 qubit, found {n_qubits} - visualization tabs not added")
    
    tabs = st.tabs(tab_titles)
    
    # Extract common data
    counts = getattr(result, 'counts', {}) or getattr(result, 'get_counts', lambda: {})()
    total_shots = sum(counts.values()) if counts else 0
    probabilities = {s: (c / total_shots * 100) for s, c in counts.items()} if total_shots > 0 else {}
    n_qubits = getattr(result, 'n_qubits', 0) or len(next(iter(counts.keys()), '')) if counts else 0
    
    # Histogram Tab
    with tabs[0]:
        try:
            if probabilities and n_qubits > 0:
                all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
                probs = [probabilities.get(s, 0) for s in all_states]
                
                # Create a new figure for each plot
                import matplotlib.pyplot as plt
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                bars = ax.bar(all_states, probs, color="#4B8BBE", edgecolor='black')
                ax.set_ylabel("Probability (%)", fontsize=12)
                ax.set_xlabel("State", fontsize=12)
                ax.set_title("Measurement Probabilities", fontsize=14, weight='bold')
                ax.set_ylim(0, 110)  # Extra space for labels
                
                plt.xticks(rotation=90, fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                
                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 1:  # Only show labels for significant probabilities
                        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, 
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                
                st.pyplot(fig)
                plt.close(fig)

                # Add Bloch sphere for single-qubit circuits
                if n_qubits == 1:
                    try:
                        bloch_fig = result.circuit.draw('bloch')
                        if bloch_fig is not None:
                            st.subheader("Bloch Sphere")
                            st.pyplot(bloch_fig)
                            plt.close(bloch_fig)
                    except Exception as e:
                        st.warning(f"Could not generate Bloch sphere: {e}")
            else:
                st.warning("No probabilities to display.")
        except Exception as e:
            st.error(f"Error generating histogram: {str(e)}")
            st.warning("Please ensure all required dependencies are installed.")
            st.code("pip install matplotlib numpy")
    
    # Debug: Show tab titles and circuit info
    st.sidebar.subheader("Debug Info")
    st.sidebar.write("Available tabs:", tab_titles)
    
    if hasattr(result, 'circuit'):
        st.sidebar.write("Circuit has", getattr(result.circuit, 'n_qubits', 'unknown'), "qubits")
        st.sidebar.write("Circuit attributes:", [attr for attr in dir(result.circuit) if not attr.startswith('_')])
    
    # Q-Sphere Tab (for single qubit circuits)
    if hasattr(result, 'circuit') and "🌐 Q-Sphere" in tab_titles:
        with tabs[tab_titles.index("🌐 Q-Sphere")]:
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
    
    # Phase Visualization Tab (for single qubit circuits)
    if hasattr(result, 'circuit') and "🔄 Phase" in tab_titles:
        with tabs[tab_titles.index("🔄 Phase")]:
            st.subheader("Phase Visualization")
            try:
                import numpy as np
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                
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
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
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
    
    # Update tab indices for the rest of the function
    tab_offset = 1 if len(tabs) > 3 and hasattr(result, 'circuit') and hasattr(result.circuit, 'analysis') else 0
    
    # Analysis Tab (only if available)
    if len(tabs) > 3:  # We added the Analysis tab
        with tabs[1]:
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
                        
                        plt.style.use('seaborn-v0_8-whitegrid')
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
    
    # Circuit Diagram Tab
    tab_idx = 1 if len(tabs) <= 3 else 2
    with tabs[tab_idx]:
        # Try to get the diagram from the result or generate it
        diagram = getattr(result, 'diagram', None)
        if diagram is None and hasattr(result, 'circuit') and hasattr(result.circuit, 'draw'):
            try:
                diagram = result.circuit.draw('text')
            except Exception as e:
                st.warning(f"Could not generate circuit diagram: {e}")
                diagram = "No diagram available."
        
        if diagram and "Error" not in str(diagram) and "empty" not in str(diagram):
            st.code(diagram, language='text')
            st.download_button("Download Diagram", diagram, "circuit_diagram.txt")
            
            # Try to show a better visualization if available
            if hasattr(result, 'circuit') and hasattr(result.circuit, 'draw'):
                try:
                    st.subheader("Enhanced Visualization")
                    fig = result.circuit.draw('mpl')
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not generate enhanced visualization: {e}")
        else:
            st.warning("No circuit diagram available.")
    
    # Details Tab
    with tabs[-1]:  # Last tab is always Details
        if counts:
            st.subheader("Measurement Counts")
            st.table(counts)
        
        # Show metadata if available
        metadata = getattr(result, 'metadata', {}) or {}
        if metadata:
            st.subheader("Metadata")
            st.json(metadata)
        
        # Show circuit info if available
        if hasattr(result, 'circuit'):
            st.subheader("Circuit Information")
            c = result.circuit
            st.write(f"**Qubits:** {c.program.n_qubits}")
            st.write(f"**Classical Bits:** {c.program.n_clbits}")
            st.write(f"**Operations:** {len(c.program.ops)}")
            
            if hasattr(c, 'depth'):
                st.write(f"**Depth:** {c.depth()}")
            
            if hasattr(c, 'count_ops'):
                ops = c.count_ops()
                if ops:
                    st.write("**Operation Counts:**")
                    st.json(ops)

def main():
    """Main function to run the Streamlit app."""
    st.title("🔮 Quantum Circuit Visualizer")

    # Initialize session state
    if 'code' not in st.session_state:
        # Default circuit example - single qubit for Q-Sphere testing
        st.session_state.code = """# Simple single-qubit circuit for Q-Sphere testing
from qx import Circuit
import numpy as np

# Create a circuit with 1 qubit
qc = Circuit(1)

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

        # Create columns for buttons
        run_btn, clear_btn, test_btn = st.columns(3)
        
        if run_btn.button("▶️ Run Circuit", use_container_width=True, type="primary"):
            with st.spinner("Executing circuit..."):
                st.session_state.execution_result = execute_circuit(code, shots, noise_cfg, backend_name)
        
        if clear_btn.button("🗑️ Clear Output", use_container_width=True):
            st.session_state.execution_result = None
            
        if test_btn.button("🌐 Test Q-Sphere", use_container_width=True):
            test_circuit = """# Simple single-qubit circuit for Q-sphere testing
from qx import Circuit
import numpy as np

# Create and initialize circuit with exactly 1 qubit
qc = Circuit(1)  # This should create a circuit with 1 qubit

# Debug: Print circuit info
print("\n=== Circuit Info ===")
print("Number of qubits:", getattr(qc, 'n_qubits', 'unknown'))
if hasattr(qc, 'qubits'):
    print("Qubits:", qc.qubits)
print("==================\n")

# Apply a Hadamard gate to create superposition
qc.h(0)

# Optional: Uncomment to test different states
# qc.x(0)     # For |1⟩ state
# qc.s(0)     # Add phase
# qc.ry(np.pi/4, 0)  # Custom rotation

# Measure the qubit
qc.measure(0)"""
            st.session_state.code = test_circuit
            st.rerun()

        if st.session_state.execution_result:
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

# This ensures the main function is called when the script is run directly
if __name__ == "__main__":
    main()