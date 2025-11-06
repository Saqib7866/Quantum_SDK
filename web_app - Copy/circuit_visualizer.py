import numpy as np
from qx_ir import Circuit, Op
from qx_ir.backend import StatevectorSimulator
import plotly.graph_objects as go

def create_ghz_circuit(n_qubits):
    """Create a GHZ state circuit."""
    circuit = Circuit(n_qubits)
    circuit.add_op(Op('h', [0]))
    for i in range(n_qubits - 1):
        circuit.add_op(Op('cx', [i, i + 1]))
    return circuit

def create_qft_circuit(n_qubits):
    """Create a QFT circuit."""
    circuit = Circuit(n_qubits)
    
    for i in range(n_qubits):
        circuit.add_op(Op('h', [i]))
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            circuit.add_op(Op('cp', [j, i], [angle]))
    
    # Add final swaps
    for i in range(n_qubits // 2):
        circuit.add_op(Op('swap', [i, n_qubits - 1 - i]))
    
    return circuit

def simulate_circuit(circuit, shots=1024):
    """Simulate the circuit and return counts."""
    simulator = StatevectorSimulator()
    result = simulator.run(circuit)
    return simulator.get_counts(result)

def plot_circuit(circuit):
    """Create a visualization of the circuit."""
    fig = go.Figure()
    
    n_qubits = circuit.n_qubits
    max_ops = max(len(op.qubits) for op in circuit.instructions) if circuit.instructions else 1
    
    # Add qubit lines
    for q in range(n_qubits):
        fig.add_trace(go.Scatter(
            x=[0, max_ops + 1],
            y=[q, q],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
    
    # Add gates
    for i, op in enumerate(circuit.instructions):
        for q in op.qubits:
            fig.add_trace(go.Scatter(
                x=[i + 1, i + 1],
                y=[q - 0.3, q + 0.3],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            fig.add_annotation(
                x=i + 1,
                y=q,
                text=op.name.upper(),
                showarrow=False,
                font=dict(size=12, color='white'),
                bgcolor='blue',
                bordercolor='blue',
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=150 + 50 * n_qubits,
        width=800
    )
    
    return fig

def plot_results(counts, title):
    """Create a bar plot of measurement results."""
    states = list(counts.keys())
    values = list(counts.values())
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=states,
        y=values,
        marker_color='#4b6cb7',
        text=values,
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='State',
        yaxis_title='Counts',
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=80),
        height=400,
        width=800
    )
    
    return fig

def get_visualizations(n_qubits):
    """Generate all visualizations for the given number of qubits."""
    # Create circuits
    ghz_circuit = create_ghz_circuit(n_qubits)
    qft_circuit = create_qft_circuit(n_qubits)
    
    # Simulate circuits
    ghz_counts = simulate_circuit(ghz_circuit)
    qft_counts = simulate_circuit(qft_circuit)
    
    # Create visualizations
    ghz_circuit_fig = plot_circuit(ghz_circuit)
    qft_circuit_fig = plot_circuit(qft_circuit)
    ghz_results_fig = plot_results(ghz_counts, 'GHZ State Measurement Results')
    qft_results_fig = plot_results(qft_counts, 'QFT Measurement Results')
    
    # Convert figures to HTML
    visualizations = {
        'ghz_circuit': ghz_circuit_fig.to_html(full_html=False, include_plotlyjs='cdn'),
        'ghz_results': ghz_results_fig.to_html(full_html=False, include_plotlyjs=False),
        'qft_circuit': qft_circuit_fig.to_html(full_html=False, include_plotlyjs=False),
        'qft_results': qft_results_fig.to_html(full_html=False, include_plotlyjs=False)
    }
    
    return visualizations
