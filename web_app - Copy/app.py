from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import io
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from qx_ir import Circuit, Op, CircuitDrawer, Program
from circuit_visualizer import get_visualizations

app = Flask(__name__)

# Load the circuit configuration from JSON
def load_circuit_config():
    try:
        with open('qxir_v1.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        return {
            "n_qubits": 5,
            "basis_gates": ["id", "rz", "sx", "x", "cx", "h", "t", "tdg"],
            "gate_fidelities": {
                "id": 0.999, "rz": 0.999, "sx": 0.998, "x": 0.998,
                "cx": 0.99, "h": 0.998, "t": 0.999, "tdg": 0.999
            }
        }

# Load circuit configuration
circuit_config = load_circuit_config()

# Temporary storage for the last figure
last_figure = None

def create_ghz_circuit(n_qubits):
    """Create a GHZ state circuit with the given number of qubits."""
    circuit = Circuit(n_qubits)
    
    # Prepare GHZ state
    circuit.add_op(Op('h', qubits=[0]))
    for i in range(n_qubits - 1):
        circuit.add_op(Op('cx', qubits=[i, i + 1]))
    
    # Add some additional gates to make it more interesting
    if n_qubits > 1:
        circuit.add_op(Op('h', qubits=[n_qubits-1]))
    
    # Add measurements
    for i in range(n_qubits):
        circuit.add_op(Op('measure', qubits=[i]))
    
    return circuit

def create_qft_circuit(n_qubits):
    """Create a QFT-inspired circuit with the given number of qubits."""
    circuit = Circuit(n_qubits)
    
    # Create a more interesting pattern
    for i in range(n_qubits):
        # Apply Hadamard to create superposition
        circuit.add_op(Op('h', qubits=[i]))
        
        # Apply controlled-phase gates to create interference
        for j in range(i + 1, min(i + 3, n_qubits)):
            # Use a simple phase that creates an interesting pattern
            angle = np.pi / (2 ** (j - i + 1))
            circuit.add_op(Op('cp', qubits=[j, i], params=[angle]))
    
    # Add some final Hadamards to mix the states
    for i in range(n_qubits):
        if i % 2 == 0:  # Only on even qubits to create a pattern
            circuit.add_op(Op('h', qubits=[i]))
    
    # Add measurements
    for i in range(n_qubits):
        circuit.add_op(Op('measure', qubits=[i]))
    
    return circuit

def circuit_to_html(circuit):
    """Convert a circuit to an HTML representation."""
    try:
        if not hasattr(circuit, 'instructions') or not circuit.instructions:
            return "<div class='alert alert-info'>No operations in circuit</div>"
            
        # Create a simple HTML table for the circuit
        html_parts = ["<div class='card'><div class='card-body'><h5 class='card-title'>Quantum Circuit</h5>"]
        
        # Add qubit labels
        n_qubits = circuit.n_qubits
        html_parts.append("<div class='circuit-container' style='font-family: monospace; line-height: 1.8;'>")
        
        # Create a grid to represent the circuit
        max_ops = max(len(qubit_ops) for qubit_ops in circuit.instructions if qubit_ops)
        
        # Add header
        html_parts.append("<div class='d-flex mb-2' style='border-bottom: 1px solid #dee2e6; padding-bottom: 5px;'>")
        html_parts.append("<div style='width: 120px; font-weight: bold;'>Qubit</div>")
        for i in range(max_ops):
            html_parts.append(f"<div style='width: 100px; text-align: center; font-weight: bold;'>Op {i+1}</div>")
        html_parts.append("</div>")
        
        # Add qubit rows
        for q in range(n_qubits):
            qubit_ops = [op for op in circuit.instructions if q in op.qubits]
            html_parts.append(f"<div class='d-flex mb-2' style='border-bottom: 1px solid #f0f0f0; padding: 3px 0;'>")
            html_parts.append(f"<div style='width: 120px; font-weight: 500;'>Qubit {q}</div>")
            
            # Add operations
            for op in qubit_ops:
                op_name = op.name.upper()
                qubits = ','.join(map(str, op.qubits))
                params = f"({','.join(map(str, op.params))})" if hasattr(op, 'params') and op.params else ""
                html_parts.append(
                    f"<div class='op px-2 py-1 mx-1 rounded' "
                    f"style='background-color: #e9ecef; width: 100px; text-align: center;'>"
                    f"{op_name}{params} @{qubits}</div>"
                )
            
            # Fill empty spaces
            for _ in range(max_ops - len(qubit_ops)):
                html_parts.append("<div style='width: 100px;'></div>")
                
            html_parts.append("</div>")
        
        html_parts.append("</div>")  # Close circuit-container
        
        # Add measurement results if available
        if hasattr(circuit, 'counts') and circuit.counts:
            html_parts.append("<div class='mt-3'><strong>Measurements:</strong> ")
            html_parts.append(", ".join(f"{k}: {v}" for k, v in circuit.counts.items()))
            html_parts.append("</div>")
        
        html_parts.append("</div></div>")  # Close card-body and card
        
        # Add some basic CSS
        css = """
        <style>
            .op {
                transition: all 0.2s;
                font-size: 0.85em;
            }
            .op:hover {
                background-color: #d1e7ff !important;
                transform: translateY(-2px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .circuit-container {
                overflow-x: auto;
                padding: 10px;
                background: white;
                border-radius: 4px;
                border: 1px solid #dee2e6;
            }
        </style>
        """
        
        return css + "\n".join(html_parts)
        
    except Exception as e:
        error_msg = f"Error generating circuit HTML: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return f"""
        <div class='alert alert-danger'>
            <h5>Error Visualizing Circuit</h5>
            <p>{error_msg}</p>
            <pre style='font-size: 0.8em; max-height: 200px; overflow: auto;'>{traceback.format_exc()}</pre>
        </div>
        """

def create_results_plot(counts, n_qubits):
    """Create a Plotly bar plot of the measurement results."""
    try:
        if not counts:
            return "<div>No measurement results available</div>"
            
        # Convert counts to a format suitable for plotting
        if isinstance(counts, dict):
            # Sort the keys for consistent display
            sorted_items = sorted(counts.items())
            x = [str(k) for k, v in sorted_items]
            y = [v for k, v in sorted_items]
            
            # Calculate total shots for probability calculation
            total_shots = sum(y)
            probabilities = [v/total_shots for v in y]
            
            # Create hover text with both counts and probabilities
            hover_text = [f'State: {state}<br>Count: {count}<br>Probability: {prob:.2%}' 
                         for state, count, prob in zip(x, y, probabilities)]
        else:
            # If counts is a list or array, use it directly
            x = [str(i) for i in range(len(counts))]
            y = counts
            hover_text = [f'State: {state}<br>Count: {count}' for state, count in zip(x, y)]
        
        # Create the bar plot with improved styling
        fig = go.Figure(data=[
            go.Bar(
                x=x,
                y=y,
                text=y,
                textposition='auto',
                hovertext=hover_text,
                hoverinfo='text',
                marker=dict(
                    color='#4b6cb7',
                    line=dict(color='#1f2c56', width=1)
                ),
                opacity=0.8
            )
        ])
        
        # Calculate dynamic height based on number of qubits
        bar_height = max(30, 400 // (2 ** min(n_qubits, 4)))  # Limit height for large qubit counts
        height = len(x) * bar_height + 150
        
        # Update layout for better appearance
        fig.update_layout(
            title=dict(
                text='Measurement Results',
                x=0.5,
                xanchor='center',
                font=dict(size=18, family='Arial, sans-serif')
            ),
            xaxis=dict(
                title='Quantum State',
                title_font=dict(size=14),
                tickfont=dict(size=12),
                tickangle=45
            ),
            yaxis=dict(
                title='Counts',
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='#f0f0f0'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=60, r=40, t=80, b=100),
            height=min(600, height),  # Cap the maximum height
            width=None,  # Let it be responsive
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Arial'
            )
        )
        
        # Convert to HTML
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        print(f"Error creating results plot: {e}")
        return f"<div style='color: red;'>Error creating plot: {str(e)[:200]}</div>"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        n_qubits = int(request.form.get('n_qubits', 2))
        
        # Validate input
        if n_qubits < 1 or n_qubits > 5:
            return jsonify({
                'success': False,
                'error': 'Number of qubits must be between 1 and 5'
            })
        
        # Get all visualizations
        visualizations = get_visualizations(n_qubits)
        
        return jsonify({
            'success': True,
            **visualizations
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
