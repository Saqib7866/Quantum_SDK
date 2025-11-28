"""
Quantum circuit visualization.

This module provides functions for visualizing quantum circuits in various formats.
"""

from typing import Union, TYPE_CHECKING

# Configure matplotlib backend before any other imports
import os
import sys

# Set environment variable for matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'  # Must be set before importing matplotlib

# Import and configure matplotlib
import matplotlib as mpl
from matplotlib.figure import Figure
mpl.use('Agg')  # Set the backend before any other matplotlib imports

# Now import other required modules
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING, Any
import numpy as np
from math import sin, cos, pi

from .ir import Op

if TYPE_CHECKING:
    from .circuit import Circuit

class CircuitDrawer:
    """Class for drawing quantum circuits."""
    
    def __init__(self, circuit: 'Circuit', output: str = 'text', **kwargs):
        """Initialize the circuit drawer.
        
        Args:
            circuit: The quantum circuit to draw.
            output: Output format ('text', 'latex', 'mpl' for matplotlib).
            **kwargs: Additional drawing options.
        """
        self.circuit = circuit
        self.output = output
        self.kwargs = kwargs
        
    def draw(self) -> str:
        """Draw the circuit in the specified format.
        
        Returns:
            String representation of the circuit.
        """
        if self.output == 'text':
            return self._draw_text()
        elif self.output == 'latex':
            return self._draw_latex()
        elif self.output == 'mpl':
            return self._draw_mpl()
        elif self.output == 'bloch':
            return self._draw_bloch()
        elif self.output == 'qsphere':
            return self._draw_qsphere()
        else:
            raise ValueError(f"Unsupported output format: {self.output}")
    
    def _draw_text(self):
        """Draw the circuit as ASCII text."""
        # Initialize lines for each qubit
        lines = []
        n_qubits = self.circuit.program.n_qubits
        
        # Initialize each line with qubit label
        for i in range(n_qubits):
            lines.append([f"q[{i}]: "])
        
        # First pass: calculate the maximum position where we have gates
        max_gate_pos = 0
        for op in self.circuit.program.ops:
            if op.name == 'cx' and len(op.qubits) == 2:
                max_gate_pos += 3  # CNOT takes 3 characters (--• or --⊕)
            else:
                max_gate_pos += 4  # Other gates take 4 characters (--H-, --M-, etc)
        
        # Second pass: build the circuit with proper spacing
        gate_pos = 0
        for op in self.circuit.program.ops:
            if op.name == 'cx' and len(op.qubits) == 2:
                control, target = sorted(op.qubits)
                # For CNOT, we need to align the control and target
                for i in range(n_qubits):
                    if i == control:
                        lines[i].append("--•")
                    elif i == target:
                        lines[i].append("--⊕")
                    else:
                        # For qubits between control and target, add a vertical line
                        if min(control, target) < i < max(control, target):
                            lines[i].append("  |")
                        else:
                            lines[i].append("---")
                gate_pos += 3  # CNOT takes 3 characters
                
            else:
                # For other gates, add them with proper spacing
                gate = op.name.upper()
                gate_len = 4  # Default length for gates (e.g., "--H-")
                
                for i in range(n_qubits):
                    if i in op.qubits:
                        lines[i].append(f"--{gate}")
                        # Add padding to match the gate length
                        if gate == 'MEASURE':
                            lines[i].append("-" * (gate_len - 2 - len(gate)))
                        else:
                            lines[i].append("-" * (gate_len - 1 - len(gate)))
                    else:
                        # Add dashes for qubits not involved in this gate
                        lines[i].append("-" * gate_len)
                
                gate_pos += gate_len
        
        # Join all parts of each line, ensuring consistent length
        max_len = max(len(''.join(line)) for line in lines)
        formatted_lines = []
        for line in lines:
            line_str = ''.join(line)
            # Pad with dashes to make all lines the same length
            line_str = line_str.ljust(max_len, '-')
            formatted_lines.append(line_str)
            
        return "\n".join(formatted_lines)
    
    def _draw_latex(self) -> str:
        """Draw the circuit as LaTeX code."""
        # This is a simplified LaTeX output
        lines = ["\\begin{quantikz}"]
        
        for i in range(self.circuit.program.n_qubits):
            line = [f"& \\qw"]
            for op in self.circuit.program.ops:
                if i in op.qubits:
                    line.append(f"& \\gate{{{op.name.upper()}}}")
                else:
                    line.append("& \\qw")
            line.append("& \\qw \\\\")
            lines.append(" ".join(line))
            
        lines.append("\\end{quantikz}")
        return "\n".join(lines)
    
    def _draw_mpl(self):
        """Draw the circuit using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle

            n_qubits = self.circuit.program.n_qubits
            
            # Adjust figure size to be wider but more compact per gate
            if 'figsize' in self.kwargs:
                fig, ax = plt.subplots(figsize=self.kwargs['figsize'])
            else:
                fig_width = min(8, 2.0 + len(self.circuit.program.ops) * 0.4)  # Wider but more compact per gate
                fig_height = max(2.0, n_qubits * 0.7)  # Slightly less vertical spacing
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Draw qubit lines in reverse order (q0 at the top)
            for i in range(n_qubits):
                y = n_qubits - 1 - i  # Reverse the y-coordinate to put q0 at the top
                ax.axhline(y=y, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
                ax.text(-0.2, y, f'q[{i}]', ha='right', va='center', 
                       fontsize=8, fontweight='bold', fontfamily='monospace')

            # Draw gates with smaller sizes
            for i, op in enumerate(self.circuit.program.ops):
                x = i + 1
                
                if op.name == 'h':
                    for q in op.qubits:
                        y = n_qubits - 1 - q
                        ax.add_patch(Rectangle((x-0.15, y-0.25), 0.3, 0.5,  # More compact gates
                                            fill=True, color='lightblue',
                                            linewidth=0.8, edgecolor='black'))
                        ax.text(x, y, 'H', ha='center', va='center', 
                               fontsize=7, fontweight='bold')
                                
                elif op.name == 'x':
                    for q in op.qubits:
                        y = n_qubits - 1 - q
                        ax.add_patch(Circle((x, y), 0.15, fill=True, color='lightblue',
                                         linewidth=0.8, edgecolor='black'))
                        ax.text(x, y, 'X', ha='center', va='center', 
                               fontsize=7, fontweight='bold')
                                
                elif op.name == 'rx':
                    for q in op.qubits:
                        y = n_qubits - 1 - q
                        ax.add_patch(Rectangle((x-0.15, y-0.25), 0.3, 0.5,
                                            fill=True, color='lightgreen',
                                            linewidth=0.8, edgecolor='black'))
                        ax.text(x, y, f'RX\n{op.params[0]:.2f}' if hasattr(op, 'params') and op.params else 'RX', 
                               ha='center', va='center', 
                               fontsize=6, fontweight='bold')
                                
                elif op.name == 'ry':
                    for q in op.qubits:
                        y = n_qubits - 1 - q
                        ax.add_patch(Rectangle((x-0.15, y-0.25), 0.3, 0.5,
                                            fill=True, color='lightcoral',
                                            linewidth=0.8, edgecolor='black'))
                        ax.text(x, y, f'RY\n{op.params[0]:.2f}' if hasattr(op, 'params') and op.params else 'RY', 
                               ha='center', va='center', 
                               fontsize=6, fontweight='bold')
                                
                elif op.name == 'cx':
                    if len(op.qubits) == 2:
                        c, t = sorted(op.qubits)
                        y_c = n_qubits - 1 - c
                        y_t = n_qubits - 1 - t
                        # Control qubit (filled circle) - smaller
                        ax.add_patch(Circle((x, y_c), 0.08, fill=True, color='black'))
                        # Target qubit (circle with plus) - smaller
                        ax.add_patch(Circle((x, y_t), 0.12, fill=False, color='black',
                                         linewidth=1.0))
                        ax.text(x, y_t, '+', ha='center', va='center', 
                               fontsize=10, fontweight='bold')
                        # Vertical line connecting control and target - thinner
                        ax.plot([x, x], [y_c, y_t], 'k-', linewidth=1.0, alpha=0.7)
                        
                elif op.name == 'measure':
                    for q in op.qubits:
                        y = n_qubits - 1 - q
                        ax.add_patch(Rectangle((x-0.15, y-0.25), 0.3, 0.5,  # More compact gates
                                            fill=True, color='lightgreen',
                                            linewidth=0.8, edgecolor='black'))
                        ax.text(x, y, 'M', ha='center', va='center', 
                               fontsize=7, fontweight='bold')

            # Set axis limits with very tight horizontal spacing
            ax.set_xlim(0.7, len(self.circuit.program.ops) + 0.3)  # Even tighter x-limits
            ax.set_ylim(-0.5, n_qubits - 0.5)
            
            # Remove axis lines and ticks
            ax.axis('off')
            
            # Adjust layout with minimal padding all around
            plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.1)  # Very tight layout
            
            return fig
            
        except ImportError:
            return "Matplotlib is required for this visualization."
            
    def _draw_bloch(self):
        """Draw the circuit's final state on a Bloch sphere.

        Returns:
            Matplotlib figure showing the Bloch sphere.
        """
        try:
            # Import required modules with proper backend setup
            import os
            os.environ['MPLBACKEND'] = 'Agg'  # Must be set before importing matplotlib
            import matplotlib as mpl

            # Now import other modules
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.patches import FancyArrowPatch
            from mpl_toolkits.mplot3d import proj3d
            import numpy as np

            # Create figure with Agg backend
            figsize = self.kwargs.get('figsize', (8, 8))
            fig = plt.figure(figsize=figsize, dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            class Arrow3D(FancyArrowPatch):
                def __init__(self, xs, ys, zs, *args, **kwargs):
                    super().__init__((0, 0), (0, 0), *args, **kwargs)
                    self._verts3d = xs, ys, zs

                def do_3d_projection(self, renderer=None):
                    xs3d, ys3d, zs3d = self._verts3d
                    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                    return np.min(zs)

            # Initialize state as |0⟩
            state = np.array([1, 0], dtype=complex)

            # Apply all gates to the state
            for op in self.circuit.program.ops:
                if len(op.qubits) != 1:  # Only handle single-qubit gates for now
                    continue
                    
                if op.name == 'h':  # Hadamard
                    h = np.array([
                        [1, 1],
                        [1, -1]
                    ]) / np.sqrt(2)
                    state = h @ state
                    
                elif op.name == 'x':  # Pauli-X
                    x = np.array([
                        [0, 1],
                        [1, 0]
                    ])
                    state = x @ state
                    
                elif op.name == 'y':  # Pauli-Y
                    y = np.array([
                        [0, -1j],
                        [1j, 0]
                    ])
                    state = y @ state
                    
                elif op.name == 'z':  # Pauli-Z
                    z = np.array([
                        [1, 0],
                        [0, -1]
                    ])
                    state = z @ state
                    
                elif op.name == 's':  # Phase gate (√Z)
                    s = np.array([
                        [1, 0],
                        [0, 1j]
                    ])
                    state = s @ state
                    
                elif op.name == 't':  # T gate (π/8)
                    t = np.array([
                        [1, 0],
                        [0, np.exp(1j * np.pi/4)]
                    ])
                    state = t @ state
                    
                elif op.name == 'sx':  # Square root of X
                    sx = np.array([
                        [1 + 1j, 1 - 1j],
                        [1 - 1j, 1 + 1j]
                    ]) / 2
                    state = sx @ state
                    
                elif op.name == 'rx' and hasattr(op, 'params') and op.params:  # Rotation around X
                    theta = op.params[0]
                    rx = np.array([
                        [np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]
                    ])
                    state = rx @ state
                    
                elif op.name == 'ry' and hasattr(op, 'params') and op.params:  # Rotation around Y
                    theta = op.params[0]
                    ry = np.array([
                        [np.cos(theta/2), -np.sin(theta/2)],
                        [np.sin(theta/2), np.cos(theta/2)]
                    ])
                    state = ry @ state
                    
                elif (op.name == 'rz' or op.name == 'p') and hasattr(op, 'params') and op.params:  # Rotation around Z or Phase
                    phi = op.params[0]
                    rz = np.array([
                        [np.exp(-1j*phi/2), 0],
                        [0, np.exp(1j*phi/2)]
                    ])
                    state = rz @ state
                    
                elif op.name == 'u3' and hasattr(op, 'params') and len(op.params) >= 3:  # General unitary
                    theta, phi, lam = op.params[:3]
                    u3 = np.array([
                        [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                        [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
                    ])
                    state = u3 @ state
            
            # Normalize the state
            state = state / np.linalg.norm(state)
            
            # Convert state to Bloch vector
            alpha, beta = state
            x = 2 * (alpha.real * beta.real + alpha.imag * beta.imag)
            y = 2 * (alpha.real * beta.imag - alpha.imag * beta.real)
            z = abs(alpha)**2 - abs(beta)**2
            
            # Set up the 3D plot
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='w', 
                          rstride=1, cstride=1, alpha=0.3, linewidth=0.1)
            
            # Draw axes
            ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=1.5)
            ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', arrow_length_ratio=0.1, linewidth=1.5)
            ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', arrow_length_ratio=0.1, linewidth=1.5)
            
            # Add labels
            ax.text(1.6, 0, 0, 'X', fontsize=12, color='r')
            ax.text(0, 1.6, 0, 'Y', fontsize=12, color='g')
            ax.text(0, 0, 1.6, 'Z', fontsize=12, color='b')
            
            # Draw state vector
            arrow = Arrow3D([0, x], [0, y], [0, z], mutation_scale=20, 
                           lw=2, arrowstyle="-|>", color="purple")
            ax.add_artist(arrow)
            
            # Add state vector components
            ax.text(x/2, y/2, z/2, f"({x:.2f}, {y:.2f}, {z:.2f})", 
                   fontsize=10, color='purple')
            
            # Add state vector in Dirac notation
            alpha_str = f"{alpha.real:.2f}" if abs(alpha.imag) < 1e-10 else f"({alpha.real:.2f}{'+' if alpha.imag >= 0 else ''}{alpha.imag:.2f}j)"
            beta_str = f"{beta.real:.2f}" if abs(beta.imag) < 1e-10 else f"({beta.real:.2f}{'+' if beta.imag >= 0 else ''}{beta.imag:.2f}j)"
            state_str = f"|ψ⟩ = {alpha_str}|0⟩ + {beta_str}|1⟩"
            ax.text2D(0.05, 0.95, state_str, transform=ax.transAxes, 
                     fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8))
            
            # Set limits and title
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.set_title('Bloch Sphere')
            
            plt.tight_layout()
            
            # Return the figure object
            return fig
            
        except ImportError as e:
            import traceback
            error_msg = f"Error importing required visualization libraries: {e}\n"
            error_msg += f"Please make sure matplotlib and numpy are installed.\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            raise ImportError(error_msg)
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating Bloch sphere: {e}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)

    def _draw_qsphere(self):
        """Draw the quantum state on a Q-sphere.
        
        Returns:
            Matplotlib figure showing the Q-sphere.
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            import numpy as np
            
            # Initialize state vector (|0> state)
            state = np.array([1, 0], dtype=complex)

            # Apply gates from the circuit
            for op in self.circuit.program.ops:
                if len(op.qubits) != 1:  # Only handle single-qubit gates for now
                    continue
                    
                if op.name == 'x':  # Pauli-X
                    x = np.array([
                        [0, 1],
                        [1, 0]
                    ])
                    state = x @ state
                    
                elif op.name == 'y':  # Pauli-Y
                    y = np.array([
                        [0, -1j],
                        [1j, 0]
                    ])
                    state = y @ state
                    
                elif op.name == 'z':  # Pauli-Z
                    z = np.array([
                        [1, 0],
                        [0, -1]
                    ])
                    state = z @ state
                    
                elif op.name == 'h':  # Hadamard
                    h = np.array([
                        [1, 1],
                        [1, -1]
                    ]) / np.sqrt(2)
                    state = h @ state
                    
                elif op.name == 's':  # Phase gate (√Z)
                    s = np.array([
                        [1, 0],
                        [0, 1j]
                    ])
                    state = s @ state
                    
                elif op.name == 't':  # T gate (π/8)
                    t = np.array([
                        [1, 0],
                        [0, np.exp(1j * np.pi/4)]
                    ])
                    state = t @ state
                    
                elif op.name == 'sx':  # Square root of X
                    sx = np.array([
                        [1 + 1j, 1 - 1j],
                        [1 - 1j, 1 + 1j]
                    ]) / 2
                    state = sx @ state
                    
                elif op.name == 'rx' and hasattr(op, 'params') and op.params:  # Rotation around X
                    theta = op.params[0]
                    rx = np.array([
                        [np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]
                    ])
                    state = rx @ state
                    
                elif op.name == 'ry' and hasattr(op, 'params') and op.params:  # Rotation around Y
                    theta = op.params[0]
                    ry = np.array([
                        [np.cos(theta/2), -np.sin(theta/2)],
                        [np.sin(theta/2), np.cos(theta/2)]
                    ])
                    state = ry @ state
                    
                elif (op.name == 'rz' or op.name == 'p') and hasattr(op, 'params') and op.params:  # Rotation around Z or Phase
                    phi = op.params[0]
                    rz = np.array([
                        [np.exp(-1j*phi/2), 0],
                        [0, np.exp(1j*phi/2)]
                    ])
                    state = rz @ state
                    
                elif op.name == 'u3' and hasattr(op, 'params') and len(op.params) >= 3:  # General unitary
                    theta, phi, lam = op.params[:3]
                    u3 = np.array([
                        [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                        [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
                    ])
                    state = u3 @ state
            
            # Normalize the state
            state = state / np.linalg.norm(state)
            
            # Create figure and canvas
            dpi = 100
            figsize = self.kwargs.get('figsize', (8, 8))
            fig = Figure(figsize=figsize, dpi=dpi)
            FigureCanvas(fig)  # Create the canvas
            ax = fig.add_subplot(111, projection='3d')
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot the sphere surface
            ax.plot_surface(x, y, z, color='white', alpha=0.2, linewidth=0.1)
            
            # Draw |0⟩ and |1⟩ points
            ax.scatter([0], [0], [1], color='blue', s=100, label='|0⟩')
            ax.scatter([0], [0], [-1], color='red', s=100, label='|1⟩')
            
            # Get state components
            alpha, beta = state
            
            # Convert to spherical coordinates
            r = 1.0
            theta = 2 * np.arccos(abs(alpha))
            phi = np.angle(beta) - np.angle(alpha)
            
            # Calculate point on sphere
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            # Draw the state point
            ax.scatter([x], [y], [z], color='purple', s=200, label='State')
            
            # Draw line from origin to point
            ax.plot([0, x], [0, y], [0, z], 'k--', alpha=0.5)
            
            # Add labels
            ax.text(0, 0, 1.2, '|0⟩', fontsize=12, ha='center')
            ax.text(0, 0, -1.2, '|1⟩', fontsize=12, ha='center')
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Q-Sphere Visualization', pad=20)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Remove grid and ticks for cleaner look
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            # Add legend
            ax.legend()
            
            return fig
            
        except ImportError as e:
            import traceback
            error_msg = f"Error importing required visualization libraries: {e}\n"
            error_msg += f"Please make sure matplotlib and numpy are installed.\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            raise ImportError(error_msg)
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating Q-sphere: {e}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)
            

def draw(circuit: 'Circuit', output: str = 'text', **kwargs) -> Union[str, Figure]:
    """Draw a quantum circuit.

    Args:
        circuit: The quantum circuit to draw.
        output: Output format ('text', 'latex', 'mpl', 'bloch', 'qsphere').
        **kwargs: Additional drawing options.

    Returns:
        String or matplotlib figure depending on the output format.
    """
    drawer = CircuitDrawer(circuit, output, **kwargs)
    if output == 'bloch':
        return drawer._draw_bloch()
    if output == 'qsphere':
        return drawer._draw_qsphere()
    return drawer.draw()