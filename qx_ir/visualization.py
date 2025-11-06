"""
Circuit visualization module for QX-IR.
Provides both text-based and matplotlib-based circuit visualization.
"""
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from collections import Counter
import math

from .core import Circuit, Op

class CircuitDrawer:
    """Draw quantum circuits in various formats and visualize results."""
    
    @staticmethod
    def draw_text(circuit: Circuit, with_measures: bool = True) -> str:
        """Generate a text-based representation of the circuit.
        
        Args:
            circuit: The quantum circuit to draw
            with_measures: Whether to include measurement operations
            
        Returns:
            String representation of the circuit
        """
        if not circuit.instructions:
            return "Empty Circuit"
            
        # Get all qubits used in the circuit
        qubits = sorted({q for op in circuit.instructions for q in op.qubits})
        n_qubits = circuit.n_qubits
        
        # Initialize grid
        grid = [[' ' for _ in range(100)] for _ in range(2 * n_qubits - 1)]
        
        # Draw qubit lines
        for i in range(n_qubits):
            grid[2*i] = ['─' if x % 2 == 0 else ' ' for x in range(100)]
            
        # Add qubit labels
        for i in range(n_qubits):
            grid[2*i][0] = f'q{i}'
        
        # Add gates
        col = 4  # Start after qubit labels
        for op in circuit.instructions:
            if not with_measures and op.name.lower() == 'measure':
                continue
                
            # Find the column where this gate should be placed
            while any(2*q < len(grid) and col < len(grid[2*q]) and grid[2*q][col] != ' ' for q in op.qubits):
                col += 2
            
            # Draw the gate
            if len(op.qubits) == 1:  # Single-qubit gate
                q = op.qubits[0]
                if 2*q < len(grid) and col+1 < len(grid[2*q]):
                    grid[2*q][col] = '■'
                    grid[2*q][col+1] = op.name[0].upper()  # First letter of gate name
                
            else:  # Multi-qubit gate (CNOT, SWAP, etc.)
                min_q = min(op.qubits)
                max_q = max(op.qubits)
                
                # Draw control qubits and vertical lines
                for q in range(min_q, max_q + 1):
                    if 2*q < len(grid) and col < len(grid[2*q]) and q in op.qubits[1:]:  # Control qubits (except first)
                        grid[2*q][col] = '●'
                    
                    # Draw vertical line
                    if col < len(grid[0]):
                        for r in range(2*min_q + 1, 2*q):
                            if 0 <= r < len(grid) and r % 2 == 1:  # Only on odd rows (between qubits)
                                grid[r][col] = '│'
                
                # Draw the gate on the target qubit
                target_row = 2 * op.qubits[0]
                if target_row < len(grid) and col+1 < len(grid[target_row]):
                    grid[target_row][col] = '■'
                    grid[target_row][col+1] = op.name[0].upper()
                
                # For SWAP gate, draw X on target
                if op.name.lower() == 'swap':
                    for q in op.qubits[1:]:
                        if 2*q < len(grid) and col < len(grid[2*q]):
                            grid[2*q][col] = '×'
                
            col += 4  # Move to next gate position
        
        # Convert grid to string
        lines = []
        for row in grid:
            # Trim trailing spaces and join
            line = ''.join(row).rstrip()
            if any(c != ' ' for c in line):
                lines.append(line)
        
        return '\n'.join(lines)
    
    @staticmethod
    def plot_results(
        counts: Union[Dict[str, int], Dict[int, int], List[int]],
        title: str = "Measurement Results",
        xlabel: str = "Bitstring",
        ylabel: str = "Counts",
        color: str = '#4b6cb7',
        figsize: tuple = (10, 6),
        filename: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Plot measurement results as a histogram.
        
        Args:
            counts: Dictionary or list of measurement results. Can be:
                   - Dict[bitstring: str, count: int]
                   - Dict[integer: int, count: int]
                   - List of measured bitstrings (will be counted)
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            color: Bar color
            figsize: Figure size (width, height)
            filename: If provided, save the figure to this file
            show: Whether to display the figure
            
        Returns:
            The matplotlib Figure object if show=False, else None
        """
        # Convert input to consistent format
        if isinstance(counts, list):
            # Convert list of measurements to counts
            counts = Counter(counts)
        
        # Convert integer keys to strings if needed
        if counts and isinstance(next(iter(counts.keys())), int):
            max_qubits = max(counts.keys()).bit_length() if counts else 0
            counts = {f"{k:0{max_qubits}b}": v for k, v in counts.items()}
        
        if not counts:
            print("No measurement results to plot.")
            return None
            
        # Prepare data for plotting
        bitstrings = list(counts.keys())
        values = list(counts.values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars
        bars = ax.bar(bitstrings, values, color=color, alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom',
                   fontsize=9)
        
        # Customize plot
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if there are many qubits
        if len(bitstrings) > 4:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        # Show plot if requested
        if show:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
                if not filename:
                    # If we can't show and no filename was provided, save to a default file
                    default_filename = "quantum_results.png"
                    plt.savefig(default_filename, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {default_filename}")
        
        return fig if not show else None
    
    @classmethod
    def draw_mpl(
        cls, 
        circuit: Circuit, 
        with_measures: bool = True,
        style: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Draw the circuit using matplotlib.
        
        Args:
            circuit: The quantum circuit to draw
            with_measures: Whether to include measurement operations
            style: Dictionary of style parameters
            filename: If provided, save the figure to this file
            show: Whether to display the figure
            
        Returns:
            The matplotlib Figure object if show=False, else None
        """
        if not circuit.instructions:
            print("Empty Circuit")
            return None
            
        # Default style
        default_style = {
            'background_color': 'white',
            'qubit_line_color': 'black',
            'qubit_line_width': 1.5,
            'gate_color': '#1f77b4',  # Default blue
            'control_color': 'black',
            'target_color': '#ff7f0e',  # Orange
            'text_color': 'white',
            'measure_color': '#2ca02c',  # Green
            'swap_color': '#ff0000',  # Red
            'font_size': 12,
            'fig_width': 12,
            'fig_height': 6,
            'h_gap': 0.5,
            'v_gap': 0.5,
        }
        
        # Update with user style
        style = {**default_style, **(style or {})}
        
        # Get all qubits used in the circuit
        n_qubits = circuit.n_qubits
        qubits = list(range(n_qubits))
        
        # Calculate figure size
        width = style['fig_width']
        height = n_qubits * style['v_gap'] + 1
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_facecolor(style['background_color'])
        
        # Remove axes
        ax.axis('off')
        
        # Draw qubit lines
        for i in range(n_qubits):
            y = n_qubits - i - 1  # Draw from top to bottom
            ax.axhline(y=y, color=style['qubit_line_color'], 
                      linewidth=style['qubit_line_width'], zorder=0)
            ax.text(-0.5, y, f'q{i}', ha='right', va='center', 
                   fontsize=style['font_size'])
        
        # Draw gates
        x_pos = 0
        gate_width = 0.5
        
        for op in circuit.instructions:
            if not with_measures and op.name.lower() == 'measure':
                continue
                
            # Find the x position for this gate
            x_pos += 0.5  # Add some space
            
            if len(op.qubits) == 1:  # Single-qubit gate
                q = op.qubits[0]
                y = n_qubits - q - 1
                
                # Draw gate box
                rect = plt.Rectangle(
                    (x_pos - gate_width/2, y - 0.4), 
                    gate_width, 0.8,
                    facecolor=style['gate_color'],
                    edgecolor='black',
                    zorder=5
                )
                ax.add_patch(rect)
                
                # Add gate text
                ax.text(
                    x_pos, y, 
                    op.name.upper(),
                    color=style['text_color'],
                    ha='center',
                    va='center',
                    fontsize=style['font_size'] - 2,
                    fontweight='bold',
                    zorder=10
                )
                
                # For measurement, add a meter symbol
                if op.name.lower() == 'measure':
                    ax.plot(
                        [x_pos + gate_width/2 + 0.1, x_pos + gate_width/2 + 0.3],
                        [y, y],
                        color=style['measure_color'],
                        linewidth=2,
                        zorder=4
                    )
                    ax.plot(
                        [x_pos + gate_width/2 + 0.3],
                        [y],
                        'o',
                        color=style['measure_color'],
                        markersize=8,
                        zorder=5
                    )
                
            else:  # Multi-qubit gate
                min_q = min(op.qubits)
                max_q = max(op.qubits)
                
                # Draw vertical line connecting gates
                ax.plot(
                    [x_pos, x_pos],
                    [n_qubits - max_q - 1, n_qubits - min_q - 1],
                    color=style['qubit_line_color'],
                    linewidth=style['qubit_line_width'],
                    zorder=1
                )
                
                # Draw control qubits (circles)
                for q in op.qubits[1:]:  # All except the first qubit
                    y = n_qubits - q - 1
                    circle = plt.Circle(
                        (x_pos, y),
                        0.15,
                        facecolor=style['control_color'],
                        edgecolor='black',
                        zorder=5
                    )
                    ax.add_patch(circle)
                
                # Draw target qubit (box with symbol)
                y = n_qubits - op.qubits[0] - 1
                
                if op.name.lower() == 'swap':
                    # Draw X for SWAP gate
                    ax.plot(
                        [x_pos - 0.2, x_pos + 0.2],
                        [y + 0.2, y - 0.2],
                        color=style['swap_color'],
                        linewidth=2,
                        zorder=5
                    )
                    ax.plot(
                        [x_pos - 0.2, x_pos + 0.2],
                        [y - 0.2, y + 0.2],
                        color=style['swap_color'],
                        linewidth=2,
                        zorder=5
                    )
                else:
                    # Draw a box for other multi-qubit gates (like CNOT)
                    rect = plt.Rectangle(
                        (x_pos - 0.2, y - 0.2),
                        0.4, 0.4,
                        facecolor=style['target_color'],
                        edgecolor='black',
                        zorder=5
                    )
                    ax.add_patch(rect)
                    
                    # Add plus sign for CNOT
                    ax.plot(
                        [x_pos - 0.1, x_pos + 0.1],
                        [y, y],
                        color='white',
                        linewidth=1.5,
                        zorder=6
                    )
                    ax.plot(
                        [x_pos, x_pos],
                        [y - 0.1, y + 0.1],
                        color='white',
                        linewidth=1.5,
                        zorder=6
                    )
            
            x_pos += 1.0  # Move to next gate position
        
        # Set plot limits
        ax.set_xlim(-1, x_pos + 1)
        ax.set_ylim(-1, n_qubits)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Show or return figure
        if show:
            plt.show()
            return None
        else:
            return fig
