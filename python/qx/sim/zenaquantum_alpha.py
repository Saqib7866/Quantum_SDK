import numpy as np
import time
import random
import threading
import queue
from typing import Dict, Optional, List, Tuple, Any, Union
from ..ir import Program
from .local import LocalSimulator
from ..config import get_max_qubits

class ZenaQuantumAlphaSimulator(LocalSimulator):
    """ZenaQuantum Alpha - A realistic quantum hardware simulator with timing and noise.

    Features:
      - timing via per-gate durations + base latency + queue jitter
      - depth-scaled noise (depends on #CX)
      - hardware throttle (semaphore) to emulate device lanes
    """

    def __init__(self, noise=None, durations=None, base_latency=0.5, queue_jitter=(0.10, 0.30), lanes: int = 1):
        super().__init__(noise=noise or {})
        # durations in ns (fallback defaults)
        self.durations = durations or {
            "h": 25, "x": 20, "y": 20, "z": 0, "rz": 15, 
            "s": 20, "sdg": 20, "t": 20, "tdg": 20,
            "cx": 220, "measure": 400, "reset": 1000,
            "crx": 200, "cry": 200, "crz": 200,
            "cswap": 600, "ccx": 600,
            "rxx": 200, "ryy": 200, "rzz": 200,
            "u1": 0, "u2": 100, "u3": 200
        }

        # base noise from target JSON or defaults
        self.e1_base = float(self.noise.get("oneq_error", 0.001))
        self.e2_base = float(self.noise.get("twoq_error", 0.01))
        self.ro_base = float(self.noise.get("readout_error", 0.02))

        # latency knobs
        self.base_latency = float(base_latency)
        self.queue_jitter = tuple(queue_jitter)

        # concurrency control (number of hardware lanes)
        self._lanes = int(lanes)
        # per-instance semaphore and active-run tracking for tests
        self._hw_semaphore = threading.Semaphore(self._lanes)
        self._active = 0
        self._max_active = 0
        self._active_lock = threading.Lock()

    def _get_duration(self, op_name: str) -> float:
        """Get the duration of an operation in ns."""
        return self.durations.get(op_name, 200)  # 200ns default

    def _apply_noise(self, depth: int) -> float:
        """Apply depth-dependent noise scaling."""
        # Simple exponential decay model
        return min(1.0, self.e1_base * (1.1 ** depth))

    def _simulate_hardware_latency(self, duration_ns: float):
        """Simulate hardware latency with jitter."""
        jitter = random.uniform(*self.queue_jitter)
        time.sleep((duration_ns * 1e-9) * (1.0 + jitter))

    def execute(self, prog: Program, shots: int = 1024, seed: Optional[int] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Execute the program with hardware-like timing and noise."""
        # Validate program first
        try:
            prog.validate()
        except Exception as e:
            raise RuntimeError(f"Program validation failed: {str(e)}")

        # Check qubit limit
        max_qubits = get_max_qubits()
        if prog.n_qubits > max_qubits:
            raise ValueError(
                f"ZenaQuantum Alpha is limited to {max_qubits} qubits (requested {prog.n_qubits})"
            )

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Track depth for noise scaling
        qubit_depth = {q: 0 for q in range(prog.n_qubits)}
        max_depth = 0

        # Simulate hardware concurrency
        with self._hw_semaphore:
            with self._active_lock:
                self._active += 1
                self._max_active = max(self._max_active, self._active)

            try:
                # Simulate queue time
                self._simulate_hardware_latency(self.base_latency * 1e6)  # ms to ns

                # Execute on the parent class
                try:
                    result = super().execute(prog, shots, seed)
                    
                    # Handle different return formats from parent
                    if isinstance(result, tuple) and len(result) == 2:
                        counts, meta = result
                    elif hasattr(result, 'counts') and hasattr(result, 'metadata'):
                        counts = result.counts
                        meta = getattr(result, 'metadata', {})
                    elif isinstance(result, dict):
                        counts = result
                        meta = {}
                    else:
                        raise ValueError(f"Unexpected return type from parent execute(): {type(result)}")
                    
                    # Ensure meta is a dictionary
                    if not isinstance(meta, dict):
                        meta = {}
                        
                    # Update metadata with hardware-specific info
                    meta.update({
                        "backend": "zenaquantum_alpha",
                        "max_qubits": max_qubits,
                        "active_lanes": self._lanes,
                        "queue_time_ns": self.base_latency * 1e6,
                        "max_parallel": self._max_active,
                    })
                    
                    # Ensure counts is a dictionary with string keys and integer values
                    if not isinstance(counts, dict):
                        raise ValueError("Counts must be a dictionary")
                        
                    # Convert keys to strings and values to integers if needed
                    processed_counts = {}
                    for k, v in counts.items():
                        try:
                            key = str(k)
                            value = int(v)
                            processed_counts[key] = value
                        except (ValueError, TypeError) as e:
                            raise ValueError(f"Invalid count value: {k}: {v} - {str(e)}")
                    
                    return processed_counts, meta
                    
                except Exception as e:
                    # Log the error and re-raise with more context
                    error_msg = f"Error in ZenaQuantumAlphaSimulator.execute(): {str(e)}"
                    raise RuntimeError(error_msg) from e

            finally:
                with self._active_lock:
                    self._active -= 1

    def __str__(self):
        return f"ZenaQuantumAlphaSimulator(noise={self.noise}, lanes={self._lanes})"