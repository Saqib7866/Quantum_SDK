from typing import Dict
import json, os
from .sim.local import LocalSimulator
from .errors import QXBackendError
from .config import get_max_qubits

# Only using LocalSimulator
ZenaQuantumAlphaSimulator = None

# default registry: use config.get_max_qubits() for the reported cap
_REG: Dict[str, dict] = {
    "sim-local": {
        "name": "sim-local",
        "type": "sim",
        "caps": {
            "n_qubits_max": get_max_qubits(),
            "native_gates": {
                # Single-qubit gates
                "h", "x", "y", "z", "s", "sdg", "t", "tdg",
                # Rotation gates
                "rx", "ry", "rz", "u1", "u2", "u3",
                # Two-qubit gates
                "cx", "cz", "swap", "crx", "cry", "crz", "rxx", "ryy", "rzz",
                # Three-qubit gates
                "ccx", "cswap", "cu1",
                # Measurement
                "measure"
            },
            "connectivity": "full",
            "durations_ns": {},
            "noise": {"readout_error": 0.0, "oneq_error": 0.0, "twoq_error": 0.0}
        },
        "runner": LocalSimulator(noise={})
    },
}


def backend(name: str, noise_level: float = 0.0) -> dict:
    # Check if the name is a path to a JSON config file
    if os.path.exists(name) and name.endswith(".json"):
        with open(name, "r") as f:
            profile = json.load(f)

        noise = {
            "readout_error": float(profile.get("readout_error", 0.0)),
            "oneq_error": float(profile.get("oneq_error", 0.0)),
            "twoq_error": float(profile.get("twoq_error", 0.0)),
            "lanes": int(profile.get("queue_lanes", 1)),
            "lat": float(profile.get("base_latency", 0.5))
        }
        durations = profile.get("durations_ns", {})
        sim_name = profile.get("name", os.path.basename(name).split('.')[0])
        
        runner = LocalSimulator(noise=noise)

        return {
            "name": sim_name,
            "type": "sim",
            "runner": runner,
            "caps": {
                "n_qubits_max": int(profile.get("n_qubits_max", 5)),
                "native_gates": set(profile.get("native_gates", [])),
                "connectivity": profile.get("connectivity", "full"),
                "durations_ns": durations,
                "noise": noise
            }
        }

    # Handle built-in backends
    if name not in _REG:
        raise QXBackendError(f"Unknown backend: {name}")

    backend_config = _REG[name].copy()  # Create a copy to avoid modifying the original
    
    # Define a consistent noise model based on the UI slider
    noise_model = {
        "readout_error": noise_level,
        "oneq_error": noise_level / 10,
        "twoq_error": noise_level / 10
    }

    # Instantiate the correct simulator with the noise model
    if name == "sim-local":
        backend_config["runner"] = LocalSimulator(noise=noise_model)
    
    # Update the capabilities dictionary to reflect the current noise model
    backend_config["caps"] = backend_config["caps"].copy()
    backend_config["caps"]["noise"] = noise_model

    # Special handling for sim-local qubit limit
    if name == "sim-local":
        backend_config["caps"]["n_qubits_max"] = get_max_qubits()

    return backend_config