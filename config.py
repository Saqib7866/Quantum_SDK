# Default configuration for the quantum simulator

DEFAULT_SHOTS = 1024
DEFAULT_NOISE_LEVEL = 0.0
DEFAULT_MAX_QUBITS = 5

# Backend configurations
BACKENDS = {
    "sim-local": {
        "description": "A simple local statevector simulator.",
        "noise_supported": True
    },
}
