# Default configuration for the quantum simulator

DEFAULT_SHOTS = 1024
DEFAULT_NOISE_LEVEL = 0.0

# Backend configurations
BACKENDS = {
    "sim-local": {
        "description": "A simple local statevector simulator.",
        "noise_model": True
    },
    "zenaquantum-alpha": {
        "description": "ZenaQuantum's high-performance simulator.",
        "noise_model": True
    }
}
