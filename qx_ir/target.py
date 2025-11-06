import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

class Target:
    """
    Represents a quantum device's capabilities and constraints.

    Attributes:
        name (str): Backend or device name.
        version (str): Backend version string.
        n_qubits (int): Number of qubits on the device.
        basis_gates (List[str]): Supported gate set.
        coupling_map (List[Tuple[int, int]]): List of connected qubit pairs.
        gate_fidelities (Dict[str, float]): Optional gate fidelities.
        custom_properties (Dict[str, Any]): Extra device-specific settings.

    Methods:
        from_file(file_path): Load a target profile from a JSON file.
        __repr__(): Return a short summary of the target device.
    """

    def __init__(self, data: Dict[str, Any]):
        self.name: str = data.get("backend_name", "unknown")
        self.version: str = data.get("backend_version", "0.0.0")
        self.n_qubits: int = data["n_qubits"]
        self.basis_gates: List[str] = data["basis_gates"]
        self.coupling_map: List[Tuple[int, int]] = [tuple(edge) for edge in data["coupling_map"]]
        self.gate_fidelities: Dict[str, float] = data.get("gate_fidelities", {})
        self.custom_properties: Dict[str, Any] = data.get("custom", {})

    @classmethod
    def from_file(cls, file_path: str):
        """Load a target profile from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Target profile not found: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(data)

    def __repr__(self) -> str:
        return f"Target(name='{self.name}', n_qubits={self.n_qubits}, basis_gates={self.basis_gates})"
