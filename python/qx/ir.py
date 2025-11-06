from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json, hashlib
import json, os

_SCHEMAPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "schema", "qx-ir-v1.json")

def validate_ir_json(data: dict):
    # Lazy import jsonschema so the module isn't required at import-time
    try:
        import importlib
        jsonschema = importlib.import_module("jsonschema")
    except Exception:
        return
    if not os.path.exists(_SCHEMAPATH):
        # schema missing — skip validation
        return
    with open(_SCHEMAPATH, "r") as f:
        schema = json.load(f)
    jsonschema.validate(data, schema)

@dataclass
class Op:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    clbits: Tuple[int, ...] = ()
    condition: Optional[int] = None

@dataclass
class Program:
    n_qubits: int
    ops: List[Op] = None
    n_clbits: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        # avoid shared mutable default
        if self.ops is None:
            self.ops = []

    @classmethod
    def from_json(cls, s: str) -> "Program":
        data = json.loads(s)
        validate_ir_json(data)
        # support either 'n_qubits' or legacy 'qubits' key
        n_qubits = int(data.get("n_qubits", data.get("qubits", 0)))
        n_clbits = int(data.get("n_clbits", 0))
        prog = Program(n_qubits=n_qubits, ops=[], n_clbits=n_clbits, metadata=data.get("metadata", {}))
        for o in data.get("ops", []):
            name = o["name"]
            qus = tuple(o.get("qubits", []))
            prs = tuple(o.get("params", []))
            cls_ = tuple(o.get("clbits", []))
            cond = o.get("condition")
            prog.ops.append(Op(name=name, qubits=qus, params=prs, clbits=cls_, condition=cond))
        return prog

    def add(self, op: Op):
        self.ops.append(op)

    def validate(self):
        for op in self.ops:
            for q in op.qubits:
                if not (0 <= q < self.n_qubits):
                    raise ValueError(f"Qubit index {q} out of range 0..{self.n_qubits-1}")
            if op.name == "measure":
                if len(op.clbits) != len(op.qubits):
                    raise ValueError("measure must map each measured qubit to a clbit")
                for c in op.clbits:
                    if not (0 <= c < self.n_clbits):
                        raise ValueError(f"Clbit index {c} out of range 0..{self.n_clbits-1}")

    def to_json_obj(self):
        return {
            "ir_version": "1.0",
            "qubits": self.n_qubits,
            "ops": [
                {
                    "name": op.name,
                    "qubits": list(op.qubits),
                    **({"params": [float(f"{p:.12f}") for p in op.params]} if op.params else {}),
                    **({"clbits": list(op.clbits)} if op.clbits else {}),
                    **({"condition": op.condition} if op.condition is not None else {})
                } for op in self.ops
            ],
            **({"metadata": self.metadata} if self.metadata else {})
        }

    def to_json_canonical(self) -> str:
        # Sorted keys + no spaces for deterministic hashing
        return json.dumps(self.to_json_obj(), sort_keys=True, separators=(',', ':'))

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_canonical().encode()).hexdigest()
