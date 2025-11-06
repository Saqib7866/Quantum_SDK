import json
from pathlib import Path
import jsonschema

def validate_circuit(circuit_file: str, schema_file: str):
    """Validate a circuit file against the QX-IR schema."""
    # Load the schema
    schema_path = Path(schema_file)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Load the circuit
    circuit_path = Path(circuit_file)
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit file not found: {circuit_file}")
    with open(circuit_path, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    # Validate
    try:
        jsonschema.validate(instance=circuit_data, schema=schema)
        print(f"✅ Circuit '{circuit_file}' is valid against '{schema_file}'.")
    except jsonschema.exceptions.ValidationError as err:
        print(f"❌ Circuit '{circuit_file}' is invalid.")
        print(f"Validation Error: {err.message}")

if __name__ == "__main__":
    validate_circuit("example_circuit.json", "circuit_schema.json")

