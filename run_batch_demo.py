import time
from qx_ir import Circuit, Op, Program
from qx_ir.batch_processor import BatchProcessor

def create_parameterized_circuit() -> Circuit:
    """Create a circuit with parameterized gates."""
    circuit = Circuit(2)
    # Parameterized gates using {param_name} syntax
    circuit.add_op(Op('ry', [0], ['{theta}']))  # Will be replaced with actual values
    circuit.add_op(Op('rx', [1], ['{phi}']))
    circuit.add_op(Op('cx', [0, 1]))
    circuit.add_op(Op('measure', [0, 1]))
    return circuit

def main():
    print("🚀 Starting batch processing demo...")
    
    # 1. Create parameter sets to sweep
    parameter_sets = [
        {'theta': 0.0, 'phi': 0.0},
        {'theta': 0.5, 'phi': 0.5},
        {'theta': 1.0, 'phi': 1.0},
        {'theta': 1.5, 'phi': 1.5},
        {'theta': 2.0, 'phi': 2.0},
    ]
    print(f"🔢 Created {len(parameter_sets)} parameter sets")

    # 2. Initialize batch processor
    processor = BatchProcessor(max_workers=3)  # Run up to 3 circuits in parallel
    print("⚙️  Initialized batch processor")

    # 3. Run the parameter sweep
    print("\n🔄 Running parameter sweep...")
    start_time = time.time()
    
    results = processor.parameter_sweep(
        circuit_template=create_parameterized_circuit(),
        parameter_sets=parameter_sets,
        shots=500
    )
    
    # 4. Display results
    print("\n✅ Parameter sweep completed!")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
    
    print("\n📊 Results:")
    print("-" * 50)
    for params, result in results.items():
        print(f"\nParameters: {params}")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            # Sort results by count (descending)
            sorted_counts = sorted(result.items(), key=lambda x: x[1], reverse=True)
            for outcome, count in sorted_counts:
                print(f"  |{outcome}>: {count}")
    print("-" * 50)

if __name__ == "__main__":
    main()
