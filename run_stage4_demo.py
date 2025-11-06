import datetime
import json
import time

from qx_ir.core import Circuit, Op, Program
from qx_ir.backend import LocalBackend
from qx_ir.job import JobStatus

def main():
    """Demonstrate the Stage 4 job submission and result retrieval API."""

    # 1. Create a simple circuit (e.g., a Bell state)
    circuit = Circuit(n_qubits=2)
    circuit.add_op(Op('h', qubits=[0]))
    circuit.add_op(Op('cx', qubits=[0, 1]))
    print("✅ Circuit created.")

    # 2. Create a program to run on the backend
    program = Program(circuits=[circuit], config={'shots': 1024})
    print("✅ Program created.")

    # 3. Initialize the local backend
    backend = LocalBackend()
    print(f"✅ Initialized backend: {type(backend).__name__}")

    # 4. Submit the job
    print("\n🚀 Submitting job...")
    job = backend.submit(program)
    print(f"  - Job submitted with ID: {job.job_id}")
    
    # Save job metadata (similar to what QXCLI does)
    from qx_ir.storage import JobMetadata
    storage = JobMetadata()
    metadata = {
        'job_id': job.job_id,
        'created_at': datetime.datetime.now().isoformat(),
        'shots': program.config.get('shots', 1024),
        'n_qubits': program.circuits[0].n_qubits if program.circuits else 0,
        'status': 'QUEUED'
    }
    storage.save_metadata(job.job_id, metadata)

    # 5. Monitor the job status until it's done
    print("\n⏳ Waiting for job to complete...")
    while job.status not in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED]:
        print(f"  - Current status: {job.status.value}")
        time.sleep(0.5) # Poll every half second

    final_status = job.status
    print(f"\n🎉 Job finished with status: {final_status.value}")

    # 6. Retrieve and display the result
    if final_status == JobStatus.DONE:
        result = backend.result(job.job_id)
        print("\n📊 Measurement Results:")
        
        # The result is a dictionary with measurement outcomes and counts
        if isinstance(result, dict):
            # If the result is already in the format we expect (e.g., {'00': 500, '11': 500})
            if all(isinstance(k, str) and k.isdigit() and isinstance(v, int) 
                  for k, v in result.items()):
                sorted_counts = sorted(result.items(), key=lambda item: item[1], reverse=True)
                for outcome, count in sorted_counts:
                    print(f"  - |{outcome}>: {count}")
            # If the result is in a different format (e.g., from a different backend)
            else:
                print("  Raw result:", json.dumps(result, indent=2))
        else:
            print("  Result format not recognized:", result)
    else:
        print("\n❌ Job failed or was cancelled. No results to display.")

if __name__ == "__main__":
    main()
