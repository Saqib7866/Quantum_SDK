import click
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path to allow relative imports
sys.path.append(str(Path(__file__).parent.parent))

from qx_ir.storage import JobMetadata
from qx_ir.core import Program, Circuit, Op
from qx_ir.backend import LocalBackend
from qx_ir.batch_processor import BatchProcessor

class QXCLI:
    """Main CLI application class."""
    
    def __init__(self):
        self.storage = JobMetadata()
        self.backend = LocalBackend()

    def run_circuit(self, circuit: Circuit, shots: int = 1024) -> str:
        """Run a circuit and return job ID."""
        job = self.backend.submit(Program([circuit], {'shots': shots}))
        
        # Save metadata
        metadata = {
            'job_id': job.job_id,
            'created_at': datetime.now().isoformat(),
            'shots': shots,
            'n_qubits': circuit.n_qubits,
            'status': 'QUEUED'
        }
        self.storage.save_metadata(job.job_id, metadata)
        
        # Save initial result
        self.storage.save_result(job.job_id, {'status': 'PENDING'})
        
        return job.job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job."""
        metadata = self.storage.get_metadata(job_id)
        if not metadata:
            return {'error': f'Job {job_id} not found'}
        
        result = self.storage.get_result(job_id) or {}
        return {**metadata, **result}

    def list_jobs(self) -> list:
        """List all jobs with their status."""
        return self.storage.list_jobs()
        
    def get_job_result(self, job_id: str) -> dict:
        """Get the result of a specific job."""
        return self.storage.get_result(job_id)

# Create CLI instance
app = QXCLI()

# Create Click command group
@click.group()
def qx():
    """Quantum SDK Command Line Interface."""
    pass

@qx.command()
@click.argument('circuit_file', type=click.Path(exists=True))
@click.option('--shots', type=int, default=1024, help='Number of shots to run')
def run(circuit_file: str, shots: int):
    """Run a quantum circuit from a JSON file."""
    try:
        # Load circuit from file
        with open(circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        # Create circuit
        circuit = Circuit(n_qubits=circuit_data['n_qubits'])
        for inst in circuit_data['instructions']:
            circuit.add_op(Op(inst['name'], inst['qubits']))
        
        # Run circuit
        job_id = app.run_circuit(circuit, shots=shots)
        click.echo(f"Job submitted with ID: {job_id}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@qx.command()
@click.argument('job_id', required=False)
@click.option('--verbose', '-v', is_flag=True, help='Show detailed job information')
def status(job_id: Optional[str] = None, verbose: bool = False):
    """Get status of jobs. If no job_id is provided, lists all jobs."""
    if job_id:
        # Show status of specific job
        job_data = app.get_job_status(job_id)
        if verbose:
            click.echo(json.dumps(job_data, indent=2, default=str))
        else:
            status = job_data.get('status', 'UNKNOWN')
            click.echo(f"Job {job_id}: {status}")
            if status == 'DONE' and 'result' in job_data:
                click.echo("Results:")
                click.echo(json.dumps(job_data['result'], indent=2))
            elif status == 'FAILED' and 'error' in job_data:
                click.echo(f"Error: {job_data['error']}")
    else:
        # List all jobs
        jobs = app.list_jobs()
        if not jobs:
            click.echo("No jobs found.")
            return
            
        # Display jobs in a table
        click.echo(f"{'Job ID':<36} {'Status':<10} {'Qubits':<6} {'Shots':<6} {'Created At'}")
        click.echo("-" * 80)
        
        # Sort jobs by creation time (newest first)
        jobs_sorted = sorted(
            jobs, 
            key=lambda x: x.get('created_at', ''), 
            reverse=True
        )
        
        for job in jobs_sorted:
            job_id = job.get('job_id', '')
            
            # Get the latest status from storage
            job_data = app.get_job_status(job_id)
            status = job_data.get('status', 'UNKNOWN')
            
            # Update the status in the job listing
            job['status'] = status
            
            click.echo(
                f"{job_id[:8]}... "
                f"{status:<10} "
                f"{job.get('n_qubits', '?'):<6} "
                f"{job.get('shots', '?'):<6} "
                f"{job.get('created_at', '')}"
            )

@qx.command()
@click.argument('job_id')
def report(job_id: str):
    """Generate a detailed report for a job."""
    try:
        result = app.get_job_result(job_id)
        if not result:
            click.echo(f"No result found for job {job_id}")
            return
            
        click.echo(f"Report for Job {job_id}")
        click.echo("=" * 50)
        click.echo(f"Status: {result.get('status', 'UNKNOWN')}")
        click.echo(f"Created at: {result.get('created_at', 'N/A')}")
        
        if 'result' in result:
            click.echo("\nResults:")
            click.echo("-" * 20)
            click.echo(json.dumps(result['result'], indent=2))
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    qx()
