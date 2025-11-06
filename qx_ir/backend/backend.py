import datetime
import threading
import concurrent.futures
from typing import Dict, Optional

from qx_ir.core import Program
from qx_ir.job import Job, JobStatus
from qx_ir.simulator import StatevectorSimulator

class LocalBackend:
    """A local backend for submitting and managing simulation jobs."""

    def __init__(self, max_workers: Optional[int] = None):
        self._simulator = StatevectorSimulator()
        self._jobs: Dict[str, Job] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or 4,
            thread_name_prefix='qx_worker_'
        )

    def submit(self, program: Program) -> Job:
        """Submit a program for execution and return a Job object."""
        job = Job()
        self._jobs[job.job_id] = job

        # Submit the job to the thread pool
        future = self._executor.submit(self._execute, job, program)
        job._future = future
        
        # Add callback to handle completion
        future.add_done_callback(lambda f: self._handle_job_completion(job, f))

        return job

    def status(self, job_id: str) -> JobStatus:
        """Get the status of a job by its ID."""
        if job_id not in self._jobs:
            raise ValueError("Job ID not found.")
        return self._jobs[job_id].status

    def result(self, job_id: str) -> Dict:
        """Get the result of a completed job by its ID."""
        if job_id not in self._jobs:
            raise ValueError("Job ID not found.")
        
        job = self._jobs[job_id]
        if job.status != JobStatus.DONE:
            # In a real scenario, you might want to wait here or just raise an error.
            return job.result()
        
        return job.result()

    def _handle_job_completion(self, job: Job, future: concurrent.futures.Future) -> None:
        """Handle the completion of a job."""
        from .storage import JobMetadata
        
        try:
            # Get the result (this will re-raise any exceptions)
            result = future.result()
            
            # Update job status and result
            job._result = result
            job.status = JobStatus.DONE
            
            # Store the result in persistent storage
            metadata = JobMetadata(job_id=job.job_id, status=JobStatus.DONE)
            metadata.save_result(job._result)
            
        except Exception as e:
            error_msg = str(e)
            job.status = JobStatus.FAILED
            job._result = {
                'status': 'FAILED',
                'error': error_msg,
                'job_id': job.job_id,
                'completed_at': datetime.datetime.now().isoformat()
            }
            
            # Store the error in persistent storage
            metadata = JobMetadata(job_id=job.job_id, status=JobStatus.FAILED)
            metadata.save_result(job._result)
            
            # Print error for debugging
            import traceback
            traceback.print_exc()
    
    def _execute(self, job: Job, program: Program) -> dict:
        """The internal execution logic that runs in a separate thread."""
        try:
            job.status = JobStatus.RUNNING
            
            # The simulator's run method is synchronous, so it blocks until the simulation is done.
            result_counts = self._simulator.run(program)
            
            result = {
                'status': 'DONE',
                'result': result_counts,
                'success': True,
                'job_id': job.job_id,
                'completed_at': datetime.datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            # Let the error propagate to the future
            raise
