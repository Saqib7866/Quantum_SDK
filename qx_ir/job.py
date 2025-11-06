import uuid
from enum import Enum
from typing import Any, Optional

class JobStatus(Enum):
    """Enumeration for the status of a job."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class Job:
    """Represents a job submitted to a backend."""

    def __init__(self, job_id: Optional[str] = None):
        """Initializes the Job object.
        
        Args:
            job_id: Optional custom job ID. If not provided, a UUID will be generated.
        """
        self._job_id = job_id if job_id else str(uuid.uuid4())
        self._status = JobStatus.QUEUED
        self._result: Any = None
        self._future = None  # Will hold the concurrent.futures.Future object

    @property
    def job_id(self) -> str:
        """Return the unique ID of the job."""
        return self._job_id

    @property
    def status(self) -> JobStatus:
        """Return the current status of the job."""
        return self._status
        
    @status.setter
    def status(self, value: JobStatus) -> None:
        """Set the status of the job."""
        self._status = value

    def result(self, timeout: Optional[float] = None) -> Any:
        """Return the result of the job.
        
        Args:
            timeout: Maximum time to wait for the job to complete, in seconds.
                    If None, the method will block until the job is done.
                    
        Returns:
            The job result if completed, or a status dictionary if still running.
            
        Raises:
            concurrent.futures.TimeoutError: If the job doesn't complete within the timeout.
            Exception: If the job raised an exception during execution.
        """
        if self._status == JobStatus.DONE:
            return self._result
            
        if self._future is not None:
            try:
                # Wait for the future to complete with a timeout
                self._result = self._future.result(timeout=timeout)
                self._status = JobStatus.DONE
                return self._result
            except concurrent.futures.TimeoutError:
                return {'status': self._status.value, 'error': 'Job not completed'}
            except Exception as e:
                self._status = JobStatus.FAILED
                self._result = {'status': 'FAILED', 'error': str(e)}
                raise
                
        return {'status': self._status.value, 'error': 'Job not started'}
