"""
Storage and metadata handling for quantum jobs.
"""
import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Union
from enum import Enum

# Create a directory to store job results if it doesn't exist
STORAGE_DIR = os.path.expanduser("~/.qx_ir/jobs")
os.makedirs(STORAGE_DIR, exist_ok=True)

class JobStatus(str, Enum):
    """Possible status values for a quantum job."""
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    DONE = 'DONE'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


def get_job_path(job_id: str) -> str:
    """Get the file path for storing job data."""
    return os.path.join(STORAGE_DIR, f"{job_id}.json")


def save_job_result(job_id: str, result: Dict[str, Any]) -> None:
    """Save job result to disk."""
    file_path = get_job_path(job_id)
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=2)


def load_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job result from disk."""
    file_path = get_job_path(job_id)
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)


@dataclass
class JobMetadata:
    """Metadata for a quantum job."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    creation_time: datetime = field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save_result(self, result: Dict[str, Any]) -> None:
        """Save the job result to persistent storage."""
        # Get status from result or use default
        status = result.get('status', JobStatus.DONE)
        
        # Convert string status to JobStatus if needed
        if isinstance(status, str):
            status = JobStatus(status.upper())
            
        self.status = status
        self.end_time = datetime.utcnow()
        
        if 'error' in result:
            self.error_message = result['error']
            
        # Prepare the result data
        result_data = result.copy()
        result_data['job_id'] = self.job_id
        result_data['status'] = self.status.value if hasattr(self.status, 'value') else str(self.status)
        result_data['end_time'] = self.end_time.isoformat()
        
        # Add creation time if not present
        if 'creation_time' not in result_data:
            result_data['creation_time'] = self.creation_time.isoformat()
        
        save_job_result(self.job_id, result_data)
    
    @classmethod
    def load(cls, job_id: str) -> 'JobMetadata':
        """Load job metadata from disk."""
        result = load_job_result(job_id)
        if not result:
            return cls(job_id=job_id, status=JobStatus.PENDING)
            
        return cls(
            job_id=job_id,
            status=JobStatus(result.get('status', 'PENDING')),
            creation_time=datetime.fromisoformat(result.get('creation_time', datetime.utcnow().isoformat())),
            end_time=datetime.fromisoformat(result['end_time']) if 'end_time' in result else None,
            error_message=result.get('error'),
            metadata=result.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metadata to a dictionary."""
        return {
            'job_id': self.job_id,
            'status': self.status.value,
            'creation_time': self.creation_time.isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
