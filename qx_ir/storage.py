import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class JobMetadata:
    """Handles storage and retrieval of job metadata and results."""
    
    def __init__(self, storage_dir: str = "qx_jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory path for a job."""
        return self.storage_dir / job_id
    
    def save_metadata(self, job_id: str, metadata: Dict[str, Any]) -> str:
        """Save metadata for a job."""
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(exist_ok=True)
        
        metadata_path = job_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return str(metadata_path)
    
    def save_result(self, job_id: str, result: Dict[str, Any]) -> str:
        """Save job results."""
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(exist_ok=True)
        
        result_path = job_dir / "result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return str(result_path)
    
    def get_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a job."""
        metadata_path = self._get_job_dir(job_id) / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path) as f:
            return json.load(f)
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job results."""
        result_path = self._get_job_dir(job_id) / "result.json"
        if not result_path.exists():
            return None
            
        with open(result_path) as f:
            return json.load(f)
    
    def list_jobs(self) -> list:
        """List all stored jobs with their metadata."""
        jobs = []
        for job_dir in self.storage_dir.glob("*"):
            if job_dir.is_dir():
                metadata = self.get_metadata(job_dir.name)
                if metadata:
                    jobs.append(metadata)
        return sorted(jobs, key=lambda x: x.get('created_at', ''), reverse=True)
