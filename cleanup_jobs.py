#!/usr/bin/env python3
"""
Script to clean up pending jobs from the job queue.
"""
import json
import shutil
from pathlib import Path
from qx_ir.storage import JobMetadata
from qx_ir.job import JobStatus

def cleanup_pending_jobs(dry_run: bool = False):
    """Remove all jobs with PENDING status.
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting.
    """
    storage = JobMetadata()
    jobs_dir = Path("qx_jobs")
    
    if not jobs_dir.exists():
        print("No jobs directory found. Nothing to clean up.")
        return
    
    pending_count = 0
    total_count = 0
    
    # Iterate through all job directories
    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
            
        total_count += 1
        job_id = job_dir.name
        metadata_path = job_dir / "metadata.json"
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            if metadata.get('status') == 'QUEUED' or metadata.get('status') == 'PENDING':
                pending_count += 1
                print(f"Found pending job: {job_id}")
                
                if not dry_run:
                    # Remove the job directory and all its contents
                    shutil.rmtree(job_dir)
                    print(f"  → Removed job {job_id}")
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not process job {job_id}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"- Total jobs found: {total_count}")
    print(f"- Pending jobs found: {pending_count}")
    
    if dry_run:
        print("\nThis was a dry run. No changes were made.")
        print("Run with '--apply' to actually remove pending jobs.")
    else:
        print(f"\nRemoved {pending_count} pending jobs.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up pending jobs from the job queue.")
    parser.add_argument(
        "--apply", 
        action="store_true",
        help="Actually remove the pending jobs. Without this flag, only shows what would be done."
    )
    args = parser.parse_args()
    
    cleanup_pending_jobs(dry_run=not args.apply)
