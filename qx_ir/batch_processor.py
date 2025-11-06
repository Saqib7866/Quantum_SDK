from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from .core import Program, Circuit, Op
from .job import Job, JobStatus
from .backend import LocalBackend

class BatchProcessor:
    """Handles batch execution of multiple quantum programs and parameter sweeps."""

    def __init__(self, backend=None, max_workers: int = 4):
        """Initialize with an optional backend and maximum worker threads."""
        self.backend = backend or LocalBackend()
        self.max_workers = max_workers

    def submit_batch(self, programs: List[Program]) -> List[Job]:
        """Submit multiple programs for parallel execution."""
        with ThreadPoolExecutor(max_workers=self.min(len(programs), self.max_workers)) as executor:
            return list(executor.map(self.backend.submit, programs))

    def parameter_sweep(
        self,
        circuit_template: Circuit,
        parameter_sets: List[Dict[str, float]],
        shots: int = 1024
    ) -> Dict[str, Any]:
        """
        Run multiple circuits with different parameter values.
        
        Args:
            circuit_template: A circuit with placeholders like {param_name}
            parameter_sets: List of parameter dictionaries
            shots: Number of shots per parameter set
            
        Returns:
            Dictionary mapping parameter sets to their results
        """
        results = {}
        programs = []
        
        # Create programs for each parameter set
        for params in parameter_sets:
            param_circuit = self._apply_parameters(circuit_template, params)
            programs.append(Program([param_circuit], {'shots': shots}))
        
        # Run all programs
        jobs = self.submit_batch(programs)
        
        # Collect results
        for params, job in zip(parameter_sets, jobs):
            while job.status() not in [JobStatus.DONE, JobStatus.FAILED]:
                continue
            if job.status() == JobStatus.DONE:
                results[str(params)] = job.result()
            else:
                results[str(params)] = {"error": "Job failed", "status": job.status().value}
                
        return results

    def _apply_parameters(self, circuit: Circuit, params: Dict[str, float]) -> Circuit:
        """Replace parameter placeholders in the circuit with actual values."""
        param_circuit = Circuit(circuit.n_qubits)
        for op in circuit.instructions:
            # Handle parameter substitution
            if op.params:
                new_params = [
                    params.get(str(p), p) if isinstance(p, str) and p.startswith('{') and p.endswith('}') else p 
                    for p in op.params
                ]
                param_circuit.add_op(Op(op.name, op.qubits, new_params))
            else:
                param_circuit.add_op(Op(op.name, op.qubits, op.params))
        return param_circuit
    
    @staticmethod
    def min(a: int, b: int) -> int:
        """Helper method to avoid shadowing built-in min() in map."""
        return a if a < b else b
