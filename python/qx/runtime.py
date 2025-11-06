from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from .passes import compile_pipeline, compile_report
from .errors import QXRuntimeError
import logging
logger = logging.getLogger("qx.runtime")
from .ir import Program
import hashlib, json, os, time, uuid, itertools, queue, threading
from datetime import datetime, timezone

# ---------- globals ----------
_JOBS: Dict[str, "Job"] = {}
_JOB_QUEUE: "queue.Queue[Job]" = queue.Queue()
_STOP = False

# ---------- helpers ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _runs_dir() -> str:
    d = os.path.join(os.getcwd(), ".qx_runs")
    os.makedirs(d, exist_ok=True)
    return d

def _runhash(prog: Program, caps: dict, shots: int) -> str:
    h = hashlib.sha256()
    h.update(prog.to_json_canonical().encode())
    h.update(json.dumps({"caps": caps.get("name",""), "shots": shots}, sort_keys=True).encode())
    return h.hexdigest()[:16]

# ---------- data ----------
@dataclass
class Result:
    counts: Dict[str,int]
    metadata: Dict[str,Any]

# ---------- job ----------
class Job:
    def __init__(self, job_id: str, circuit=None, backend_desc=None, shots:int=1024, seed:Optional[int]=None):
        self._id = job_id
        self._circuit = circuit
        self._backend_desc = backend_desc
        self._shots = shots
        self._seed = seed
        self._path: Optional[str] = None
        self._result: Optional[Result] = None
        self._status = "QUEUED"
        self._error: Optional[str] = None

    # basic getters
    def id(self): return self._id
    def status(self): return self._status
    def result(self): return self._result
    def artifact_path(self): return self._path
    def error(self): return self._error

    def run_id(self):
        base = os.path.basename(self._path or "")
        return base[:-5] if base.endswith(".json") else ""

# ---------- core synchronous run ----------
def run(circuit, backend_desc:dict, shots:int=1024, params:Optional[Dict[str,float]]=None, seed:Optional[int]=None)->Job:
    caps = backend_desc["caps"]
    name = backend_desc.get("name","unknown-target")
    prog = compile_pipeline(circuit.program, caps)
    logger.info("Target: %s", name)
    logger.info("%s", compile_report(prog, caps))

    runner = backend_desc["runner"]
    # pass optional seed through to runner.execute for reproducibility
    try:
        result = runner.execute(prog, shots=shots, seed=seed)
    except TypeError:
        # runner doesn't accept seed — fall back
        result = runner.execute(prog, shots=shots)
    
    # Ensure we have the expected return format (counts, meta)
    if isinstance(result, tuple) and len(result) == 2:
        counts, meta = result
    elif hasattr(result, 'counts') and hasattr(result, 'metadata'):
        # Handle case where execute returns an object with counts and metadata attributes
        counts = result.counts
        meta = result.metadata
    elif isinstance(result, dict):
        # Handle case where execute returns a single dictionary with counts
        counts = result
        meta = {}
    else:
        raise ValueError(f"Unexpected return type from runner.execute(): {type(result)}")
    
    # Ensure meta is a dictionary
    if not isinstance(meta, dict):
        meta = {}
        
    # Ensure counts is a dictionary with string keys and integer values
    if not isinstance(counts, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in counts.items()):
        raise ValueError("Runner.execute() must return a dictionary of counts with string keys and integer values")
    meta.update({
        "target_name": name,
        "shots": shots,
        "n_qubits": prog.n_qubits,
        "run_started_at": _now_iso(),
        "seed": seed,
    })

    rid = _runhash(prog, {"name": name}, shots)
    jid = f"job-{uuid.uuid4().hex[:8]}"
    artifact = {
        "job_id": jid,
        "run_id": rid,
        "created_at": _now_iso(),
        "counts": counts,
        "metadata": meta
    }
    path = os.path.join(_runs_dir(), f"{rid}.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)

    job = Job(jid, circuit, backend_desc, shots, seed=seed)
    job._result = Result(counts, meta)
    job._path = path
    job._status = "DONE"
    return job

# ---------- background worker ----------
def _worker_loop():
    while not _STOP:
        try:
            job: Job = _JOB_QUEUE.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            job._status = "RUNNING"
            done_job = run(job._circuit, job._backend_desc, shots=job._shots, seed=getattr(job, "_seed", None))
            job._result = done_job._result
            job._path = done_job._path
            job._status = "DONE"
        except Exception as e:
            job._error = str(e)
            job._status = "FAILED"
        finally:
            _JOB_QUEUE.task_done()

# spawn one background thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()

# ---------- Job API ----------
def submit(circuit, backend_desc:dict, shots:int=1024)->Job:
    """Synchronous run (blocks until complete)."""
    return run(circuit, backend_desc, shots=shots)

def submit_async(circuit, backend_desc:dict, shots:int=1024)->Job:
    """Enqueue job for background execution."""
    jid = f"job-{uuid.uuid4().hex[:8]}"
    job = Job(jid, circuit, backend_desc, shots)
    _JOBS[jid] = job
    _JOB_QUEUE.put(job)
    return job
def job_status(job_id: str) -> str:
    j = _JOBS.get(job_id)
    return j._status if j else "UNKNOWN"
def list_jobs() -> Dict[str, str]:
    return {jid: j._status for jid, j in _JOBS.items()}
def cancel(job_id:str)->bool:
    j = _JOBS.get(job_id)
    if not j: return False
    if j._status == "QUEUED":
        # drain queue and rebuild without this job
        import queue as _q
        items=[]
        while True:
            try: items.append(_JOB_QUEUE.get_nowait())
            except _q.Empty: break
        for it in items:
            if it._id != job_id:
                _JOB_QUEUE.put(it)
        j._status = "CANCELLED"
        return True
    if j._status in ("RUNNING",):
        j._status = "CANCEL_REQUESTED"
        return True
    return False

def list_runs()->List[str]:
    return sorted([f[:-5] for f in os.listdir(_runs_dir()) if f.endswith(".json")])

def load_run(run_id:str)->Result:
    path = os.path.join(_runs_dir(), f"{run_id}.json")
    with open(path) as f: data = json.load(f)
    return Result(counts=data["counts"], metadata=data["metadata"])

# ---------- sweeps ----------
def sweep_circuits(circuits, backend_desc:dict, shots:int=1024):
    out=[]
    for c in circuits:
        out.append(run(c, backend_desc, shots=shots).result())
    return out

def sweep_params(build_fn, param_grid:Dict[str,list], backend_desc:dict, shots:int=1024):
    keys=list(param_grid.keys()); results=[]
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        params=dict(zip(keys,vals))
        circ=build_fn(params)
        res=run(circ, backend_desc, shots=shots).result()
        res.metadata["sweep_params"]=params
        results.append((params,res))
    return results
def export_sweeps(results, path_csv:str=None, path_json:str=None):
    import csv, json
    if path_json:
        json.dump([{"params":p,"counts":r.counts,"metadata":r.metadata} for p,r in results], open(path_json,"w"), indent=2)
    if path_csv:
        # flatten counts' top outcomes
        with open(path_csv,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["param","key","value"])
            for p,r in results:
                for k,v in r.counts.items():
                    w.writerow([json.dumps(p), k, v])

def set_global_seed(seed:int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)

def list_runs_filtered(target: str | None = None, since: str | None = None, until: str | None = None, runhash: str | None = None):
    """Filter runs by optional target name, ISO since/until dates, or run_id substring/hash.

    Returns list of run ids (filenames without .json) sorted ascending.
    """
    from datetime import datetime
    out = []
    for rid in list_runs():
        if runhash and runhash not in rid:
            continue
        data = load_run(rid)
        if target and data.metadata.get("target_name") != target:
            continue
        if since:
            t = datetime.fromisoformat(data.metadata.get("run_started_at"))
            if t < datetime.fromisoformat(since):
                continue
        if until:
            t = datetime.fromisoformat(data.metadata.get("run_started_at"))
            if t > datetime.fromisoformat(until):
                continue
        out.append(rid)
    return out

def load_last_run():
    ids = list_runs()
    return load_run(ids[-1]) if ids else None