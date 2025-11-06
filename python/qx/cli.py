"""Small CLI wrappers used as package entry points.

Provides run_main and report_main so installers can expose console scripts.
"""
import sys
import json
from . import Circuit, backend
from .config import set_max_qubits
from .runtime import run as runtime_run
import os

def run_main(argv=None):
    # If argv is None, use sys.argv; if an empty list is provided, treat it as no args.
    argv = sys.argv[1:] if argv is None else argv
    import argparse

    ap = argparse.ArgumentParser(description="Run a simple circuit on a backend")
    ap.add_argument("--backend", default="sim-local", help="sim-local or path to target json")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--example", choices=["bell"], default="bell")
    ap.add_argument("--max-qubits", type=int, help="Override max qubits for this process (sets QX_MAX_QUBITS)")
    ap.add_argument("--seed", type=int, help="Deterministic RNG seed to use for this run")
    args = ap.parse_args(argv)

    def bell():
        with Circuit() as qc:
            a, b = qc.allocate(2)
            qc.h(a); qc.cx(a, b); qc.measure(a, b)
        return qc

    # allow process-level override of max qubits (set before backend lookup so caps reflect it)
    if getattr(args, "max_qubits", None) is not None:
        set_max_qubits(args.max_qubits)
    b = backend(args.backend)
    qc = bell()
    job = runtime_run(qc, b, shots=args.shots, seed=getattr(args, "seed", None))
    res = job.result()
    # user-facing output should go to stdout; also log at info level
    import logging
    logger = logging.getLogger("qx.cli")
    logger.info("Run completed: %s", job.id())
    print(json.dumps({"counts": res.counts, "metadata": res.metadata}, indent=2))
    return 0

def report_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    import argparse
    from .runtime import list_runs, load_run

    ap = argparse.ArgumentParser(description="List runs or show a run artifact")
    ap.add_argument("--list", action="store_true", help="List run_ids in ./.qx_runs")
    ap.add_argument("--run", help="Show one run by run_id (16 hex)")
    args = ap.parse_args(argv)

    if args.list:
        for rid in list_runs():
            print(rid)
        return 0
    if args.run:
        res = load_run(args.run)
        print(json.dumps({"counts": res.counts, "metadata": res.metadata}, indent=2))
        return 0
    print("Use --list or --run <id>")
    return 0
