from .ir import Program, Op
from .errors import QXCompileError
from collections import Counter, deque

# ---------- Validation & capability checks ----------
def validate_ir(prog: Program) -> Program:
    prog.validate()
    return prog

def check_qubit_limit(prog: Program, n_qubits_max: int) -> None:
    if prog.n_qubits > n_qubits_max:
        raise QXCompileError(f"Program uses {prog.n_qubits} qubits; device max is {n_qubits_max}")

def check_native_gates(prog: Program, native: set) -> None:
    # Legacy-friendly: if native set is empty/None, treat as unrestricted
    if not native:
        return
    bad = [op.name for op in prog.ops if op.name not in native]
    if bad:
        uniq = sorted(set(bad))
        raise QXCompileError(f"Non-native gates found: {uniq}. Allowed: {sorted(native)}")

# ---------- Lightweight optimization ----------
def optimize_cancel_x(prog: Program) -> Program:
    new_ops = []
    last_x = {}
    for op in prog.ops:
        if op.name == "x" and len(op.qubits)==1:
            q = op.qubits[0]
            if last_x.get(q, False):
                new_ops.pop()
                last_x[q] = False
                continue
            else:
                last_x[q] = True
        else:
            if len(op.qubits)==1 and op.name != "x":
                last_x[op.qubits[0]] = False
        new_ops.append(op)
    prog.ops = new_ops
    return prog

def optimize_merge_rz(prog: Program) -> Program:
    merged = []
    i = 0
    while i < len(prog.ops):
        op = prog.ops[i]
        if op.name == "rz" and i+1 < len(prog.ops):
            nxt = prog.ops[i+1]
            if nxt.name == "rz" and nxt.qubits == op.qubits:
                theta = op.params[0] + nxt.params[0]
                merged.append(Op("rz", op.qubits, (theta,), ()))
                i += 2
                continue
        merged.append(op); i += 1
    prog.ops = merged
    return prog

# ---------- Routing (shortest-path SWAP insertion) ----------
def _adj(connectivity, n):
    # Clip edges to active nodes [0..n-1]
    g = {i: set() for i in range(n)}
    if connectivity == "full":
        return g
    for u, v in connectivity:
        if 0 <= u < n and 0 <= v < n:
            g[u].add(v); g[v].add(u)
    return g

def _shortest_path(g, s, t):
    if s == t: return [s]
    q = deque([s]); prev = {s: None}
    while q:
        u = q.popleft()
        for v in g[u]:
            if v not in prev:
                prev[v] = u
                if v == t:
                    path = [v]
                    while u is not None:
                        path.append(u); u = prev[u]
                    return list(reversed(path))
                q.append(v)
    return None

def route_swaps(prog: Program, connectivity, n_qubits):
    if connectivity == "full":
        return prog
    g = _adj(connectivity, n_qubits)
    new_ops = []
    loc_of = {q: q for q in range(n_qubits)}
    log_at = {q: q for q in range(n_qubits)}

    def emit_swap(p, q):
        nonlocal loc_of, log_at, new_ops
        new_ops.append(Op("swap", (p,q)))
        lp, lq = log_at[p], log_at[q]
        log_at[p], log_at[q] = lq, lp
        loc_of[lp], loc_of[lq] = q, p

    for op in prog.ops:
        if op.name in ("h","x","y","z","rx","ry","rz"):
            p = loc_of[op.qubits[0]]
            new_ops.append(Op(op.name, (p,), op.params))
        elif op.name == "cx":
            lc, lt = op.qubits
            pc, pt = loc_of[lc], loc_of[lt]
            if pc == pt:
                new_ops.append(Op("x", (pc,)))
                continue
            if connectivity == "full" or pt in g[pc]:
                new_ops.append(Op("cx", (pc, pt)))
            else:
                path = _shortest_path(g, pt, pc)
                if not path or len(path) < 2:
                    raise QXCompileError(f"No path between {pc} and {pt} on given connectivity")
                for i in range(len(path)-1):
                    a, b = path[i], path[i+1]
                    emit_swap(a, b)
                pc, pt = loc_of[lc], loc_of[lt]
                if connectivity != "full" and pt not in g[pc]:
                    raise QXCompileError("Routing failed to make qubits adjacent")
                new_ops.append(Op("cx", (pc, pt)))
        elif op.name == "measure":
            p = loc_of[op.qubits[0]]
            new_ops.append(Op("measure", (p,), (), op.clbits))
        else:
            raise QXCompileError(f"Unsupported op during routing: {op.name}")
    prog.ops = new_ops
    return prog

# ---------- Cap normalization ----------
def _normalize_caps(caps: dict) -> dict:
    default_native = {"h","x","y","z","rx","ry","rz","cx","measure","swap"}
    native = caps.get("native_gates", default_native)
    if not native:
        native = default_native
    if not isinstance(native, set):
        native = set(native)
    return {
        "n_qubits_max": int(caps.get("n_qubits_max", 5)),
        "native_gates": native,
        "connectivity": caps.get("connectivity", "full"),
        "noise": caps.get("noise", {}),
        "durations_ns": caps.get("durations_ns", {}),
    }

# ---------- Compile pipeline + report ----------
def compile_pipeline(prog: Program, caps: dict) -> Program:
    caps = _normalize_caps(caps)
    prog = validate_ir(prog)
    check_qubit_limit(prog, caps["n_qubits_max"])
    check_native_gates(prog, caps["native_gates"])
    prog = optimize_cancel_x(prog)
    prog = optimize_merge_rz(prog)
    prog = route_swaps(prog, caps["connectivity"], prog.n_qubits)
    check_native_gates(prog, caps["native_gates"])  # after routing (swap)
    return prog
def _per_qubit_depth(prog):
    depth = [0]*prog.n_qubits
    for op in prog.ops:
        qs = list(op.qubits)
        if not qs: continue
        d = max(depth[q] for q in qs) + 1
        for q in qs: depth[q] = d
    return depth

def _avg_cx_distance(prog):
    cx = [abs(a-b) for a,b in (op.qubits for op in prog.ops if op.name=="cx")]
    return sum(cx)/len(cx) if cx else 0.0

def compile_report(prog: Program, caps: dict) -> str:
    counts = Counter(op.name for op in prog.ops)
    # crude depth: 1 layer for groups of 1q; +1 for each two-qubit/swap layer
    depth = 0; in_1q = False
    for op in prog.ops:
        if op.name in ("cx","swap"):
            depth += 1; in_1q = False
        else:
            if not in_1q:
                depth += 1; in_1q = True
    noise = caps.get("noise", {}) if isinstance(caps.get("noise", {}), dict) else {}
    e1 = float(noise.get("oneq_error", 0.0))
    e2 = float(noise.get("twoq_error", 0.0))
    er = float(noise.get("readout_error", 0.0))
    n1 = sum(v for k,v in counts.items() if k in ("h","x","y","z","rx","ry","rz"))
    n2 = counts.get("cx", 0)
    nm = counts.get("measure", 0)
    fidelity = (1.0 - e1)**max(n1,0) * (1.0 - e2)**max(n2,0) * (1.0 - er)**max(nm,0)
    lines = ["GateCounts:"] + [f"  {k}: {v}" for k,v in sorted(counts.items())] + [
        f"Depth≈{depth}",
        f"EstFidelity≈{fidelity:.4f} (e1={e1}, e2={e2}, ro={er})"
    ]
    pq = _per_qubit_depth(prog)
    avgcx = _avg_cx_distance(prog)
    return "\n".join([
        f"GateCounts:",
        *[f"  {k}: {v}" for k,v in counts.items()],
        f"Depth≈{depth}",
        f"PerQubitDepth={pq}",
        f"AvgCXDistance≈{avgcx:.2f}",
        f"EstFidelity≈{fidelity:.4f} (e1={e1}, e2={e2}, ro={er})"
    ])
