import numpy as np
import os
from typing import Dict, Optional
from ..config import get_max_qubits
from ..ir import Program

H  = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
SX = 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex)
SXDAG = SX.conj().T

def _rot(axis: str, theta: float):
    if axis=="x":
        return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*X
    if axis=="y":
        return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*Y
    if axis=="z":
        return np.array([[np.exp(-1j*theta/2),0],[0,np.exp(1j*theta/2)]], dtype=complex)
    raise ValueError("bad axis")

def _one_to_n(U, n, q):
    # Little-endian convention:
    # - Qubit 0 is least-significant bit in the state index
    # - Build operator from MSB -> LSB so that index (n-1-q) maps to the Kronecker position
    op = 1
    for idx in reversed(range(n)):  # idx = n-1, ..., 0
        op = np.kron(op, U if idx == q else np.eye(2))
    return op


def _h(n, q):
    return _one_to_n(H, n, q)

def _rx(n, theta, q):
    return _one_to_n(_rot("x", theta), n, q)

def _ry(n, theta, q):
    return _one_to_n(_rot("y", theta), n, q)

def _rz(n, theta, q):
    return _one_to_n(_rot("z", theta), n, q)

def _cx(n, c, t):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        j = i ^ (1<<t) if b[c]==1 else i
        U[j, i] = 1.0
    return U

def _cz(n, c, t):
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        if b[c] == 1 and b[t] == 1:
            U[i,i] = -1
    return U


def _controlled_single_qubit(U: np.ndarray, n: int, c: int, t: int):
    """Embed a controlled single-qubit gate for arbitrary control/target."""
    dim = 2**n
    mat = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        bits = [(col >> k) & 1 for k in range(n)]
        if bits[c] == 0:
            mat[col, col] = 1.0
            continue
        initial = bits[t]
        for new_state in (0, 1):
            new_bits = bits.copy()
            new_bits[t] = new_state
            row = sum(new_bits[k] << k for k in range(n))
            mat[row, col] += U[new_state, initial]
    return mat

def _s(n, q):
    """S gate (√Z)"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> q) & 1:
            U[i,i] = 1j
    return U

def _sdg(n, q):
    """S† gate (inverse of S)"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> q) & 1:
            U[i,i] = -1j
    return U

def _t(n, q):
    """T gate (4th root of Z)"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    phase = np.exp(1j * np.pi / 4)
    for i in range(dim):
        if (i >> q) & 1:
            U[i,i] = phase
    return U

def _tdg(n, q):
    """T† gate (inverse of T)"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    phase = np.exp(-1j * np.pi / 4)
    for i in range(dim):
        if (i >> q) & 1:
            U[i,i] = phase
    return U

def _crx(n, theta, c, t):
    """Controlled-RX gate"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    cos_theta = np.cos(theta/2)
    sin_theta = np.sin(theta/2)
    
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        if b[c] == 1:  # Only apply if control is |1⟩
            if b[t] == 0:
                # |0⟩ → cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
                U[i,i] = cos_theta
                U[i, i ^ (1 << t)] = -1j * sin_theta
            else:
                # |1⟩ → -i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩
                U[i,i] = cos_theta
                U[i, i ^ (1 << t)] = -1j * sin_theta
    return U

def _cry(n, theta, c, t):
    """Controlled-RY gate"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    cos_theta = np.cos(theta/2)
    sin_theta = np.sin(theta/2)
    
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        if b[c] == 1:  # Only apply if control is |1⟩
            if b[t] == 0:
                # |0⟩ → cos(θ/2)|0⟩ - sin(θ/2)|1⟩
                U[i,i] = cos_theta
                U[i, i ^ (1 << t)] = -sin_theta
            else:
                # |1⟩ → sin(θ/2)|0⟩ + cos(θ/2)|1⟩
                U[i,i] = cos_theta
                U[i, i ^ (1 << t)] = sin_theta
    return U

def _crz(n, phi, c, t):
    """Controlled-RZ gate"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    phase = np.exp(1j * phi / 2)
    
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        if b[c] == 1:  # Only apply if control is |1⟩
            if b[t] == 1:
                U[i,i] = np.conj(phase)  # |1⟩ → e^(-iφ/2)|1⟩
            else:
                U[i,i] = phase  # |0⟩ → e^(iφ/2)|0⟩
    return U

def _cswap(n, c, a, b):
    """Controlled-SWAP (Fredkin) gate"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    
    for i in range(dim):
        bv = [(i>>k)&1 for k in range(n)]
        if bv[c] == 1:  # Only swap if control is |1⟩
            if bv[a] != bv[b]:  # Only swap if qubits are different
                j = i ^ ((1 << a) | (1 << b))
                U[i,i] = 0
                U[j,j] = 0
                U[i,j] = 1
                U[j,i] = 1
    return U


def _cy(n, c, t):
    return _controlled_single_qubit(Y, n, c, t)


def _csx(n, c, t):
    return _controlled_single_qubit(SX, n, c, t)


def _cp(n, lam, c, t):
    return _cu1(n, lam, c, t)


def _two_qubit_gate(U: np.ndarray, n: int, q1: int, q2: int):
    if q1 == q2:
        raise ValueError("Two-qubit gate requires distinct qubits")
    dim = 2**n
    mat = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        bits = [(col >> k) & 1 for k in range(n)]
        basis_index = bits[q1] + 2 * bits[q2]
        for row_state in range(4):
            new_bits = bits.copy()
            new_bits[q1] = row_state & 1
            new_bits[q2] = (row_state >> 1) & 1
            row = sum(new_bits[k] << k for k in range(n))
            mat[row, col] += U[row_state, basis_index]
    return mat


def _iswap(n, q1, q2):
    iswap_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    return _two_qubit_gate(iswap_matrix, n, q1, q2)


def _ccz(n, c1, c2, t):
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for i in range(dim):
        bits = [(i >> k) & 1 for k in range(n)]
        if bits[c1] == 1 and bits[c2] == 1 and bits[t] == 1:
            U[i, i] = -1
    return U

def _rxx(n, theta, q1, q2):
    """Ising XX coupling gate"""
    # RXX(θ) = exp(-i θ/2 X⊗X)
    # Decomposed as: H(q1) • H(q2) • RZZ(θ) • H(q1) • H(q2)
    h1 = _one_to_n(H, n, q1)
    h2 = _one_to_n(H, n, q2)
    rzz = _rzz(n, theta, q1, q2)
    return h1 @ h2 @ rzz @ h1 @ h2

def _ryy(n, theta, q1, q2):
    """Ising YY coupling gate"""
    # RYY(θ) = exp(-i θ/2 Y⊗Y)
    # Decomposed as: RX(π/2, q1) • RX(π/2, q2) • RZZ(θ) • RX(-π/2, q1) • RX(-π/2, q2)
    rx_pi2_1 = _one_to_n(_rot("x", np.pi/2), n, q1)
    rx_pi2_2 = _one_to_n(_rot("x", np.pi/2), n, q2)
    rx_mpi2_1 = _one_to_n(_rot("x", -np.pi/2), n, q1)
    rx_mpi2_2 = _one_to_n(_rot("x", -np.pi/2), n, q2)
    rzz = _rzz(n, theta, q1, q2)
    return rx_pi2_1 @ rx_pi2_2 @ rzz @ rx_mpi2_1 @ rx_mpi2_2

def _rzz(n, theta, q1, q2):
    """Ising ZZ coupling gate"""
    # RZZ(θ) = exp(-i θ/2 Z⊗Z)
    dim = 2**n
    U = np.eye(dim, dtype=complex)

    same_phase = np.exp(-1j * theta / 2)   # when qubits are equal (00 or 11)
    diff_phase = np.exp(1j * theta / 2)    # when qubits differ (01 or 10)

    for i in range(dim):
        b = [(i >> k) & 1 for k in range(n)]
        if b[q1] == b[q2]:
            U[i, i] = same_phase
        else:
            U[i, i] = diff_phase
    return U

def _u1(n, lam, q):
    """U1 gate (phase shift)"""
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    phase = np.exp(1j * lam)
    
    for i in range(dim):
        if (i >> q) & 1:  # If qubit is |1⟩
            U[i,i] = phase
    return U

def _u2(n, phi, lam, q):
    """U2 gate"""
    # U2(φ,λ) = RZ(φ) • RY(π/2) • RZ(λ)
    rz_phi = _one_to_n(_rot("z", phi), n, q)
    ry_pi2 = _one_to_n(_rot("y", np.pi/2), n, q)
    rz_lam = _one_to_n(_rot("z", lam), n, q)
    return rz_phi @ ry_pi2 @ rz_lam

def _u3(n, theta, phi, lam, q):
    """U3 gate (general single-qubit unitary)"""
    # U3(θ,φ,λ) = RZ(φ) • RY(θ) • RZ(λ)
    rz_phi = _one_to_n(_rot("z", phi), n, q)
    ry_theta = _one_to_n(_rot("y", theta), n, q)
    rz_lam = _one_to_n(_rot("z", lam), n, q)
    return rz_phi @ ry_theta @ rz_lam

def _ccx(n, c1, c2, t):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        j = i ^ (1<<t) if b[c1]==1 and b[c2]==1 else i
        U[j, i] = 1.0
    return U

def _cu1(n, lam, c, t):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        b = [(i>>k)&1 for k in range(n)]
        if b[c]==1 and b[t]==1:
            U[i, i] = np.exp(1j*lam)
        else:
            U[i, i] = 1.0
    return U

class LocalSimulator:

    def __init__(self, noise: Optional[Dict[str, float]] = None):
        self.noise = noise if noise is not None else {}
        
    def execute(self, prog: Program, shots: int = 1024, seed: Optional[int] = None):
        import random
        import numpy as _np
        from collections import Counter

        # validate IR before execution to catch out-of-range qubits etc.
        try:
            prog.validate()
        except Exception:
            raise

        # Simulator limit: consult central config
        _max_q = get_max_qubits()
        if prog.n_qubits > _max_q:
            raise ValueError(f"Simulator limited to {_max_q} qubits (requested {prog.n_qubits})")

        # seed RNGs for reproducibility when provided
        if seed is not None:
            random.seed(seed)
            _np.random.seed(seed)

        n = prog.n_qubits
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        clbit_vals = [0] * prog.n_clbits
        measured: Dict[int, int] = {}
        # Track how many gates touch each qubit to approximate pre-measurement gate noise per shot
        oneq_gate_counts = [0] * n
        twoq_gate_counts = [0] * n

        # --- Apply all quantum operations ideally ---
        for op in prog.ops:
            if op.condition is not None and clbit_vals[op.condition] == 0:
                continue

            if op.name in ("h", "x", "y", "z", "rx", "ry", "rz", "sx", "sxdg"):
                if op.name == "h":  U1 = H
                elif op.name == "x": U1 = X
                elif op.name == "y": U1 = Y
                elif op.name == "z": U1 = Z
                elif op.name == "rx": U1 = _rot("x", op.params[0])
                elif op.name == "ry": U1 = _rot("y", op.params[0])
                elif op.name == "rz": U1 = _rot("z", op.params[0])
                elif op.name == "sx": U1 = SX
                elif op.name == "sxdg": U1 = SXDAG
                U = _one_to_n(U1, n, op.qubits[0])
                psi = U @ psi
                oneq_gate_counts[op.qubits[0]] += 1
                # Simple 1Q depolarizing channel after 1Q gates
                e1 = float(self.noise.get("oneq_error", 0.0))
                if e1 > 0:
                    # With probability e1, apply an X error to the acted qubit
                    if np.random.rand() < e1:
                        psi = _one_to_n(X, n, op.qubits[0]) @ psi

            elif op.name == "cx":
                U = _cx(n, op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                # Simple 2Q depolarizing channel after CX
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0:
                    if np.random.rand() < e2:
                        # Apply independent X errors on both qubits as a crude model
                        psi = _one_to_n(X, n, op.qubits[0]) @ _one_to_n(X, n, op.qubits[1]) @ psi

            elif op.name == "ccx":
                U = _ccx(n, op.qubits[0], op.qubits[1], op.qubits[2])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                twoq_gate_counts[op.qubits[2]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[2]) @ psi

            elif op.name == "ccz":
                U = _ccz(n, op.qubits[0], op.qubits[1], op.qubits[2])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                twoq_gate_counts[op.qubits[2]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    # model error as X on target qubit
                    psi = _one_to_n(X, n, op.qubits[2]) @ psi

            elif op.name == "swap":
                a, b = op.qubits
                U = _cx(n, a, b); psi = U @ psi
                U = _cx(n, b, a); psi = U @ psi
                U = _cx(n, a, b); psi = U @ psi
                twoq_gate_counts[a] += 3
                twoq_gate_counts[b] += 3
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, a) @ _one_to_n(X, n, b) @ psi

            elif op.name == "iswap":
                q1, q2 = op.qubits
                U = _iswap(n, q1, q2)
                psi = U @ psi
                twoq_gate_counts[q1] += 1
                twoq_gate_counts[q2] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))

                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, q1) @ _one_to_n(X, n, q2) @ psi

            elif op.name == "cu1":
                U = _cu1(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi

            elif op.name == "cp":
                U = _cp(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi
                
            # General single-qubit unitaries
            elif op.name in ("u1", "u2", "u3", "p"):
                if op.name == "u1" or op.name == "p":
                    U = _u1(n, op.params[0], op.qubits[0])
                elif op.name == "u2":
                    U = _u2(n, op.params[0], op.params[1], op.qubits[0])
                else:  # u3
                    U = _u3(n, op.params[0], op.params[1], op.params[2], op.qubits[0])
                psi = U @ psi
                oneq_gate_counts[op.qubits[0]] += 1
                e1 = float(self.noise.get("oneq_error", 0.0))
                if e1 > 0 and np.random.rand() < e1:
                    psi = _one_to_n(X, n, op.qubits[0]) @ psi

            # Phase gates
            elif op.name == "s":
                U = _s(n, op.qubits[0])
                psi = U @ psi
                oneq_gate_counts[op.qubits[0]] += 1
                e1 = float(self.noise.get("oneq_error", 0.0))
                if e1 > 0 and np.random.rand() < e1:
                    psi = _one_to_n(X, n, op.qubits[0]) @ psi
                
            elif op.name == "sdg":
                U = _sdg(n, op.qubits[0])
                psi = U @ psi
                oneq_gate_counts[op.qubits[0]] += 1
                e1 = float(self.noise.get("oneq_error", 0.0))
                if e1 > 0 and np.random.rand() < e1:
                    psi = _one_to_n(X, n, op.qubits[0]) @ psi
                
            elif op.name == "t":
                U = _t(n, op.qubits[0])
                psi = U @ psi
                oneq_gate_counts[op.qubits[0]] += 1
                e1 = float(self.noise.get("oneq_error", 0.0))
                if e1 > 0 and np.random.rand() < e1:
                    psi = _one_to_n(X, n, op.qubits[0]) @ psi
                
            elif op.name == "tdg":
                U = _tdg(n, op.qubits[0])
                psi = U @ psi
                oneq_gate_counts[op.qubits[0]] += 1
                e1 = float(self.noise.get("oneq_error", 0.0))
                if e1 > 0 and np.random.rand() < e1:
                    psi = _one_to_n(X, n, op.qubits[0]) @ psi
                
            # Controlled rotation gates
            elif op.name == "crx":
                U = _crx(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi
                
            elif op.name == "cry":
                U = _cry(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi
                
            elif op.name == "crz":
                U = _crz(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi
                
            # Multi-qubit gates
            elif op.name == "cswap":
                U = _cswap(n, op.qubits[0], op.qubits[1], op.qubits[2])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                twoq_gate_counts[op.qubits[2]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ _one_to_n(X, n, op.qubits[2]) @ psi
                
            elif op.name == "rxx":
                U = _rxx(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[0]) @ _one_to_n(X, n, op.qubits[1]) @ psi
                
            elif op.name == "ryy":
                U = _ryy(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[0]) @ _one_to_n(X, n, op.qubits[1]) @ psi
                
            elif op.name == "rzz":
                U = _rzz(n, op.params[0], op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[0]) @ _one_to_n(X, n, op.qubits[1]) @ psi
                
            elif op.name == "cz":
                U = _cz(n, op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[0]) @ _one_to_n(X, n, op.qubits[1]) @ psi

            elif op.name == "cy":
                U = _cy(n, op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi

            elif op.name == "csx":
                U = _csx(n, op.qubits[0], op.qubits[1])
                psi = U @ psi
                twoq_gate_counts[op.qubits[0]] += 1
                twoq_gate_counts[op.qubits[1]] += 1
                e2 = float(self.noise.get("twoq_error", 0.0))
                if e2 > 0 and np.random.rand() < e2:
                    psi = _one_to_n(X, n, op.qubits[1]) @ psi

            elif op.name == "measure":
                probs = np.abs(psi) ** 2
                outcome = _np.random.choice(range(2**n), p=probs)
                measured_bit = (outcome >> op.qubits[0]) & 1
                clbit_vals[op.clbits[0]] = measured_bit
                measured[op.clbits[0]] = op.qubits[0]
            else:
                raise ValueError(f"Unsupported op: {op.name}")

        probs = np.abs(psi) ** 2
        if not measured:
            return {}, {"shots": shots, "notes": "no measurement"}

        # --- Base sampling ---
        clbits = sorted(measured.keys())
        outcomes = Counter()
        readout_error = self.noise.get("readout_error", 0.0)

        # Pre-compute effective per-shot flip probabilities from gate noise
        e1 = float(self.noise.get("oneq_error", 0.0))
        e2 = float(self.noise.get("twoq_error", 0.0))
        p1 = [1.0 - (1.0 - e1) ** c for c in oneq_gate_counts]
        # Approximate probability that at least one 2Q gate experiences an error in this shot
        total_twoq_gates = max(0, int(sum(twoq_gate_counts) // 2))
        p2_event = 1.0 - (1.0 - e2) ** total_twoq_gates if total_twoq_gates > 0 else 0.0

        for _ in range(shots):
            # Sample a single outcome from the ideal probability distribution
            outcome = np.random.choice(range(2**n), p=probs)
            
            # Convert the outcome to a bitstring
            bitstring = [int(bit) for bit in format(outcome, f'0{n}b')]

            # Apply per-shot gate noise approximations first (pre-measurement effects)
            if e1 > 0.0 or e2 > 0.0:
                for i in range(n):
                    if p1[i] > 0.0 and np.random.rand() < p1[i]:
                        bitstring[i] = 1 - bitstring[i]
                # Two-qubit error: when an event happens, randomly apply IX/XI/XX on a qubit pair
                if p2_event > 0.0 and np.random.rand() < p2_event:
                    # choose a pair; weight choice by twoq_gate_counts to prefer active qubits
                    weights = np.array(twoq_gate_counts, dtype=float)
                    if weights.sum() == 0:
                        weights = None
                    q1 = np.random.choice(range(n), p=(weights/weights.sum()) if weights is not None else None)
                    # ensure a different second qubit
                    mask = np.ones(n, dtype=bool); mask[q1] = False
                    if weights is not None and mask.any():
                        w2 = weights.copy(); w2[~mask] = 0; w2 = w2 / (w2.sum() if w2.sum() > 0 else 1)
                        q2 = np.random.choice(range(n), p=w2 if w2.sum() > 0 else None)
                    else:
                        q2 = (q1 + 1) % n
                    err = np.random.choice(["IX","XI","XX"])  # simple depolarizing-like choices
                    if err in ("IX","XX"):
                        bitstring[q2] = 1 - bitstring[q2]
                    if err in ("XI","XX"):
                        bitstring[q1] = 1 - bitstring[q1]

            # Apply readout error
            if readout_error > 0:
                for i in range(n):
                    if np.random.rand() < readout_error:
                        bitstring[i] = 1 - bitstring[i] # Flip the bit

            # Format the final bitstring for the results
            s = []
            for c in clbits:
                s.append(str(bitstring[measured[c]]))
            key = "".join(reversed(s))
            outcomes[key] += 1

        meta = {
            "shots": shots,
            "n_qubits": n,
            "clbits": len(clbits),
            "noise": self.noise,
            "oneq_gate_counts": oneq_gate_counts,
            "twoq_gate_counts": twoq_gate_counts,
            "effective_flip_probabilities": {"oneq": p1, "twoq_event": p2_event},
        }
        return dict(outcomes), meta