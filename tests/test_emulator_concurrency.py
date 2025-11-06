import threading
from qx.sim.zenaquantum_alpha import ZenaQuantumAlphaSimulator
from qx.ir import Program, Op


def _make_prog():
    # 2-qubit program with one CX and a measure
    p = Program(n_qubits=2, ops=[Op("h", (0,)), Op("cx", (0, 1)), Op("measure", (0,), (), (0,))], n_clbits=1)
    return p


def test_alpha_semaphore_enforces_lanes():
    # durations in ns; set CX large enough to create ~0.2s runtime per job
    durations = {"h": 10, "cx": 200000, "measure": 10}
    sim = ZenaQuantumAlphaSimulator(noise={}, durations=durations, base_latency=0.01, queue_jitter=(0.0, 0.0), lanes=2)

    prog = _make_prog()

    def worker():
        sim.execute(prog, shots=10, seed=42)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    # Ensure at no point more than 2 concurrent runs were active
    assert sim._max_active <= 2
