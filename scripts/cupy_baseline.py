"""CuPy vector-add baseline for the Phase-1 microbench (SC-6, D-06).

A **feasibility sanity check, NOT an algorithm speed claim** (PITFALLS 1/13).
This is the independent reference the cudarc + NVRTC microbench
(`crates/sylva-cuda/benches/microbench.rs`) is compared against. It obeys the
SAME fairness protocol so the comparison is fair:

  - WARMUP: 50 discarded launches (excludes CuPy's first-call JIT / autotune).
  - MEDIAN of >= 10 timed runs (not mean).
  - Device-event timing (cupy.cuda.Event), not host time.
  - CORRECTNESS FIRST: assert max_abs_err(out, a+b) == 0 before any timing.
  - Same FIXED 1e7-element f32 vector; bytes = 3 * 4 * N; GB/s = bytes / time.
  - Same GPU / driver / CUDA as the Sylva path (recorded in VERSIONS.md).

CuPy has NO cp314 Windows wheel (verified 01-RESEARCH.md Pitfall 2), so this MUST
run in a separate Python 3.12 venv with `cupy-cuda12x` installed:

    py -3.12 -m venv .venv-cupy
    .venv-cupy\\Scripts\\pip install cupy-cuda12x
    .venv-cupy\\Scripts\\python scripts\\cupy_baseline.py

Both processes hit the same RTX 4060 Ti, so the cross-process comparison is fair.
Prints `cupy_version`, `launch_overhead_us_per_launch`, `throughput_gbps`.
"""

import statistics

import cupy as cp

# Fairness constants — MUST match crates/sylva-cuda/benches/microbench.rs.
WARMUP_LAUNCHES = 50
TIMED_RUNS = 10
LAUNCHES_PER_OVERHEAD_RUN = 1_000
THROUGHPUT_N = 10_000_000
BYTES_PER_F32 = 4
ARRAYS_TOUCHED = 3  # two reads (a, b) + one write (out)

# A raw elementwise add kernel so the baseline measures a launch of the SAME
# shape of op as the Sylva vector_add (not a fused cupy ufunc that may differ).
_vector_add = cp.ElementwiseKernel(
    "float32 a, float32 b", "float32 out", "out = a + b", "vector_add_baseline"
)


def _elapsed_ms(start: "cp.cuda.Event", end: "cp.cuda.Event") -> float:
    """Device-event elapsed time in ms (CuPy returns ms)."""
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end)


def assert_correct() -> None:
    """Assert out == a + b exactly BEFORE any timing (correctness-first)."""
    n = 4096
    a = cp.arange(n, dtype=cp.float32)
    b = (2 * cp.arange(n, dtype=cp.float32)).astype(cp.float32)
    out = _vector_add(a, b)
    cp.cuda.Stream.null.synchronize()
    max_abs_err = float(cp.max(cp.abs(out - (a + b))))
    if max_abs_err != 0.0:
        raise SystemExit(f"FAIL: cupy vector_add incorrect, max_abs_err={max_abs_err}")
    print("correctness: max_abs_err(out, a+b) = 0  [asserted before timing]")


def measure_launch_overhead_us() -> float:
    """us/launch: warmup, then median of TIMED_RUNS event-timed tight loops."""
    a = cp.asarray([1.0], dtype=cp.float32)
    b = cp.asarray([2.0], dtype=cp.float32)
    out = cp.empty_like(a)

    for _ in range(WARMUP_LAUNCHES):
        _vector_add(a, b, out)
    cp.cuda.Stream.null.synchronize()

    per_launch_us = []
    for _ in range(TIMED_RUNS):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(LAUNCHES_PER_OVERHEAD_RUN):
            _vector_add(a, b, out)
        end.record()
        ms = _elapsed_ms(start, end)
        per_launch_us.append((ms * 1000.0) / LAUNCHES_PER_OVERHEAD_RUN)
    return statistics.median(per_launch_us)


def measure_throughput_gbps() -> float:
    """GB/s over THROUGHPUT_N f32: warmup, then median of TIMED_RUNS runs."""
    n = THROUGHPUT_N
    a = (cp.arange(n, dtype=cp.float32) % 1000).astype(cp.float32)
    b = ((cp.arange(n, dtype=cp.float32) + 1) % 1000).astype(cp.float32)
    out = cp.empty_like(a)

    for _ in range(WARMUP_LAUNCHES):
        _vector_add(a, b, out)
    cp.cuda.Stream.null.synchronize()

    bytes_moved = ARRAYS_TOUCHED * BYTES_PER_F32 * n
    gbps = []
    for _ in range(TIMED_RUNS):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        _vector_add(a, b, out)
        end.record()
        s = _elapsed_ms(start, end) / 1000.0
        gbps.append(bytes_moved / s / 1e9)
    return statistics.median(gbps)


def main() -> None:
    print("== CuPy baseline (feasibility sanity check, NO algorithm speed claim) ==")
    print(f"cupy_version = {cp.__version__}")
    dev = cp.cuda.Device()
    print(f"device = {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()}")

    assert_correct()
    overhead_us = measure_launch_overhead_us()
    gbps = measure_throughput_gbps()

    print(
        f"warmup={WARMUP_LAUNCHES} launches; median of {TIMED_RUNS} runs; "
        f"overhead loop={LAUNCHES_PER_OVERHEAD_RUN} launches/run; N={THROUGHPUT_N}"
    )
    print(f"launch_overhead_us_per_launch = {overhead_us:.4f}")
    print(f"throughput_gbps = {gbps:.2f}")
    print("(feasibility sanity check, no algorithm speed claim)")


if __name__ == "__main__":
    main()
