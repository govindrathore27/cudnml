"""Clean-venv import + entrypoint-call smoke test for the abi3 wheel (TOOL-03).

Run inside a FRESH virtualenv that has ONLY the built
`sylva_cuda-*-cp310-abi3-win_amd64.whl` installed (no repo source on
PYTHONPATH). It proves the final two links of the Phase-1 walking-skeleton
chain: the dynamic-loading wheel resolves the CUDA driver at runtime, and the
full Rust -> CUDA(NVRTC) -> Python(abi3) path works from a clean install.

What it checks:
  1. `import sylva_cuda` succeeds and the module resolves to the venv's
     site-packages (the installed wheel), NOT the repo source tree.
  2. The `#[pyfunction]` entrypoint `run_vector_add` launches the NVRTC-compiled
     CUDA kernel on the GPU and returns the correct elementwise sum.

Exits 0 on success after printing `OK: run_vector_add correct`; raises (non-zero
exit) on any failure — no silent fallback.
"""

import sys

import sylva_cuda

# 1. Prove the module came from the installed wheel, not the source tree.
module_path = sylva_cuda.__file__
print(f"sylva_cuda.__file__ = {module_path}")
if "site-packages" not in module_path:
    raise SystemExit(
        f"FAIL: sylva_cuda did not load from site-packages (got {module_path!r}); "
        "the import may be resolving the repo source instead of the installed wheel"
    )

# 2. Call the entrypoint on a tiny vector and assert exact correctness.
a = [1.0, 2.0, 3.0, 4.0]
b = [10.0, 20.0, 30.0, 40.0]
expected = [x + y for x, y in zip(a, b)]

result = sylva_cuda.run_vector_add(a, b)
print(f"run_vector_add({a}, {b}) = {result}")

if list(result) != expected:
    raise SystemExit(
        f"FAIL: run_vector_add returned {list(result)!r}, expected {expected!r}"
    )

print("OK: run_vector_add correct")
sys.exit(0)
