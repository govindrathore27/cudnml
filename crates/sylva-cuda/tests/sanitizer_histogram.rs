//! Standalone `compute-sanitizer` target for the privatized histogram (TOOL-02).
//!
//! This is a deliberately minimal, isolated test binary: it launches
//! `run_histogram` exactly ONCE on a fixed input and asserts correctness. It
//! exists so `compute-sanitizer` (memcheck / racecheck / synccheck / initcheck)
//! has a single clean histogram launch to wrap, with no other GPU work to
//! confuse the report.
//!
//! The histogram kernel was NVRTC-compiled with `-lineinfo` (see
//! `nvrtc_launch::compile_for_sm89`), so any hazard this binary surfaces carries
//! a kernel source-line attribution (Pitfall 4). A reported hazard is a KERNEL
//! BUG to fix (shared-mem zero-init / `__syncthreads` placement), NOT a
//! toolchain kill (D-05) — fix the kernel and re-run until every tool reports
//! `ERROR SUMMARY: 0 errors`.
//!
//! Run (full paths required on this host — compute-sanitizer is not on PATH):
//! ```text
//! cargo test -p sylva-cuda --test sanitizer_histogram --no-run
//! compute-sanitizer --tool memcheck  <target.exe>
//! compute-sanitizer --tool racecheck <target.exe>
//! compute-sanitizer --tool synccheck <target.exe>
//! compute-sanitizer --tool initcheck <target.exe>
//! ```

/// Fixed input size for the sanitizer launch. Small enough that all four
/// sanitizer tools run quickly, large enough to span multiple blocks
/// (n / 256 > 1) so the privatized shared-mem path + two-level reduction are
/// actually exercised under the sanitizer.
const SANITIZER_N: usize = 100_000;

/// Launch the privatized histogram exactly once and verify it against a CPU
/// reference. The assertion makes the binary fail loudly if the launch is
/// wrong; the compute-sanitizer wrapper independently checks for memory/race/
/// sync/init hazards on the same single launch.
#[test]
fn sanitizer_histogram_single_launch() {
    // Deterministic bins in [0, 256) (no RNG) — same mixing scheme as the
    // correctness test, so the sanitizer target matches the proven kernel path.
    let bins: Vec<u8> = (0..SANITIZER_N)
        .map(|i| ((i * 31 + 7) % 256) as u8)
        .collect();

    let mut expected = vec![0u32; 256];
    for &b in &bins {
        expected[b as usize] += 1;
    }

    let got =
        sylva_cuda::run_histogram(&bins).expect("histogram launch (sanitizer target) must succeed");

    assert_eq!(
        got, expected,
        "sanitizer-target histogram must match CPU reference"
    );
}
