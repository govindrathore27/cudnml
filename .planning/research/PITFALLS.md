# Pitfalls Research

**Domain:** GPU-native tree-ensemble library (Extra Trees + Random Forest), Rust core + CUDA kernels + Python (PyO3/maturin), single NVIDIA GPU, dense float32, deterministic mode, exact tree SHAP. Solo dev on Windows 11.
**Researched:** 2026-06-19
**Confidence:** MEDIUM-HIGH (technical claims cross-checked against rust-cuda docs, cuML API, TreeSHAP literature; strategic claims grounded in the project's own feasibility studies)

> **How to read this file.** This project is *validation-gated*: PROJECT.md defines success as a **pre-registered benchmark crossover surface (n×d)** measured end-to-end, with explicit kill criteria. Most pitfalls below are not "bugs to fix later" — they are **conditions that, if true, mean the project should pivot or stop.** The single most important discipline is: **measure the end-to-end crossover before building anything broad.** Every critical pitfall ties back to that.

---

## Critical Pitfalls

### Pitfall 1: H2D transfer + data conversion swamps the kernel win (the "fast kernel, slow library" trap)

**What goes wrong:**
You write a histogram/split kernel that beats sklearn *on-device*, but the end-to-end `fit()` — including pandas→numpy→float32 coercion, host-to-device copy over PCIe, and result copy-back — is **slower than sklearn on CPU**. The benchmark that matters (PROJECT.md: "measured end-to-end including transfers") loses despite a "winning" kernel. This is the most common way GPU ML libraries die: the demo microbenchmark wins, the real workload loses.

**Why it happens:**
PCIe gen4 x16 is ~25 GB/s; HBM on the GPU is 1–3 TB/s. A single H2D copy of an (n×d) float32 matrix can cost more wall-clock than the entire histogram pass if the kernel is efficient. Developers benchmark `kernel_time` in isolation (the satisfying number) instead of `time-from-numpy-array-to-fitted-model`. Float64→float32 conversion and non-contiguous arrays add hidden copies.

**How to avoid:**
- **Pre-register the benchmark to time the *public API call*, not the kernel.** Clock `model.fit(X, y)` from a host numpy array, cold and warm. Include the conversion and transfer in every reported number. Bake this into the harness in the first measurement phase — never retrofit it.
- Amortize transfer: forests reuse the same (n×d) matrix across *every* tree and *every* node. Transfer **once**, keep resident on device for the whole fit. If your design re-transfers per tree, that is an architectural bug.
- Make `execution_report_` (already a requirement) **surface bytes transferred and transfer time** so the cost is never invisible to the user or to you.
- Use pinned (page-locked) host memory for the one big transfer; overlap with first kernel launch via streams once correctness is proven.

**Warning signs:**
- Your benchmark script reports `kernel_ms` but not `fit_ms`.
- Speedup shrinks dramatically when you switch from "data already on GPU" to "data starts as numpy."
- `nsys` shows memcpy time ≥ compute time.

**Phase to address:** **Benchmark/validation phase (must come first, before kernel build-out).** This is the crossover gate. End-to-end timing is the definition of success, not a later optimization.

---

### Pitfall 2: Small-data crossover — CPU wins below threshold, and you never find the threshold honestly

**What goes wrong:**
For small/medium (n×d), sklearn on a modern multicore CPU (or oneDAL) is *faster* than any GPU path because of kernel-launch latency, transfer fixed cost, and GPU underutilization. If you market "GPU is faster" without publishing the **crossover surface**, every user with a 50k-row dataset (the common case) gets a slower library and concludes the project is snake oil.

**Why it happens:**
GPUs have high fixed cost per operation (launch latency ~5–20µs, transfer setup, allocation). That cost only amortizes when (samples × features × bins × trees) is large enough. sklearn's `n_jobs=-1` is genuinely fast and embarrassingly parallel across trees. Developers test on one big dataset, see a win, and generalize.

**How to avoid:**
- **The crossover surface IS the deliverable**, per PROJECT.md. Produce a 2D map over (n, d) showing where GPU wins, where CPU wins, and the boundary. Ship it as documentation, not marketing.
- Implement the **honest dispatch** requirement (`device="auto"` + `execution_report_`): below the measured crossover, `auto` should *recommend or route to CPU* and say why. The differentiator is honesty, not universal speed.
- Pick the *strongest* CPU baseline (sklearn `n_jobs=-1` **and** oneDAL/sklearnex), not a crippled single-thread one. Beating a weak baseline is benchmark dishonesty (see Pitfall 13).

**Warning signs:**
- You only ever benchmark one dataset size.
- The CPU baseline runs single-threaded "to be fair."
- `auto` dispatch always picks GPU regardless of input size.

**Phase to address:** **Benchmark/validation phase**, then encoded into the dispatch-policy phase.

---

### Pitfall 3: Shrinking-node GPU underutilization at deep tree levels

**What goes wrong:**
The root split processes all n rows — massively parallel, GPU-saturating. By depth 12, nodes hold dozens of rows each; thousands of tiny nodes each launch underutilized kernels, occupancy collapses, and the GPU spends its time on launch overhead for nodes that a CPU would finish in nanoseconds. The deep-tree tail dominates wall-clock and erases the shallow-level win.

**Why it happens:**
Tree training has an inverted parallelism profile: maximal work at the top, exponentially shrinking per-node work toward the leaves. A naive "one kernel launch per node" design (the obvious first implementation) is pathological here. cuML solves this with a **breadth-first, whole-level** build (one kernel processes an entire tree level across all its nodes at once).

**How to avoid:**
- **Architect for breadth-first, level-at-a-time tree construction from day one** (process all nodes at a given depth in one fused kernel keyed by node-id), not depth-first per-node recursion. Retrofitting this is a near-rewrite.
- Set a **node-size / level cutover**: below a row-count threshold, finish subtrees on CPU or with a single small "leaf-finishing" kernel rather than per-node launches.
- Extra Trees mitigates this somewhat (random thresholds = cheaper per-node work), which is *why* it's the chosen wedge — lean into it.

**Warning signs:**
- `fit` time scales super-linearly with `max_depth`.
- `nsys` shows thousands of tiny kernel launches in the back half of training.
- GPU occupancy (Nsight Compute) drops toward zero as depth increases.

**Phase to address:** **Core kernel architecture phase** — the tree-build strategy decision. Must be settled in the architecture spec, not discovered during implementation.

---

### Pitfall 4: Atomic contention in histogram construction

**What goes wrong:**
The histogram kernel uses global-memory `atomicAdd` to accumulate per-(feature, bin) gradient/count sums. With many threads hitting the same hot bins (common — real data is skewed), atomics serialize and the "parallel" histogram runs nearly serially on the hot bins. Throughput craters and the bandwidth-bound advantage evaporates.

**Why it happens:**
Histogram building is the hot path (PROJECT.md correctly identifies tree training as "bandwidth- and atomic-contention-bound"). The naive global-atomic approach is the textbook first attempt and the textbook performance trap.

**How to avoid:**
- **Privatized histograms in shared memory:** each thread block builds a private histogram in `__shared__`, then one reduced atomic merge to global. This is the standard GPU histogram pattern and should be the baseline, not an optimization.
- Tune bin count (e.g., 128–256) to fit shared-memory budget; this also constrains the float-determinism design (Pitfall 5).
- Consider per-warp privatization for very hot distributions.
- Profile with **Nsight Compute** specifically for atomic stalls / shared-memory bank conflicts.

**Warning signs:**
- Histogram kernel throughput far below memory-bandwidth roofline.
- Performance is highly data-dependent (skewed columns much slower than uniform).
- Nsight shows high "long scoreboard" / atomic stalls.

**Phase to address:** **Core kernel architecture phase** (histogram kernel design). Couple the design decision with Pitfall 5 — privatization strategy and determinism are the same decision.

---

### Pitfall 5: Nondeterministic float atomic reductions break the determinism contract

**What goes wrong:**
`deterministic=True` is a headline differentiator and a stated requirement, but `atomicAdd` on floats accumulates in **nondeterministic order** (thread scheduling varies run to run). Floating-point addition is non-associative, so the same model trained twice gives **bit-different split scores → different splits → different trees**. The flagship feature silently doesn't work, and you only notice when a user files a reproducibility bug.

**Why it happens:**
Float non-associativity + nondeterministic GPU thread order is a fundamental, well-known GPU hazard. It's invisible in accuracy tests (results are "close") and only shows up under exact bit-comparison — which most test suites don't do.

**How to avoid:**
- **Determinism requires an ordering or integer-accumulation strategy, not float atomics.** Options: (a) fixed-order segmented reduction / prefix-sum instead of atomics; (b) integer/fixed-point accumulation of histograms (counts are exact integers; sums can be scaled to fixed-point then converted once); (c) deterministic parallel reduction trees with a fixed reduction order.
- The **prefix-sum split-evaluation kernel** (blueprint fix #7, already in scope) is a natural deterministic primitive — exploit it; scans have well-defined ordering.
- **Test determinism with exact bit-equality**, not `np.allclose`. Train twice, assert identical serialized trees. Add this to the differential-test suite as a first-class invariant.
- Document the perf cost of `deterministic=True` honestly (it will be slower — that's expected and a requirement already acknowledges it).

**Warning signs:**
- Two runs with the same seed produce models that score identically but aren't bit-identical.
- Determinism is tested with tolerance-based comparison instead of exact equality.
- The histogram path uses float `atomicAdd` and there's no separate deterministic path.

**Phase to address:** **Histogram/kernel architecture phase** (decide deterministic accumulation primitive up front) + **differential-testing phase** (bit-exact determinism test).

---

### Pitfall 6: Matching sklearn EXACTLY — tie-breaking and RNG semantics, not just "similar accuracy"

**What goes wrong:**
You target "sklearn parity" but interpret it as "similar accuracy." Users (and your own differential tests) then find that for the *same seed* your trees differ from sklearn's: different feature chosen on ties, different threshold, different sample draw. The "sklearn-compatible" claim is false in the strict sense, and you can't use sklearn as a bit-level correctness oracle — undermining the entire CPU-reference-as-oracle strategy.

**Why it happens:**
sklearn's exactness lives in details that are easy to miss: (1) **tie-breaking** — which feature/threshold wins when impurity gains are equal (sklearn has specific `<=` vs `<` and "first/best" conventions); (2) **RNG semantics** — sklearn's `random_state` drives a specific sequence of feature subsampling and (for Extra Trees) threshold draws via a specific PRNG with specific draw order; (3) Extra Trees draws thresholds *uniformly between min and max of the feature within the node* — reproducing that draw order on GPU (where parallelism reorders everything) is genuinely hard. PROJECT.md is explicit: "match algorithmic semantics (not merely similar accuracy)."

**How to avoid:**
- **Decide the parity contract precisely and write it down before coding:** Is the goal (a) bit-identical trees to sklearn given a seed, or (b) statistically equivalent models (same expected accuracy/distribution) with your *own* documented, reproducible RNG? Bit-identical-to-sklearn-on-GPU is likely **infeasible** (parallel threshold draws can't replay sklearn's serial PRNG order); pursuing it can sink the project. The pragmatic, defensible contract is **(b): your own deterministic, documented RNG + a differential test that asserts statistical equivalence (accuracy within CI, split-distribution KS test), plus exact self-reproducibility.**
- Use the **CPU reference backend as the oracle for *your own* semantics** (bit-identical CPU vs GPU for a given seed), and use sklearn as a *distributional* oracle (equivalent accuracy/behavior), not a bit oracle.
- Nail tie-breaking conventions explicitly in the CPU reference and replicate them on GPU.

**Warning signs:**
- Spec says "sklearn-compatible" without defining whether that means bit-identical or distributional.
- Differential tests use only accuracy, never split-structure comparison.
- You're sinking days trying to replay sklearn's exact PRNG draw order on parallel hardware.

**Phase to address:** **Spec/architecture phase** (define parity contract) + **CPU reference phase** (oracle) + **differential-testing phase**. This contract decision is high-leverage — get it wrong and you chase an infeasible goal.

---

### Pitfall 7: Missing-value (NaN) routing is undefined or divergent

**What goes wrong:**
NaNs in `X` either crash the kernel, get silently bucketed into bin 0, or route differently on CPU vs GPU — producing different predictions and breaking parity. sklearn's RandomForest/ExtraTrees have specific NaN-handling behavior (historically: error; recent versions: learned/default routing), and not matching it is a correctness divergence that surfaces only on real (dirty) data.

**Why it happens:**
Synthetic benchmark data has no NaNs, so the gap is invisible until a real user passes messy tabular data. Quantization/binning code paths are written assuming finite floats.

**How to avoid:**
- **Decide NaN policy explicitly and early**, matching the targeted sklearn version's behavior. For MVP dense-float32, the simplest defensible choice: error on NaN (matches older sklearn, fails loud, no silent wrong answer) — consistent with the project's "no silent fallback" ethos.
- Make CPU and GPU paths share the **same** NaN routing logic (test with NaN-containing fixtures in differential tests).
- Surface NaN handling in `execution_report_`.

**Warning signs:**
- No test fixture contains NaN.
- Binning code uses `float` comparisons that NaN silently fails (NaN comparisons are always false → silent misrouting).

**Phase to address:** **CPU reference + differential-testing phase**; policy decided in **spec phase**.

---

### Pitfall 8: The exact-SHAP / "WOODELF-HD" trap — building a novel method to solve an already-polynomial problem

**What goes wrong:**
The project proposes **exact high-depth tree SHAP via a "WOODELF-HD" vectorized Strassen-like scheme + UFDP path compression** (blueprint fix #2). This risks (a) being **unverifiable / unpublished** — a method with no peer-reviewed reference is a research bet, not an engineering task; (b) chasing an **O(3^D) blowup** that, on investigation, applies to the *wrong* problem; (c) numerical instability from Strassen-style recombination of small floats.

**Why it happens — and the key correction:**
Verified against the TreeSHAP literature: **standard path-dependent TreeSHAP (Lundberg et al., 2020) is already polynomial — O(T·L·D²)** — and exact **interventional** TreeSHAP is also polynomial via dynamic programming. The exponential / **O(3^D)** cost arises specifically for **Shapley *interaction* values** (interaction indices enumerate per-feature ternary states → 3^D-flavored blowup), **not** for standard SHAP attributions. So:
- If the goal is **exact standard SHAP values**, the published Lundberg algorithm already does it in polynomial time. A novel Strassen-like scheme would be **re-solving a solved problem** — pure scope risk with no payoff. Implement (or port) the known polynomial algorithm.
- If the goal is **exact high-order SHAP *interactions* at high depth**, *that* is the genuinely hard, possibly-exponential case — and "WOODELF-HD" would be an unproven research contribution, not an MVP feature.

**How to avoid:**
- **Spike/feasibility-gate the SHAP method before committing.** First answer: "what exactly are we computing — attributions or interactions?" If attributions: ship the known polynomial TreeSHAP (well-trodden, GPU-parallelizable across trees/rows). If interactions: treat WOODELF-HD as a **research spike with a kill date**, not a roadmap deliverable.
- **Require a citation or a written proof** before building any "novel" exact-SHAP kernel. No reference + no proof = not in MVP scope.
- Validate every SHAP output against the reference `shap.TreeExplainer` (which implements Lundberg's polynomial algorithm) for exactness within float tolerance — this is your correctness oracle and it already exists.
- Defend against numerical instability: avoid Strassen-style recombination unless a stability analysis exists; the standard DP TreeSHAP is numerically benign by comparison.

**Warning signs:**
- "WOODELF-HD" appears in a plan with no paper, no reference implementation, and no complexity proof.
- The complexity argument cites O(3^D) for *standard* SHAP values (that's the interaction case — a red flag the premise is muddled).
- SHAP outputs aren't checked against `shap.TreeExplainer`.

**Phase to address:** **Dedicated SHAP feasibility spike (pre-roadmap or earliest research phase), with explicit kill criteria.** Do NOT let exact-SHAP become a load-bearing MVP promise until the spike resolves whether you're building the easy (solved) case or the hard (research) case.

---

### Pitfall 9: Rust↔CUDA toolchain fragility on Windows (the top technical risk, per PROJECT.md)

**What goes wrong:**
The kernel-authoring path is **explicitly unresolved** in PROJECT.md and named the single biggest technical risk. The three families have very different failure modes on Windows:
- **rust-cuda (`rustc_codegen_nvvm`, Rust→PTX):** supports native Windows but is **nightly-pinned** (requires a specific `rust-toolchain.toml`), needs CUDA 12.x/13.x, MSVC C++ build tools, and `nvvm\bin\x64` on `PATH` or it fails with "couldn't load codegen backend." Pinned-nightly means the whole project is hostage to a moving toolchain; a Rust nightly bump can break the build.
- **cudarc / cust (host-side, load hand-written CUDA C/PTX):** more stable host side, but you author kernels in CUDA C compiled by `nvcc` — a separate, well-supported toolchain. You lose "all Rust" but gain stability and standard CUDA tooling (compute-sanitizer, Nsight) working normally.

**Why it happens:**
Rust-GPU tooling is young and nightly-coupled. Windows adds PATH/libNVVM/MSVC fragility on top. A solo dev can burn the entire budget fighting the build instead of writing kernels.

**How to avoid:**
- **Resolve the kernel-authoring path in a dedicated spike before any kernel work** (PROJECT.md already mandates this). Decision criteria: build reproducibility, Windows-native success, debugging/sanitizer support, and *time-to-first-working-kernel*.
- **Strong recommendation: prefer `cudarc` + hand-written CUDA C for the MVP.** Rationale: kernels compiled by `nvcc` work with the *full, mature* CUDA tooling (compute-sanitizer, Nsight Compute/Systems, `cuda-gdb`), which directly mitigates Pitfalls 4, 5, 10, and 11. Rust→PTX (rust-cuda) couples you to nightly churn and immature GPU debugging exactly where you most need mature tools. Keep Rust for the host-side orchestration, memory management, and Python boundary — its actual strengths here.
- **Pin everything:** exact CUDA toolkit version, exact Rust toolchain, exact maturin/PyO3 versions, in-repo. Document the WSL fallback (PROJECT.md allows it) as the escape hatch if Windows-native blocks you.

**Warning signs:**
- "couldn't load codegen backend" (rust-cuda PATH/libNVVM issue).
- A `cargo update` or nightly bump breaks the GPU build.
- compute-sanitizer/Nsight can't introspect your kernels (a sign the Rust→PTX path is fighting the tooling).

**Phase to address:** **Toolchain spike phase (mandatory, before kernel build).** This is a named gating decision in PROJECT.md.

---

### Pitfall 10: maturin / abi3 / PyO3 wheel packaging on Windows with a CUDA dependency

**What goes wrong:**
You get kernels working locally but can't produce a `pip install`-able wheel that actually runs on another machine: the wheel hardcodes a CUDA toolkit path, abi3 (stable-ABI) constraints conflict with how you link CUDA, or the wheel is tied to one CUDA minor version and breaks on the user's driver. Packaging — not kernels — becomes the thing that prevents shipping.

**Why it happens:**
GPU Python packaging is genuinely hard (RAPIDS ships its own conda channels and CUDA-version-suffixed wheels precisely because pip+CUDA is painful). A solo dev underestimates this and discovers it at the worst time — right before release.

**How to avoid:**
- **Decide the CUDA-linkage model early:** link against the CUDA *driver* API (via cudarc/cust — driver is forward-compatible across toolkits) rather than statically baking a toolkit, and load PTX/cubin at runtime. This decouples the wheel from the user's exact toolkit version.
- Prototype the **full `maturin build` → install-in-clean-env → import → run-a-kernel** loop in an *early* phase, not at the end. A "hello-GPU" wheel that imports and runs one kernel on a clean machine is a critical early milestone.
- Be explicit about abi3: abi3 wheels are attractive (one wheel per platform across Python versions) but verify it doesn't conflict with your linking; if it does, ship per-Python-version wheels — don't fight abi3.
- Mirror RAPIDS's reality: you may need a CUDA-version-suffixed wheel naming scheme. Accept this rather than pretending one wheel fits all.

**Warning signs:**
- The wheel only runs on your dev machine.
- Import works but the first kernel launch fails with a CUDA version/driver mismatch on a clean machine.
- You haven't tried installing your wheel in a fresh environment until late.

**Phase to address:** **Early packaging spike** (a "hello-GPU wheel" milestone), revisited at **release-prep phase**.

---

### Pitfall 11: Debugging GPU code from Rust + compute-sanitizer usability

**What goes wrong:**
A correctness bug (race, out-of-bounds in shared-memory histogram, uninitialized read) produces *plausible-but-wrong* numbers. You can't set a breakpoint in a Rust→PTX kernel, `compute-sanitizer` can't symbolicate it, and you spend days bisecting by hand. GPU bugs are brutal precisely because they're often silent and nondeterministic.

**Why it happens:**
Rust→PTX kernels don't integrate with the mature CUDA debug/sanitizer toolchain the way `nvcc`-compiled CUDA C does. Memory races in privatized-histogram code (Pitfall 4) are the classic silent corruptor.

**How to avoid:**
- This is a **decisive argument for the cudarc + CUDA-C path** (Pitfall 9): `nvcc`-compiled kernels work with `compute-sanitizer` (memcheck, racecheck, initcheck, synccheck), `cuda-gdb`, and Nsight out of the box.
- **Run `compute-sanitizer --tool racecheck` and `memcheck` in CI** (or in your local pre-commit) on every kernel against small fixtures. Catch races before they corrupt a benchmark.
- Keep the **CPU reference backend as the correctness oracle**: any GPU output that diverges from the bit-exact CPU reference (for a given seed) is a bug — this turns silent GPU corruption into a loud, localizable test failure. This is the single most valuable debugging asset; build it early.

**Warning signs:**
- compute-sanitizer reports "no source correlation" on your kernels.
- Bugs only reproduce at certain block sizes or occupancy (classic race signature).
- You're debugging GPU numbers without a CPU oracle to diff against.

**Phase to address:** **Toolchain spike** (ensure sanitizer works with chosen path) + **CPU reference phase** (oracle) + **CI phase**.

---

### Pitfall 12: The matmul-premise trap — chasing Tensor Cores for a workload that has no GEMM

**What goes wrong:**
Effort (or a future contributor, or a reviewer's suggestion) drifts toward "use Tensor Cores / reformulate histograms as matmul / SpMM kernels" to chase headline FLOPS. This is **explicitly debunked in PROJECT.md**: tree training is bandwidth- and atomic-bound; there is **no GEMM in the hot path**; Tensor Cores sit idle. Time spent here is pure waste and produces no speedup.

**Why it happens:**
The pervasive (wrong) intuition "GPUs are fast because matmul is fast." It's seductive, it sounds sophisticated, and the "7 fixes" blueprint already contained SpMM/Tensor-Core reformulation ideas (fix #1, Tensor-Core histograms) that PROJECT.md correctly moved to Out of Scope.

**How to avoid:**
- **Treat Out-of-Scope items #1 and "Tensor-Core reformulation of histograms" as hard fences.** Any plan that reintroduces them must re-justify against the roofline analysis (the workload is bandwidth/atomics-bound — prove a GEMM formulation helps before spending an hour on it).
- Anchor every optimization to the **memory-bandwidth roofline**, not FLOPS. Success = approaching HBM bandwidth on the histogram pass, not lighting up Tensor Cores.

**Warning signs:**
- A plan, issue, or PR mentions Tensor Cores, WMMA, cuBLAS, or "histogram as matmul."
- Optimization targets FLOPS utilization instead of achieved memory bandwidth.

**Phase to address:** **Architecture phase** (lock the roofline framing) and **every plan review** (scope fence).

---

### Pitfall 13: Benchmark dishonesty — comparing non-equivalent algorithms or weak baselines

**What goes wrong:**
You report a big speedup that doesn't survive scrutiny because you compared (a) your GPU forest against a *single-threaded* CPU baseline, (b) different hyperparameters (your shallower trees vs sklearn's deeper ones), (c) GPU "data already resident" vs CPU "from cold," or (d) Extra Trees (cheap random splits) against RF (expensive exact splits) as if equivalent. The credibility of the whole project — which lives or dies on benchmark honesty — collapses on first external review.

**Why it happens:**
Confirmation bias under solo-dev pressure to show a win; subtle non-equivalence (same-looking params, different effective work). The feasibility studies stress "benchmark-dependent advantage," meaning the benchmark methodology *is* the product's credibility.

**How to avoid:**
- **Pre-register the benchmark protocol** (PROJECT.md requirement): fixed datasets, fixed hyperparameters held identical across implementations, end-to-end timing (Pitfall 1), strongest available CPU baseline (sklearn `n_jobs=-1` **and** oneDAL/sklearnex), and cuML RF as the GPU baseline. Write it down *before* you have results so you can't move the goalposts.
- **Compare like-for-like algorithms:** your Extra Trees vs sklearn Extra Trees; your RF vs sklearn RF / cuML RF. Never your Extra Trees vs sklearn RF.
- Report **accuracy parity alongside speed** — a fast model that's less accurate isn't a win.
- Make benchmarks reproducible (scripted, seeded, versioned) so anyone can rerun them.

**Warning signs:**
- The CPU baseline isn't using all cores.
- Hyperparameters differ between implementations "for fairness."
- No accuracy numbers next to the speed numbers.
- The benchmark was tuned after seeing results.

**Phase to address:** **Benchmark/validation phase** — and the protocol is pre-registered *before* the crossover gate is evaluated.

---

### Pitfall 14: The kill-risk materializes — cuML ships first-class Extra Trees / sparse RF and erases the wedge

**What goes wrong:**
PROJECT.md names this the **top kill-risk**. Verified (June 2026): cuML's stable API currently lists **only** RandomForestClassifier/Regressor — **no Extra Trees** — so the wedge is real *today*. But cuML ships ~every 2 months; first-class GPU Extra Trees could land at any release and instantly erase the primary differentiator, leaving this project as a slower, less-maintained duplicate.

**Why it happens:**
The wedge exists *because* a well-resourced incumbent hasn't filled a small gap yet. Small gaps in front of fast-moving incumbents close. NVIDIA has every capability to add Extra Trees to cuML.

**How to avoid:**
- **Diversify the moat beyond "Extra Trees exists on GPU."** PROJECT.md already identifies the durable differentiators: **(1) the determinism contract, (2) exact tree SHAP, (3) strict sklearn parity, (4) no-silent-fallback honest dispatch.** Even if cuML adds Extra Trees, it is unlikely to ship bit-deterministic training + exact high-depth SHAP + audit-grade dispatch. *Lead with those*, not with "we have Extra Trees."
- **Design kernels for upstreamability** (Apache-2.0, clean separation) so that if cuML moves, contributing/merging is a *path*, not a dead end — turning the kill-risk into an exit.
- **Pre-register kill criteria and monitor cuML releases.** If cuML ships Extra Trees *and* it's deterministic *and* it has exact SHAP, the wedge is gone — pivot to the SHAP/determinism layer (which can sit *on top of* cuML) or upstream and stop.
- **Ship fast.** The wedge is a race. A perfect library in two years loses to a good-enough one that proves the crossover this year.

**Warning signs:**
- A cuML changelog mentions Extra Trees, randomized splits, or deterministic training.
- Your only stated differentiator in docs/marketing is "GPU Extra Trees."

**Phase to address:** **Continuous (monitor every cuML release).** Differentiator architecture decided in **spec/architecture phase**; kill criteria pre-registered alongside benchmarks.

---

### Pitfall 15: Scope creep back toward the rejected "7 fixes"

**What goes wrong:**
The project deliberately narrowed from an incoherent 7-fix, 4-subfield blueprint to "wedge + interpretability." Under solo-dev enthusiasm, the rejected items (SpMM Tensor-Core kernels, CPU BLAS autotuning/ADSALA, INT8/INT4 quantized inference, sTiles matrix inversion) creep back in as "while I'm here" additions, fragmenting effort across unrelated subfields and guaranteeing nothing ships.

**Why it happens:**
Each rejected fix sounds individually interesting; the discipline to stay narrow erodes one "small addition" at a time. The feasibility studies' core finding was **narrow-or-don't**.

**How to avoid:**
- **Treat the PROJECT.md Out-of-Scope list as a contract.** Items #1, #3, #4, #6, sparse/CSR, categorical, multi-GPU, multi-output are *fenced*. Adding any one requires re-opening the milestone scope explicitly, not a casual PR.
- Every new work item must trace to one of: the crossover proof, the determinism contract, exact SHAP, or sklearn parity. If it doesn't, it's out.
- Keep the MVP definition brutally small: prove the crossover on dense float32 Extra Trees first. Everything else is post-validation.

**Warning signs:**
- A plan references INT8, quantization, sparse/CSR, matrix inversion, or CPU BLAS tuning.
- "While I'm in here, I'll also..." appears in commit messages or plans.
- The roadmap grows new subfields before the crossover gate is passed.

**Phase to address:** **Every phase planning + plan review.** The scope fence is enforced continuously.

---

### Pitfall 16: Solo-maintainer CI/packaging burden eats the kernel budget (testing GPU code with no CI GPU)

**What goes wrong:**
Standard CI runners (GitHub-hosted) have **no GPU**, so your kernel tests, determinism tests, and benchmarks can't run in cloud CI. You either (a) sink weeks into a self-hosted GPU runner (security, maintenance, cost — for a solo dev this can exceed the kernel work itself), or (b) ship with GPU code that's never tested in CI and regresses silently.

**Why it happens:**
GPU CI is a known, real burden that ML-infra teams have whole engineers for. A solo dev is one person doing kernel R&D *and* DevOps. PROJECT.md's solo-Windows context makes this acute.

**How to avoid:**
- **Split the test pyramid by hardware need:**
  - **CPU-only CI (cloud, free):** Rust unit tests, the **CPU reference backend** tests, Python API/dispatch tests, packaging/import smoke tests, *property-based invariants that run on the CPU path*. This is the bulk of correctness coverage and needs no GPU.
  - **GPU tests (local Windows box, scripted, run before release/merge):** kernel correctness vs CPU oracle, bit-exact determinism, compute-sanitizer race/mem checks, benchmarks. Automate as a single local script (`make gpu-test`), not a cloud runner.
- **Defer self-hosted GPU CI** until there's a contributor or a release cadence that justifies it. For MVP, a documented "run this locally before tagging a release" gate is sufficient and far cheaper.
- Make the **CPU reference backend carry most correctness weight** (it's a stated requirement and doubles as the oracle) — this is what makes GPU-less CI viable.
- Keep packaging simple early (Pitfall 10) so release isn't a separate heroic effort.

**Warning signs:**
- You're configuring a self-hosted GPU runner before the kernel even works.
- GPU tests exist but only ever run "when you remember."
- CI is green but has never executed a single kernel.

**Phase to address:** **CI/test-infrastructure phase** (design the split early) + leverage the **CPU reference phase**.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Depth-first per-node kernel launches | Easiest first kernel to write | Near-rewrite to breadth-first; deep-tree underutilization (Pitfall 3) | Never — architect breadth-first from start |
| Float `atomicAdd` histograms | Simplest histogram; fast to demo | Breaks determinism contract (Pitfall 5); contention (Pitfall 4) | Only in a throwaway spike, never on the deterministic path |
| Benchmark "data already on GPU" | Shows a big number | Hides the real end-to-end loss (Pitfall 1); benchmark dishonesty | Never in reported numbers; OK as an internal kernel-only diagnostic clearly labeled |
| rust-cuda Rust→PTX for kernels | "All Rust" purity | Nightly churn; weak sanitizer/debug support (Pitfalls 9, 11) | Only if the spike proves tooling works; otherwise use cudarc+CUDA-C |
| "WOODELF-HD" novel exact SHAP without a reference | Sounds like a differentiator | Possibly re-solving a solved (polynomial) problem; unverifiable (Pitfall 8) | Only as a time-boxed research spike with a kill date |
| Self-hosted GPU CI runner early | "Real" CI | DevOps burden > kernel work for a solo dev (Pitfall 16) | After a contributor/release cadence exists |
| Skip NaN fixtures in tests | Faster green tests | Silent CPU/GPU divergence on real data (Pitfall 7) | Never — add NaN fixtures from the first differential test |
| One-wheel-fits-all CUDA versions | Simpler packaging | Breaks on user's driver/toolkit (Pitfall 10) | Never — link driver API, load PTX at runtime |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| scikit-learn (parity oracle) | Treating sklearn as a *bit*-level oracle; trying to replay its serial PRNG on parallel GPU | Use sklearn as a *distributional* oracle (accuracy/split-distribution equivalence); use the CPU reference backend as the bit oracle for your own RNG (Pitfall 6) |
| cuML (baseline + kill-risk) | Ignoring cuML release cadence; benchmarking only against CPU | Benchmark cuML RF as the GPU baseline; monitor every cuML release for Extra Trees (Pitfall 14) |
| `shap.TreeExplainer` (SHAP oracle) | Building a novel exact-SHAP method without validating against the known polynomial algorithm | Validate all SHAP outputs against `shap.TreeExplainer` within tolerance; it implements Lundberg's polynomial TreeSHAP (Pitfall 8) |
| Treelite / FIL (export target) | Assuming any tree representation exports cleanly | Match Treelite's exact model schema; test round-trip export→FIL-inference parity early |
| CUDA driver vs runtime API | Static-linking a toolkit version into the wheel | Link the forward-compatible driver API (cudarc/cust), load PTX/cubin at runtime (Pitfall 10) |
| pandas/numpy input | Assuming contiguous float32 input; hidden float64→32 copies | Detect dtype/contiguity, do one explicit conversion, report it in `execution_report_` (Pitfall 1) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| H2D transfer swamps kernel | `fit_ms` >> `kernel_ms`; win vanishes from cold start | Transfer once, keep resident; time the public API; pinned memory | Always present; dominates at small/medium n×d |
| Small-data crossover | GPU slower than sklearn `n_jobs=-1` below a size | Publish crossover surface; route `auto` to CPU below threshold | Below the measured (n×d) crossover boundary |
| Shrinking-node underutilization | `fit` super-linear in `max_depth`; many tiny launches | Breadth-first level-at-a-time build; CPU leaf-finishing cutover | Deep trees (depth >~10), many small nodes |
| Histogram atomic contention | Throughput far below bandwidth roofline; data-dependent slowdown | Shared-memory privatized histograms; per-warp privatization | Skewed feature distributions; many threads, few hot bins |
| Tensor-Core chasing | Effort on WMMA/matmul with zero speedup | Roofline framing (bandwidth, not FLOPS); Out-of-Scope fence | Never pays off — no GEMM in hot path (Pitfall 12) |
| O(3^D) SHAP interaction blowup | SHAP time explodes with depth | Use polynomial standard TreeSHAP for attributions; gate interactions as research | High `max_depth` *if* computing interactions, not attributions |

## Security Mistakes

Low surface area (a numerical library, not a network service), but domain-relevant:

| Mistake | Risk | Prevention |
|---------|------|------------|
| Copying GPL / Snap ML source for algorithms | License contamination; project must be Apache-2.0 (PROJECT.md) | Reuse *algorithms from papers*, never copy restrictively-licensed source; document provenance |
| Self-hosted CI runner exposed to public PRs | Untrusted code executes on your GPU/dev machine | Don't run untrusted-PR CI on a self-hosted runner; require manual approval (relates to Pitfall 16) |
| Unvalidated array shapes/dtypes into kernels | Out-of-bounds GPU reads → silent corruption or crash | Validate shape/dtype/contiguity at the Python↔Rust boundary before any device launch |
| Unsafe Rust at the CUDA FFI boundary | Memory unsafety crossing Rust↔CUDA↔Python | Minimize `unsafe`, wrap FFI in checked abstractions, run compute-sanitizer |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Silent CPU fallback | User thinks they're on GPU; gets CPU perf with no signal | `fallback="error"` default + `execution_report_` (already a requirement — the core differentiator) |
| `auto` always picks GPU | Small-data users get *slower* results than sklearn | `auto` routes below crossover to CPU and explains why (Pitfall 2) |
| Opaque conversions | Hidden float64→32 / copy costs surprise the user | Report every conversion + bytes transferred in `execution_report_` |
| "sklearn-compatible" overclaim | User expects bit-identical trees, gets different ones | Document the parity contract precisely (distributional, not bit-identical to sklearn) (Pitfall 6) |
| SHAP that's "exact" but isn't validated | User trusts wrong explanations | Validate against `shap.TreeExplainer`; state exactness guarantees and limits |

## "Looks Done But Isn't" Checklist

- [ ] **Fast kernel:** Often missing end-to-end timing — verify `model.fit(numpy_array)` cold-start beats CPU, not just `kernel_ms`.
- [ ] **Deterministic mode:** Often missing *bit*-exact verification — verify two same-seed runs produce byte-identical serialized models, not just `allclose`.
- [ ] **sklearn parity:** Often missing tie-break / RNG / NaN equivalence — verify split-structure and NaN-routing on differential tests, not just accuracy.
- [ ] **Exact SHAP:** Often missing validation against a reference — verify outputs match `shap.TreeExplainer` within tolerance; confirm you're solving attributions (polynomial, solved) not unwittingly chasing interactions (exponential).
- [ ] **GPU correctness:** Often missing sanitizer coverage — verify `compute-sanitizer racecheck/memcheck` is clean on every kernel.
- [ ] **The wheel:** Often missing clean-env validation — verify `pip install` of the built wheel imports and runs a kernel on a *fresh* machine/env.
- [ ] **Crossover claim:** Often missing the surface — verify a published (n×d) map exists, with the strongest CPU baseline and cuML RF.
- [ ] **No silent fallback:** Often missing the error path — verify `device="cuda"` with an unmet requirement *raises*, and `execution_report_` records every decision.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Depth-first kernel architecture (Pitfall 3) | HIGH | Re-architect to breadth-first level-at-a-time; treat as a rewrite — why it must be designed correctly first |
| Float-atomic nondeterminism (Pitfall 5) | MEDIUM | Swap histogram accumulation to integer/fixed-point or fixed-order scan; re-validate bit-determinism |
| H2D swamping (Pitfall 1) | LOW-MEDIUM | Make data resident across trees; pinned memory; usually a host-orchestration fix, not a kernel rewrite |
| Chosen toolchain (rust-cuda) too fragile (Pitfall 9) | MEDIUM | Pivot to cudarc + CUDA-C; host orchestration in Rust is reusable, only kernel-authoring layer changes |
| WOODELF-HD unverifiable (Pitfall 8) | LOW (if gated) | Fall back to known polynomial TreeSHAP for attributions; descope interactions — *cheap only if spiked before committing* |
| cuML ships Extra Trees (Pitfall 14) | HIGH (strategic) | Pivot to determinism+SHAP layer atop cuML, or upstream and stop; pre-registered kill criteria make this a decision, not a crisis |
| GPU CI burden (Pitfall 16) | MEDIUM | Move correctness to CPU-reference CI; demote GPU tests to a local pre-release script |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 9 Rust↔CUDA toolchain | Toolchain spike (FIRST, gating) | Working "hello-kernel" on Windows-native; sanitizer functions; build reproducible from clean checkout |
| 8 Exact-SHAP / WOODELF-HD | SHAP feasibility spike (early, gated) | Decision recorded: attributions (use polynomial TreeSHAP) vs interactions (research, kill-date); outputs match `shap.TreeExplainer` |
| 1 H2D swamping | Benchmark/validation phase (crossover gate) | Reported numbers are end-to-end `fit()` from numpy, cold + warm |
| 2 Small-data crossover | Benchmark/validation phase | Published (n×d) crossover surface vs sklearn `n_jobs=-1` + oneDAL + cuML RF |
| 13 Benchmark dishonesty | Benchmark/validation phase (pre-registered protocol) | Protocol written before results; like-for-like algos; accuracy reported alongside speed |
| 3 Shrinking-node | Core kernel architecture phase | `fit` time roughly linear in depth; no per-node launch storm in `nsys` |
| 4 Atomic contention | Histogram kernel architecture phase | Histogram throughput near bandwidth roofline; Nsight shows low atomic stalls |
| 5 Float nondeterminism | Histogram architecture + differential-test phase | Two same-seed runs byte-identical |
| 6 sklearn exact parity | Spec + CPU reference + differential-test phase | Parity contract documented; CPU=GPU bit-exact per seed; sklearn distributional equivalence |
| 7 NaN routing | Spec + CPU reference + differential-test phase | NaN fixtures pass; CPU and GPU route NaN identically |
| 10 maturin/abi3 packaging | Early packaging spike + release-prep | Built wheel installs and runs a kernel in a clean env |
| 11 GPU debugging | Toolchain spike + CPU reference + CI | compute-sanitizer clean; CPU oracle catches GPU divergence |
| 12 Matmul/Tensor-Core trap | Architecture phase + every plan review | No WMMA/cuBLAS in plans; optimization targets bandwidth, not FLOPS |
| 14 cuML kill-risk | Spec/architecture (differentiators) + continuous monitoring | Moat ≠ "Extra Trees exists"; kill criteria pre-registered; cuML releases watched |
| 15 Scope creep (7 fixes) | Every phase planning + plan review | No Out-of-Scope item enters a plan without explicit milestone re-scope |
| 16 Solo GPU CI burden | CI/test-infra phase + CPU reference | CPU-only CI carries correctness; GPU tests in a local pre-release script |

## Kill / Pivot Criteria (from the feasibility studies — pre-register these)

These are the conditions under which the project should **stop or pivot**, not push harder. PROJECT.md frames success as a validated crossover with pre-registered kill criteria; make these explicit before building:

1. **Crossover never materializes.** If, after the toolchain + kernel + benchmark phases, there is **no region of the (n×d) surface where end-to-end GPU Extra Trees beats sklearn `n_jobs=-1` / oneDAL** — the core premise is false. **Kill or pivot to the SHAP/determinism layer** (which can ride on cuML).
2. **Toolchain proves intractable on Windows.** If neither rust-cuda nor cudarc+CUDA-C yields a reproducible, debuggable, packageable kernel path within the toolchain-spike timebox (even with the WSL fallback) — **stop kernel work; reconsider stack.**
3. **cuML ships first-class deterministic Extra Trees + exact SHAP** before this proves its wedge — the differentiator is erased. **Pivot to upstreaming or to the interpretability layer; do not build a slower duplicate.**
4. **Exact-SHAP turns out to be either trivial (already-solved polynomial) or infeasible (unverifiable research).** If trivial → it's not a differentiator, lean on determinism instead. If infeasible → descope it; do **not** let it block MVP.
5. **Determinism contract can't be met at acceptable cost.** If bit-exact deterministic training requires a slowdown that destroys the crossover, the determinism *differentiator* and the speed *premise* are in conflict — **re-scope which one is the product.**

> **Meta-discipline:** the failure mode that sinks this kind of project is **sunk cost in a broad build before the crossover is proven.** The feasibility studies said *narrow-or-don't* and *validate-first*. Order the roadmap so the gating spikes (toolchain, crossover benchmark, SHAP feasibility) come **before** broad kernel build-out, and let their pre-registered criteria decide whether to proceed.

## Sources

- scikit-learn / cuML feasibility studies and "7 fixes" blueprint as summarized in `.planning/PROJECT.md` (project's own debunked-premise and kill-risk analysis) — HIGH confidence for strategic framing.
- cuML stable API reference (docs.rapids.ai) — confirmed only RandomForest{Classifier,Regressor} in tree ensembles, **no Extra Trees**, June 2026 — MEDIUM-HIGH.
- The Rust CUDA Guide (rust-gpu.github.io/rust-cuda) — Windows-native support, nightly-pinning, CUDA 12.x/13.x, MSVC + `nvvm\bin\x64` PATH requirement, "couldn't load codegen backend" failure mode — MEDIUM-HIGH.
- Rust GPU project blog (2025 rust-cuda reboot) — toolchain maturity / nightly-2025 pinning — MEDIUM.
- TreeSHAP literature (Lundberg et al. "Explainable AI for Trees" arXiv:1905.04610; "Understanding Interventional TreeSHAP" arXiv:2209.15123; Fast TreeSHAP arXiv:2109.09847) — standard/interventional TreeSHAP is **polynomial O(TLD²)**; exponential cost is specific to interaction indices — MEDIUM-HIGH; **this corrects the project's apparent premise that exact high-depth SHAP needs a novel method.**
- cuML breadth-first / level-at-a-time tree build and shared-memory privatized histograms — standard GPU tree/histogram patterns (NVIDIA RAPIDS blogs) — MEDIUM-HIGH.
- General GPU performance knowledge (PCIe vs HBM bandwidth, atomic contention, float non-associativity, kernel-launch latency) — HIGH.

---
*Pitfalls research for: GPU-native tree-ensemble library (Rust/CUDA/Python, Windows, solo dev)*
*Researched: 2026-06-19*
