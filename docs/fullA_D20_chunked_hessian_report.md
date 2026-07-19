# Full-A D=20 chunked-BLAS exact Hessian (Continuation 10, Part 1)

Branch `c10-chunked-hessian`, worktree
`/bbkinghome/edav/gravity_robustness/trade_robustness_modular/.claude/worktrees/agent-a60f519ea80642e1b`
(a git worktree of the same shared repo as `diag/fullA-d4-exact`, branched from
its HEAD `690b8f5`). Measured on `demand.mit.edu`, `JULIA_NUM_THREADS=20`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, KNITRO 14.2.0. Real-data
context via `context_real_d20.jl::d20_real_setup` (France focal, calibration
point, same point used throughout this investigation's D=20 work).

Read first (per this task's brief): `docs/fullA_fully_compressed_inner_report.md`
— a prior session (Continuation 9, Phase 3C) already tried several
dense-Hessian-avoidance strategies for this exact inner solve and found the
existing dense-materialize-then-one-big-gemm approach beats every
"fully compressed" alternative tried (2.7x-4.5x slower), because many small
BLAS calls lose to one big BLAS gemm. This task's chunking is a DIFFERENT
axis — not fewer FLOPs via compression, but the SAME FLOPs reorganized into
row-chunks for memory locality/peak-footprint reasons — but the same
underlying "many medium BLAS calls vs one big BLAS call" tension applies, so
the headline finding here (chunking is a wash at W=80,000, not a win) is
consistent with, not contradicted by, that prior finding.

## 1. What was built

`full_aod_diag/d4_exact/chunked_hessian.jl`:

- `hessian_chunked!(h, obj, chunk_size; zbuf=nothing)` — a drop-in alternative
  to `cc_algo/PsiObjectiveBundle.jl`'s `hessian!(h, obj::Union{PsiObjectiveBundleImplicit,
  PsiObjectiveBundleDelta})` (the method actually used by `d20_real_setup`'s
  `PsiObjectiveBundleImplicit` object). The baseline copies the FULL
  `W x outer_constr_index` scaled moment slice into `obj.H_copy` and forms the
  Hessian via ONE `BLAS.gemm!('T','N', H_copy, H_copy)` call. `hessian_chunked!`
  processes draws in row-chunks of a configurable `chunk_size`: only a
  `chunk_size x outer_constr_index` buffer is ever materialized, and the
  Hessian is accumulated across chunks via repeated `BLAS.gemm!` calls with
  `beta=1` (additive accumulation of partial Gram matrices). Mathematically
  EXACT — the packed-triangular output uses the identical formula/packing
  loop as the baseline, verbatim.
- `_prep_for_hessian!(obj, x)` — replicates the `arg0`/`arg1` refresh the
  production callable method runs unconditionally before dispatching to
  `hessian!`, needed because the chunked callback calls `hessian_chunked!`
  directly rather than going through the full callable.
- `inner_loop_KNITRO_chunked`/`inner_loop_internal_chunked` — faithful mirrors
  of `oracle_fast.jl`'s `inner_loop_KNITRO_profiled`/`inner_loop_internal_profiled`,
  swapping ONLY the Hessian callback registration (a closure capturing
  `chunk_size`/a reusable `zbuf`, since KNITRO ties one `userParams` value per
  callback handle, already used to pass `obj` to the FG callback). FG
  callback, variable/bound setup, complementarity wiring, solve-status
  handling are all reused UNCHANGED from `oracle_fast.jl`.

**Nothing in `cc_algo/PsiObjectiveBundle.jl` or `oracle_fast.jl` was
modified.** The dense baseline remains exactly as before; this file is purely
additive, exactly as this task's brief requires (`evaluate_fullA_fast` and the
main inner solver's default behavior are unchanged).

## 2. Correctness

### 2.1 D=4 (synthetic, `c10_chunked_hessian_correctness.jl`)

Hessian formula (direct call, same `x`, isolating the Hessian-forming step
from KNITRO's own iteration):

| chunk_size | bit_identical | max\|diff\| |
|---|---|---|
| 1 | false | 4.44e-15 |
| 2 | false | 2.78e-15 |
| 5 | false | 1.55e-15 |
| 8000 (=W, single chunk) | **true** | 0.0 |

Full inner solve (`inner_loop_internal_chunked` vs `_profiled`): identical
`status`/`n_fg_calls`/`n_hess_calls` at every chunk size; `K` bit-identical;
dual solution `max|dx|` at machine noise (0 to 7.8e-15).

### 2.2 D=20/W=80,000 (`c10_chunked_hessian_bench_d20.jl`)

Same pattern, at the real calibration point (`gamma'=0.987762`):

| chunk_size | bit_identical | max\|diff\| | max_rel_diff |
|---|---|---|---|
| 5000 | false | 2.73e-12 | 2.37e-15 |
| 10000 | false | 9.09e-13 | 2.18e-15 |
| 20000 | false | 5.46e-12 | 1.99e-15 |
| 40000 | false | 1.36e-12 | 2.55e-15 |
| 80000 (=W) | **true** | 0.0 | 0.0 |

**Every chunk size < W is within ~2-3x machine epsilon of the baseline, not
bit-identical** — this is expected floating-point summation-order noise
(splitting one big Gram-matrix accumulation into K sequential partial-Gram
accumulations changes the order of a large sum), explicitly anticipated in
this task's own framing ("mathematically EXACT... up to floating-point
summation-order noise, which a single large BLAS gemm's own internal
cache-blocking already reorders anyway"). `chunk_size=80000` (a single chunk
covering all draws) reduces to the mathematically identical computation and
is bit-identical, confirming the reduction is sound. At every chunk size, the
full inner solve's `nStatus`, `n_fg_calls`, and `n_hess_calls` matched the
baseline exactly (5 Hessian calls, 6 FG calls, `status=0` throughout) — no
iteration-count divergence, i.e. no sign the tiny numerical noise perturbs
KNITRO's own convergence path. Dual solution `max|dx|` stayed at 0 to 4.2e-17
(machine noise) across all chunk sizes.

**Verdict: exact to floating-point noise, not merely "close."**

## 3. W=80,000 benchmark

### 3.1 Isolated Hessian-callback-only timing (the clean, JIT-order-independent number)

Both the baseline `hessian!` and each `hessian_chunked!` variant were warmed
up (JIT-compiled) before their own 10-repetition median-timing loop, so this
comparison is NOT confounded by first-call compilation cost (see §3.2 for why
that confound matters):

| chunk_size | median (s) | speedup vs dense | alloc/call |
|---|---|---|---|
| REFERENCE (dense `hessian!`) | 0.5037 | 1.00x | 0 B (steady-state) |
| 5000 | 0.4899 | 1.028x | 32 B |
| 10000 | 0.4757 | 1.059x | 32 B |
| 20000 | 0.4806 | 1.048x | 32 B |
| 40000 | 0.4827 | 1.043x | 32 B |
| 80000 | 0.5324 | 0.946x | 32 B |

**Chunking is a wash on the Hessian-callback-only metric** — every chunk size
lands within ±6% of the dense baseline (0.946x-1.059x), no chunk size is a
clear win, none is a clear loss. Allocations are essentially zero either way
(32 bytes/call, likely a small bookkeeping allocation, negligible) since the
`zbuf`/`H_copy` buffers are reused across calls in both paths.

### 3.2 Full cold-solve timing — a JIT-ordering artifact caught and corrected

The main sweep script (`c10_chunked_hessian_bench_d20.jl`) always ran the
dense reference's cold solve strictly FIRST in the process, then each chunk
size's cold solve afterward. Naively read, this showed a suspiciously large
and uniform **~1.7-1.8x cold-solve speedup for every chunk size, including
chunk_size=80000** (mathematically identical to the baseline, bit-identical
Hessian) — a strong tell that the number was measuring first-call JIT
compilation cost, not a genuine algorithmic difference, per this
investigation's own standing "verify before causal claims" discipline.

**Confirmed directly** (`c10_chunked_hessian_jitcheck.jl`): after a warm-up
pass that exercises every variant once (chunks first this time, reference
LAST — the reverse order from the main sweep) in the SAME process, a second,
fully-JIT-warm timed pass gives:

| variant | median cold-solve (s), JIT-warm | speedup vs dense |
|---|---|---|
| reference (dense) | 4.128 | 1.00x |
| chunk_size=5000 | 3.954 | 1.044x |
| chunk_size=10000 | 3.968 | 1.041x |
| chunk_size=20000 | 4.129 | 1.000x |
| chunk_size=40000 | 4.278 | 0.965x |
| chunk_size=80000 | 4.231 | 0.976x |

The first-in-process solve, REGARDLESS of which variant it was, cost
~7.3-7.7s; every subsequent solve (any variant) cost ~4.1-4.3s. This
confirms: **the ~1.7x "cold speedup" in the naive sweep was ~100% a
JIT-compilation-order artifact** (mostly shared KNITRO.jl wrapper
compilation, paid once by whichever variant runs first), **not a real
algorithmic difference**. The TRUE, JIT-controlled cold-solve comparison
(§3.2's table) agrees with §3.1's isolated measurement: chunking is a wash,
not a win, at W=80,000 (0.965x-1.044x across all five chunk sizes).

### 3.3 Warm-solve timing

Warm solves need only 1 FG call and 0 Hessian calls (the KKT is already
satisfied from the prior solve's converged point), so the Hessian callback is
never invoked — warm timing is ~1.0x across all chunk sizes by construction
(1.341s baseline vs 1.296-1.402s chunked), consistent with this.

## 4. W=800,000 microbenchmark

Per the standing memory-safety discipline (a real server-wide memory
incident hit this exact W=800,000/D=20 combination earlier in this
investigation — a mis-defaulted flag scaled a diagnostic tensor to
780GB+ before being killed, `docs/fullA_continuation9_handoff.md`), a
memory probe was run BEFORE touching W=800,000:

- Analytic chunk-buffer footprint at `chunk_size=40000`: `40000 x 402 x 8
  bytes = 128.64 MB`.
- `Base.summarysize` of a REAL allocated chunk buffer: `128.64 MB` (matches
  the analytic estimate to within the ~40-byte Matrix object header
  `Base.summarysize` includes — not bit-for-bit equal to the raw analytic
  formula for that reason, but the same to 2 decimal places).
- For comparison, `obj.H` itself at `W=800,000`: `2.59 GB` — UNCHANGED by
  this task's chunking (Part 2 addresses building this matrix faster, not
  smaller; chunking targets only the Hessian callback's own second buffer).
  The baseline's `H_copy` (the second full-size buffer chunking eliminates):
  another `2.59 GB` avoided per Hessian call.

Probe passed (128.64 MB is nowhere near any plausible safety ceiling); one
run (chunk_size=40000, not a full sweep, per this task's instruction)
proceeded. **No safety-ceiling abort was triggered at any step; VmHWM stayed
flat at 16.97 GB from setup through every solve** (matches
`docs/fullA_D20_W800k_microbenchmark.md`'s previously-published 17.0-17.03GB
for this exact W/D):

| step | wall | VmHWM |
|---|---|---|
| `d20_real_setup(W=800000)` | 187.00s (matches that doc's published 172-190s range) | 16.97 GB |
| baseline cold solve | 42.491s (status=0, n_fg=5, n_hess=4) | 16.97 GB |
| baseline warm solve | 13.796s (n_fg=1, n_hess=0) | — |
| chunked (chunk_size=40000) cold solve | 37.099s (status=0, n_fg=5, n_hess=4, matches baseline exactly) | 16.97 GB |
| chunked warm solve | 13.206s (n_fg=1, n_hess=0) | — |
| dual solution vs baseline | `\|dK\|=0`, `max\|dx\|=1.1e-17` (machine noise) | — |

**Full cold/warm solve numbers are NOT a trustworthy signal here** — this
single-run W=800,000 microbenchmark ran baseline strictly first (paying
first-call JIT for both paths, since much of the KNITRO.jl wrapper machinery
is shared) then chunked second, the EXACT same ordering that produced a
spurious ~1.7-1.8x "cold speedup" at W=80,000 (§3.2) before the JIT-order
control corrected it to ~1.0x. The naive numbers above (`1.145x` cold,
`1.045x` warm) are presented for completeness but should NOT be read as real
— a proper JIT-order-controlled re-run (as done at W=80,000) was not
performed at W=800,000 given this task's explicit "ONE run, not a sweep"
instruction at this scale (a second controlled run would double this already
expensive step's cost).

**The one CLEAN, JIT-order-independent number from this run** is the isolated
Hessian-callback-only timing, taken AFTER both paths were already exercised
(and thus JIT-warm) by the cold/warm solves above:

| | wall | speedup |
|---|---|---|
| `hessian!` (baseline) | 5.014s | 1.00x |
| `hessian_chunked!` (chunk_size=40000) | 5.719s | **0.877x** |

**This is a genuine, real result, not a wash like W=80,000**: at
W=800,000, `chunk_size=40000` (20 separate chunks, 20 separate `BLAS.gemm!`
calls instead of 1) costs ~14% MORE wall-clock than the single-big-gemm
baseline for the Hessian-forming step alone. `max_rel_diff=6.3e-15` confirms
this is still the same floating-point-noise-level-exact computation as at
W=80,000 — the overhead is real but purely a time cost, not a correctness
concern. This is consistent with a per-chunk fixed overhead (buffer-fill pass
+ separate BLAS call dispatch, ×20 chunks) that becomes more visible relative
to the ever-larger single-gemm baseline as W grows — the opposite direction
from what would be needed to call chunking a win at this scale.

## 5. Verdict / recommendation

- **Chunked-BLAS Hessian is mathematically exact** (bit-identical at
  `chunk_size=W`, floating-point-noise-level at smaller chunk sizes, zero
  divergence in `n_fg_calls`/`n_hess_calls`/`nStatus` at any chunk size
  tested, D=4 and D=20).
- **At W=80,000, no chunk size is a clear win** over the existing dense
  `H_copy`-materialize-then-one-big-`gemm!` baseline — every chunk size
  landed within ±6% (0.946x-1.06x) once JIT-compilation-order noise is
  properly controlled for. This is consistent with, not contradicted by,
  Continuation 9 Phase 3C's finding that many medium/small BLAS calls lose to
  one big BLAS call for this same inner solve.
- **At W=800,000, chunking (chunk_size=40000) is a genuine, modest LOSS on
  the clean isolated-Hessian metric** (~14% slower, 0.877x) — the per-chunk
  overhead (buffer refill + separate `BLAS.gemm!` dispatch, ×20 chunks at
  this chunk size) becomes visible relative to an ever-larger single-gemm
  baseline as W grows, the opposite of what "chunking helps more at larger W"
  would need to show.
- **Peak-memory story (the genuine, verified benefit)**: chunking eliminates
  the SECOND full-size `W x outer_constr_index` buffer (`H_copy`) the
  baseline allocates every Hessian call, replacing it with a single reusable
  `chunk_size x outer_constr_index` buffer (128.64 MB at `chunk_size=40000`,
  W=800,000 — vs the ~2.59 GB `H_copy` the baseline would otherwise carry).
  `obj.H` itself (the raw moment matrix) is unchanged either way, and overall
  process VmHWM at W=800,000 was identical (16.97 GB) whether or not chunking
  was used in this run — because `obj.H`/`jac_h`-class arrays already
  dominate memory at this scale, not the Hessian callback's own working
  buffer, so the `H_copy` saving does not (yet) show up in whole-process peak
  RSS at W=800,000 either.
- **RECOMMENDATION (not applied — production default untouched)**: chunking
  is perf-neutral at W=80,000 and a modest (~14%) TIME cost at W=800,000 on
  the one clean metric measured, while its memory benefit (avoiding a second
  ~2.59GB `H_copy`-class buffer at W=800,000) did not translate into a
  measurable whole-process VmHWM reduction in this run. **This is a weaker
  case for chunking as a "safety valve" than originally anticipated** — it
  is architecturally available (correct, tested, additive) for a future scale
  where `H_copy`'s own footprint specifically becomes the binding constraint
  (e.g. if `outer_constr_index` grows much larger than D=20's 402, since
  `H_copy` scales with `W x outer_constr_index` while the chunk buffer scales
  with `chunk_size x outer_constr_index`), but there is no evidence here to
  adopt it as either the W=80,000 or the W=800,000 production default.

## 6. Files

New, all under `full_aod_diag/d4_exact/`: `chunked_hessian.jl` (the
implementation), `c10_chunked_hessian_correctness.jl` (D=4 correctness),
`c10_chunked_hessian_bench_d20.jl` (W=80,000 chunk-size sweep + correctness),
`c10_chunked_hessian_jitcheck.jl` (JIT-order control re-measurement),
`c10_chunked_hessian_w800k_probe.jl` (memory-safety probe + single W=800,000
run). No existing file modified.

Raw logs/CSVs: `results/fullA_d4/690b8f5/c10_chunked_hessian_bench_d20/`
(`harness_log.txt`, `chunked_hessian_w80k_summary.csv`, `jitcheck_log.txt`),
`results/fullA_d4/690b8f5/c10_chunked_hessian_w800k/harness_log.txt`.
