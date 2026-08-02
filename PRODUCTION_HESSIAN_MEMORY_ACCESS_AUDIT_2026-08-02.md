# Production Hessian Memory-Access / Loop-Order Audit (2026-08-02)

## Hardware counter availability (limitation, disclosed per task brief §11)

`perf`/hardware performance counters were checked and are **not available** on this host to this
user (no `perf` binary on PATH, no `/proc/sys/kernel/perf_event_paranoid` access confirmed). Per
the task brief's own fallback instruction ("If hardware counters are unavailable, use arithmetic
and memory-traffic models and document the limitation"), this audit uses the source-level
loop-order/access-pattern reading below plus the already-measured block-timing/allocation data as
indirect evidence, and does not claim direct cache-miss/bandwidth measurements.

## H_EC/H_EF full-width winner/bin tables (flexible_cm, common_frechet, cm_meanzc)

`build_bin_tables_threaded!`/`prefix_sum_tables_threaded!` (cm_hessian_threaded.jl) -- confirmed
the SINGLE largest block for both flexible_cm and common_frechet (~55% of callback wall time, per
the block-timing map). Source-level access pattern:
- `Ttab`/`Stab` per-thread scratch (`D x D x L1 x L1` / `D x NCORE x L1`) -- statically sized once
  at context construction (`build_thread_local_scratch`), each thread writes ONLY into its own
  `tls.Ttab[t]`/`tls.Stab[t]` slice during the parallel accumulation pass (confirmed: `for t in
  1:nt; fill!(tls.Ttab[t], 0.0); ...; end` then a `Threads.@threads` accumulation loop reading
  `Bidx`/writing `tls.Ttab[tid]` per draw `s`) -- this is a standard thread-local-accumulate-then-
  reduce pattern with NO shared-write contention and NO false sharing across threads for the
  accumulation itself (each thread's slice is a fully separate, large (`D*D*L1*L1*8` bytes)
  allocation, not adjacent cache lines of a shared array).
- The FINAL fixed-order reduction step (summing the `nt` per-thread `Ttab`/`Stab` slices into the
  shared `cctx.CScum`/`cctx.CT`) is inherently a full read of `nt` separate large arrays -- for
  `nt=10`, `D=20`, `L=50`, `Ttab` alone is `20*20*51*51*8 bytes ≈ 3.3MB` PER THREAD, `33MB` total
  read for the reduction. This is real, unavoidable O(nt*D^2*L^2) memory traffic inherent to the
  thread-local-accumulate design (the alternative -- a single shared array with atomics or locks --
  would trade this bandwidth cost for contention cost; this codebase's own header comments
  document the accumulate-then-reduce choice as deliberate, not accidental).
- The DRAW-LOOP itself (`for s in 1:Wraw`) reads `Bidx[s, x]` for each `x in 1:D` -- `Bidx` is a
  `W x D` matrix, so this is a ROW-MAJOR-style access into a COLUMN-MAJOR Julia array (each `x`
  iteration jumps `W` elements in memory) -- i.e. `Bidx[s,:]` for fixed `s` is STRIDED, not
  contiguous, in Julia's native column-major layout. This is a genuine, real memory-access-pattern
  finding: **the draw loop's inner dimension (D=20) strides through `Bidx` with stride W=20,000-
  100,000 elements (160KB-800KB) between consecutive reads**, likely causing a cache miss on every
  single element for W at this scale (far larger than any cache level). Confirmed by direct
  source read (`cm_hessian_threaded.jl`'s `build_bin_tables_threaded!` inner loop structure,
  `for s in 1:Wraw ... for x in 1:D ... Bidx[s,Bidx[s,x]] ...`-style indexing) -- NOT verified with
  a hardware counter (see limitation above), so this is a plausible, source-level-supported
  finding, not a directly measured one.

**This is a real, plausible optimization candidate NOT implemented in this pass**: transposing
`Bidx` to `D x W` (draws as the fast-varying/contiguous dimension) would make the inner-`x` loop's
memory access contiguous instead of strided. This is explicitly OUT OF SCOPE for this audit's
"surgical, one-change-at-a-time" policy as accepted work -- it would require re-deriving `Bidx`'s
construction and every OTHER consumer of `cctx.Bidx` (confirmed by grep: read in at least
`build_bin_tables_threaded!`, `build_bin_tables!`, and the origin-ZC `nu0` construction in this
audit's own harness) to use the transposed layout consistently, a change with a much larger blast
radius than this audit's two accepted allocation fixes. Flagged as a candidate for a SEPARATE,
dedicated task with its own D4/D20 correctness gates, not implemented or partially implemented
here.

## H_EZ drawmajor_v2 / H_CZ draw_chunk_reordered (cm_meanzc, origin_zc)

Both candidate names ("drawmajor", "draw_chunk_reordered") are self-documenting about their own
loop-order design intent -- confirmed by reading `hez_drawmajor_v2_candidate_2026-08-01.jl` and
`hcz_drawchunk_candidate_2026-07-29.jl`'s own header comments: both were explicitly built as
loop-order optimizations over earlier ("winner_bin"/"draw_chunk_thread_local") backends, already
validated and merged as the production defaults (per the 21fa6ec production integration commit).
This audit did not find a further loop-order improvement beyond what that prior work already
established -- re-deriving or second-guessing an already-validated, already-production loop-order
choice was not attempted given this audit's scope is allocation/efficiency AUDITING, not a second
independent loop-order optimization pass over already-optimized kernels.

## H_ZZ weighted workspace (`zc_gram_dispatch!`, `:blas_syrk`)

Delegates to `BLAS.syrk!` (a vendor-optimized OpenBLAS routine) once the raw weighted workspace
(`ensure_zc_raw_weighted_workspace!`) is built -- the memory-access pattern inside the actual GEMM/
SYRK call is OpenBLAS's own internal blocking/tiling, not something this audit's source-level
reading can meaningfully second-guess or improve; the BLAS-thread-policy audit (separate doc)
already established that this call benefits materially from BLAS=8 threading, which is the
practically-actionable lever for this block, not a hand-written loop-order change.

## Feature tiling / draw tiling

No W-scale kernel found in this audit's source reading uses explicit cache-blocking/tiling beyond
what BLAS itself provides internally for the SYRK/GEMM calls, and beyond the thread-chunking
(`cross_hessian_chunk_ranges`, static contiguous per-thread draw ranges) already used by the
threaded cross-Hessian fills. No untiled O(W) loop was found allocating or scanning in a way that
would obviously benefit from additional manual tiling beyond the existing thread-level chunking.

## Summary

One concrete, plausible (not hardware-counter-confirmed) finding: `Bidx`'s `W x D` column-major
layout makes `build_bin_tables_threaded!`'s per-draw inner loop over `D` origins strided rather
than contiguous. Flagged as a candidate for future work, not implemented here (out of this audit's
single-change-at-a-time scope, given its cross-cutting blast radius). No other material
memory-access-pattern defect was found in the W-scale kernels audited; the ZC-family drawmajor/
draw_chunk_reordered backends were confirmed to be already the product of a prior, validated
loop-order optimization pass, and the SYRK/GEMM-backed blocks' internal access pattern is
OpenBLAS's own concern, addressed at the thread-count level (see the BLAS/thread-policy doc) not
the source-loop level.
