# Continuation 10, Section 4: BLAS/batching audit beyond the Hessian

Branch `c10-prod-wiring` (forked from `diag/fullA-d4-exact` @ `690b8f5`), worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c10-prod-wiring`. Measured on
`demand.mit.edu`, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
Julia 1.12.6, KNITRO (linked) 13.0.1. Scope: the remaining pipeline OUTSIDE the Hessian
itself (owned by the parallel `c10-chunked-hessian` workstream, not duplicated here) and
outside destination-batching/kernel-v2 (already covered, both nulls or mixed-not-wired,
by Continuation 9 Phase 4 — `docs/fullA_D20_winner_kernel_optimization_report.md`).

**Headline**: two real, measured wins found; the rest of the brief's candidate list is an
honest null for the reasons Continuation 9 already established (scattered/indirect
winner-index gathers do not benefit from BLAS at this codebase's access pattern), reported
here rather than re-run to avoid duplicating that work.

---

## 1. Real win #1: allocation-reuse in the L_fix central-FD gradient loop (~1.06x wall, ~1.23x fewer allocations)

**Where**: `composite_gradient_at_fast`'s per-coordinate loop
(`composite_gradient_fast.jl`) calls `a_block_fd_component` → `lfix_incremental_at`
(`lfix_incremental.jl`) twice per coordinate (plus/minus FD probe). Each call allocates a
FRESH `q = copy(cache.q0)` (W-length) and `lfix_from_q` allocates a FRESH
`Psi_q = similar(q)` (W-length) — every single probe, never reused. At D=20/W=80,000 a
full D²-coordinate gradient performs 2×399×2 = 1,596 such W-length (625KB) allocations,
~1.0GB of pure allocation churn per gradient call, for buffers used once and immediately
discarded.

**Fix** (new file `full_aod_diag/d4_exact/lfix_buffer_reuse.jl`, additive only — does not
modify `lfix_incremental.jl` or `composite_gradient_fast.jl`): `!`-suffixed variants
(`lfix_incremental_at!`, `a_block_fd_component!`, `lfix_from_q!`) that accept a
caller-owned `(q_buf, psi_buf)` pair and reuse it via `copyto!`/in-place `Psi!` instead of
`copy`/`similar`. A new `composite_gradient_at_fast_buffered` wrapper gives each THREAD one
persistent buffer pair (allocated once per gradient call, not per coordinate/probe).

**A real correctness hazard found and fixed**: the first implementation sized the
per-thread buffer pool as `Threads.nthreads()` and indexed it by `Threads.threadid()`.
This crashed immediately (`BoundsError` at index 21 of a 20-element vector) — under
`JULIA_NUM_THREADS=20`, `Threads.nthreads()` only counts the `:default` thread pool, but a
`Threads.@threads`-scheduled task can also land on the `:interactive` pool's thread(s),
whose `threadid()` is NOT bounded by `Threads.nthreads()`. Fixed by sizing the buffer pool
with `Threads.maxthreadid()` (confirmed empirically: `Threads.nthreads()==20` but
`Threads.maxthreadid()==40` on this machine/config) — the documented upper bound on
`threadid()` across ALL pools. Also switched `Threads.@threads` to explicit `:static`
scheduling (not the Julia 1.8+ default `:dynamic`): a per-thread buffer keyed by
`threadid()` captured once at the top of `do_coord!` is only safe if the task cannot
migrate to a different OS thread mid-coordinate (e.g. across the `lock()` call the
existing bandwidth-cache policy uses) — `:dynamic` scheduling permits exactly that
migration; `:static` pins each loop chunk to one thread for its whole execution, ruling
this out by construction. Neither hazard is hypothetical — both were caught live by
actually running the benchmark, not by inspection.

**Correctness** (`c10_buffer_reuse_d20_bench.jl`, D=20/W=80,000 real data): buffered vs.
original gradient, `h_mode=:cached`, threaded (cold-cache and warm-cache) and serial —
**max|diff| = 0.0 in all three checks** (bit-identical, not merely close).

**Performance** (warm-cache steady-state gradient call, the realistic in-KNITRO-loop case,
N=12 reps):

| | median wall | median allocation |
|---|---|---|
| original (`composite_gradient_at_fast`) | 3.0735s | 5224 MB |
| buffered (`composite_gradient_at_fast_buffered`) | 2.9046s | 4241 MB |
| **speedup / reduction** | **1.058x** | **1.232x** |

Reported plainly: this is a real but modest win (~6% wall-clock), not a blockbuster — the
allocation reduction (23%) is larger than the wall-clock gain, consistent with this being
mostly a GC/allocator-pressure fix rather than a FLOP-count fix; most of the ~2.9s per
call is still genuine compute (399 coordinates × up to 4 FD-adjacent O(W) probes each),
which this change does not touch. Not yet wired into `c10_d20_production_driver.jl`'s
default path (which still uses the original `composite_gradient_at_fast`) — offered as a
validated, available alternative; adopting it is a one-line swap (`composite_gradient_at_fast`
→ `composite_gradient_at_fast_buffered` in `run_profile_checkpointed`/`run_polish_checkpointed`'s
`cb_G!`) once the coordinating session wants the extra ~6%.

---

## 2. Real win #2: BLAS gemv/reduction for the KKT-residual and moment-residual post-solve tail (~2.1-2.2x, isolated)

**Where**: `oracle_fast.jl::evaluate_fullA_fast`'s post-solve bookkeeping AND
`infeasibility_screen.jl::evaluate_fullA_screened_compressed`'s tail (the SAME pattern,
independently present in both files) compute, every single evaluation:

```julia
# max_abs_moment_kkt_resid: nested Julia loop, nkkt≈400 outer x W=80000 inner
acc = 0.0
for j in 1:nkkt
    s = 0.0
    for ω in 1:W
        s += m_weights[ω] * G[ω, j]
    end
    acc = max(acc, abs(s / W))
end

# moment_resid: a manual column-sum, d≈402 x W=80000
mr = zeros(d)
for j in 1:d, ω in 1:W
    mr[j] += G[ω, j]
end
mr ./= W
```

Both are literally a vector-matrix product / column-sum over a DENSE `W x d` matrix —
textbook `gemv`/reduction territory, currently hand-rolled as nested Julia loops instead of
dispatching to BLAS.

**Synthetic microbenchmark** (no context needed — pure numerical operation on a random
`W=80,000 x d=402` matrix, `BLAS.set_num_threads(1)` to match this repo's
`OPENBLAS_NUM_THREADS=1` discipline, N=20 reps):

| computation | current (loop) | BLAS candidate | speedup |
|---|---|---|---|
| KKT residual (`nkkt=400`) | 0.0539s | `maximum(abs.(G[:,1:nkkt]'*m_weights))/W` — 0.0240s | **2.247x** |
| moment residual (`d=402`), `sum(dims=1)` | 0.0503s | `vec(sum(G,dims=1))./W` — 0.0302s | 1.666x |
| moment residual (`d=402`), `gemv` | 0.0503s | `(G'*ones(W))./W` — 0.0240s | **2.095x** |

Correctness: differences at 1e-16/1e-17 (floating-point summation-order noise from BLAS's
own reduction tree vs. the loop's strict left-to-right order — expected and immaterial,
not a bug).

**Not wired into production in this pass**: both call sites (`oracle_fast.jl`,
`infeasibility_screen.jl`) are shared, load-bearing production code touched by every other
active workstream this continuation (`c10-chunked-hessian` in particular touches
`oracle_fast.jl`-adjacent Hessian code); per this repo's established discipline of small,
independently-reviewable changes, this audit reports the validated substitution rather
than landing it mid-session in files another workstream may be concurrently editing. The
one-line swap for each site is given above verbatim and is ready to adopt.

---

## 3. Audited and confirmed NULL (not re-run — already covered)

Per the task's own explicit instruction not to duplicate Continuation 9 Phase 4's work
(`docs/fullA_D20_winner_kernel_optimization_report.md`):

- **Destination-block/coordinate batching of `L_fix`** (the outer-loop `lfix_composite`
  object): Phase 4 §3 found loop-REORDERING gives no measurable effect (0.993-1.008x,
  noise) because the pivot-reduced z-space's natural column-major linear index is ALREADY
  destination-major — there is no reordering left to find this way. A genuine
  restructured kernel sharing base-score computation across coordinates was flagged there
  as the only remaining lever and was out of scope for that task too; still out of scope
  here given the additive-only, non-duplicating brief for this session.
- **Batching several Hessian-vector-product right-hand sides into one matrix call**: owned
  by the parallel `c10-chunked-hessian` workstream this session; not investigated here to
  avoid duplicate work.
- **Rank-one/low-rank centering (`e_w - lambda_hat_d`) operations OUTSIDE the Hessian**:
  the closest analogue is `compressed_moments.jl::compressed_dual_contraction`'s per-draw
  inner loop (`acc += κ[cf.winner[s,d],d] * cf.wval[s,d]`) — a SCATTERED gather indexed by
  the (data-dependent, per-draw) winner identity, not a regular dense operation. This is
  exactly the access pattern Continuation 9 Phase 3C/4 already diagnosed as NOT benefiting
  from BLAS/vectorization at this codebase's scale (`docs/fullA_fully_compressed_inner_report.md`
  §3 finding 2, `docs/fullA_D20_winner_kernel_optimization_report.md` §4.3's
  `compressed_transpose_contraction_v2` regression) — not re-tested, would reproduce the
  same negative result.
- **Batched fixed-dual scalar index lookups** (`dest_contrib_incremental_o1`/
  `dest_contrib_incremental_top3`, `lfix_incremental.jl`): same class as above — the
  winner/runner-up/third-place identity varies per draw, so any "batch" would still be a
  per-draw indirect gather, not a regular matrix op. No new candidate found here beyond
  what Phase 4 already tested and rejected.
- **Chunked primal-weight calculations**: `m_weights`/`p_weights` (normalized dual weights)
  are already O(W) elementwise vector ops (`m_weights ./ sum(m_weights)`), already
  vectorized by Julia's broadcast machinery — no loop to batch.
- **Destination-pair Hessian block accumulation**: Hessian-proper, owned by
  `c10-chunked-hessian`; not investigated here.

---

## 4. Files

New (all under `full_aod_diag/d4_exact/`, additive only — no existing file modified):
- `lfix_buffer_reuse.jl` — buffer-reuse gradient variant (win #1, wired-available not
  wired-default).
- `c10_buffer_reuse_d20_bench.jl` — correctness + D=20/W=80,000 benchmark for win #1.
- KKT/moment-residual BLAS candidate (win #2) benchmarked in a scratch script
  (`/tmp/c10_kkt_blas_bench.jl`, synthetic — not committed, reproduced verbatim in §2 above
  since it needs no repo-specific context to rerun).

Raw logs: `/tmp/c10_buffer_bench2.log` (win #1), timing numbers for win #2 reproduced
inline above.
