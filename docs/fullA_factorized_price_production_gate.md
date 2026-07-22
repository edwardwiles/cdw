# Factorized price-cache backend: production gate

Written 2026-07-22, `perf/fullA-factorized-price-production` branch. Addendum to the price-tensor
elimination report (`docs/fullA_price_tensor_elimination_report.md`), which found the
factorized "Backend C" representation 6.75x faster / 6x less memory than the Reference —
**measured against the obsolete unbuffered `composite_gradient_at`, not the actual production
path**. This document redoes that comparison fairly, against the real production gradient,
combines the factorized representation with a persistent workspace (not previously built) and
the existing `GradWorkspacePool`, and gates production adoption on a much more exhaustive
correctness suite than the original report ran.

## 1. Production head

- **Branch**: `perf/fullA-factorized-price-production`, forked from `audit/fullA-postmerge-correctness`
  at commit `dc3196c`.
- **`dc3196c`** is itself `docs/fullA_nested_knitro_solve_hang_rootcause.md`'s fix commit
  (reverted a real regression — `par_concurrent_evals` deadlocking the outer/inner nested
  `KN_solve` pattern — found and fixed earlier this session). Contains: the CM/parallel +
  driver/δ5 production merge (`3855430`), all 14 independent-audit-remediation fixes
  (`docs/fullA_independent_audit_remediation.md`), the allocation/cache-cleanup workspace code
  (`GradWorkspacePool`, `CrossDeltaExactCache`, both opt-in), and the price-tensor-elimination
  experiment (Backends A/B/C, `docs/fullA_price_tensor_elimination_report.md`).
- **`git status`**: clean except untracked diagnostic artifacts from the hang investigation
  (`full_aod_diag/d4_exact/test_isolate_hang_*.jl`, two throwaway single-threaded `.opt` variants,
  raw log files) — confirmed dead ends, kept for the record until end-of-session cleanup, not
  part of this branch's own work.
- **Julia**: 1.12.6 (juliaup-managed, `~/.juliaup/bin`). **KNITRO**: 13.0.1 (`KNITRODIR=/opt/shared_sw/knitro/13.0.1`,
  confirmed via `KN_get_release` at runtime), **KNITRO.jl** package version 1.2.1.
- **Thread env**: no `JULIA_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` set by default
  in this shell (`Threads.nthreads()==1` unless explicitly requested); production driver runs
  typically set `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
  (`docs/fullA_D20_checkpoint_resume_report.md`).

## 2. Fair production baseline — traced from the real callback, not assumed

`c10_d20_production_driver.jl`'s `cb_G!` (both `run_profile_checkpointed` and
`run_polish_checkpointed`) dispatches:

```julia
if use_pooled_gradient
    gfull, meta = composite_gradient_at_fast_pooled(xf, ctx, pe, grad_pool; base = base, threaded = true, ...)
else
    gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true, ...)
end
```

`use_pooled_gradient` **defaults to `false`** everywhere it's declared
(`c10_d20_production_driver.jl:387,742`) — **`composite_gradient_at_fast_buffered` is the actual
current production gradient function**, not `composite_gradient_at` (the addendum's own
correction of my prior report's denominator) and not `composite_gradient_at_fast_pooled` either
(that path exists, is validated, but is opt-in and not the default).

Traced directly, not assumed:
- **Is `GradWorkspacePool` persistent?** Yes, when `use_pooled_gradient=true`: `grad_pool =
  build_grad_workspace_pool(W)` is built ONCE per `ctx` (outside the callback, at driver-function
  scope), not per gradient call. But this is the **opt-in** path, not the default.
- **Is `LFixBaseWorkspace` (this session's Backend A) persistent / wired in anywhere?** **No.**
  Neither `composite_gradient_at_fast_buffered` nor `composite_gradient_at_fast_pooled` uses it —
  both call the allocating `build_lfix_base_cache(x_free0, ctx, base; validate_dense=false)`
  directly (confirmed: `lfix_buffer_reuse.jl:145`, `gradient_workspace.jl:203`). `GradWorkspacePool`
  only covers the SIX small per-coordinate scratch buffers (`q`/`psi`/`price`/`pTσ`/`contrib`/`cf`,
  each length `W`) used inside the coordinate-probe loop — **not** the two big `W×D×D`
  `price0`/`pTσ0` tensors, which are still allocated fresh (~589.6MB) on every single gradient
  call, in BOTH the buffered and pooled paths.
- **Does the reference still store both full price tensors?** Yes, confirmed, in both existing
  production paths, always (this is the addendum's own §2 question — answered: yes, unconditionally,
  regardless of `use_pooled_gradient`).

**Correct denominator for every speedup number in this document**: `composite_gradient_at_fast_buffered`
(threaded=true, the actual default), NOT `composite_gradient_at` (the prior report's denominator,
which additionally lacks the `lfix_buffer_reuse.jl`-level `q_bufs`/`psi_bufs` per-coordinate reuse
that `_fast_buffered` already has) and NOT `composite_gradient_at_fast_pooled` alone (a real but
non-default alternative, itself lacking `LFixBaseWorkspace`).

## 3. The two production candidates, both built

**Backend A+** (`lfix_base_workspace_pooled.jl::composite_gradient_at_Aplus`): identical to
`composite_gradient_at_fast_pooled` except sourcing the base cache from `lfix_base_workspace.jl`'s
persistent `build_lfix_base_cache!` instead of the allocating builder. Validated bit-identical
to `composite_gradient_at_fast_pooled` at D=4 (`test_lfix_base_workspace_pooled.jl`, 4/4, both
serial and threaded).

**Backend C+** (`lfix_factorized_workspace.jl::composite_gradient_at_Cplus`): the factorized
(`logCC`+`mulU`, O(W·D)) representation from `lfix_factorized.jl`, now with (1) a persistent
`LFixFactorizedWorkspace` (mirrors Backend A's aliasing design: build once per `(D,W)`, refill in
place, `ws.valid` lifecycle, same alias-safety invariant) instead of allocating `WinnerRefCache`'s
arrays fresh every call, and (2) genuine use of the existing `GradWorkspacePool`'s per-thread
`contrib`/`cf`/`q`/`psi` buffers via new buffer-aware `dest_contrib_incremental_top3_C!`/
`cf_contrib_at_C!`/`lfix_incremental_at_Cplus!`/`a_block_fd_component_Cplus!` (mirroring
`gradient_workspace.jl`'s own in-place pattern for the two-tensor backend). Validated bit-identical
to the standalone allocating Backend C at D=4, and against the Reference to ~1e-14
(`test_lfix_factorized_workspace.jl`, 20/20; the ~1e-14 gap is Backend C's own already-documented
log-sum-exp-vs-direct-power floating-point difference, not new).

## 4. Exact factorization (unchanged from the original report, restated for this document)

```
log p_{sod} = mulU_{so} + logCC_{od}         (mulU = μ·log U, fixed given the draws + μ;
                                               logCC = log constCons, updated once per base point
                                               and again per coordinate probe's affected cells)
p_{sod}^{1-σ} = exp((1-σ)·[mulU_{so} + logCC_{od}])   (reconstructed on demand, O(1) per queried
                                                        cell -- no W×D×D tensor, ever)
```

- **Permanent across all outer evaluations**: nothing (μ can change if it's ever made free; in
  the current parameterization μ is fixed data, so `mulU` is effectively permanent per context).
- **Updated once per base point** (`build_winner_ref!`): `logCC0` (D×D), `mulU` (W×D, only if μ
  could change — currently invariant), full O(W·D²) top-3 ranking scan (`winner`/`runnerup`/
  `third` + their scores), `contrib0`/`q0`/cf pieces.
- **Updated for one coordinate probe**: `logCC′` at the 1-2 affected `(o,d)` cells only (O(D²)
  full recompute of `constCons_matrix`, cheap; the O(W) `dest_contrib_incremental_top3_C!` pass
  itself only evaluates the changed origins' new scores, reusing cached top-3 for everything
  else).
- **Dependent on the gravity pivot**: the second of the 1-2 affected `(o,d)` cells (pivot's own
  destination column) — handled identically to every other backend via `affected_cells(pe,
  coord_idx)`, unchanged.

No new assertions added for "a supposedly fixed component changed" — `μ`/`σ` are read fresh from
`base.θ_full0`/`ctx.σ` on every `build_winner_ref!`/`build_lfix_base_cache_C!` call, so a future
change to what's "fixed" would simply be picked up correctly rather than silently going stale
(there is no cached copy of μ/σ anywhere that could drift from the source of truth).

## 8-10. Fair D=20/W=80,000 benchmark — the headline result, against the REAL production gradient

`c17_production_gate_benchmark_d20.jl`, `JULIA_NUM_THREADS=20`/`OPENBLAS_NUM_THREADS=1`/
`MKL_NUM_THREADS=1` (matching documented production config), `threaded=true` for all three
(genuine parallelism, not simulated), median of 3 repeats, at δ=1 (typical) and δ=5 (difficult):

| | Reference (`composite_gradient_at_fast_buffered`, actual production default) | A+ | C+ |
|---|---:|---:|---:|
| Wall time, δ=1 | 3.117s | 2.801s (**1.11x**) | 0.774s (**4.03x**) |
| Wall time, δ=5 | 3.140s | 2.613s (**1.20x**) | 0.748s (**4.20x**) |
| Allocated bytes | 3556.3MB | 131.9MB (**27.0x less**) | 53.3MB (**66.8x less**) |
| Gradient correctness | — | **0.000e+00** (bit-identical) | 4.337e-17 (machine precision) |

**This supersedes the original price-tensor-elimination report's 6.75x/6x claim**, which
compared against the obsolete, unbuffered `composite_gradient_at` — not a fair baseline. Against
the REAL production gradient (already buffer-reusing and genuinely thread-parallel), the honest
numbers are more modest: **A+ is a small, essentially risk-free win** (bit-identical, ~10-20%
faster, 27x less allocation — mostly reduces GC pressure over a sustained run rather than raw
wall-clock); **C+ is the substantial win** (a real 4x wall-clock speedup, 67x less memory,
correct to machine precision at both a typical and a difficult point).

## 5-7. Exhaustive D=4 correctness

`test_production_gate_exhaustive_d4.jl`, 28/28 passed:
- **Every origin pair, every destination, 4 step sizes, single-changed-origin** (64 cases at 2
  fixed points): A+ exactly 0 (bit-identical, as expected — it's the same computation, just
  persistent-buffer-backed); C+ within 5.7e-14 to 2.3e-13 (still machine precision, slightly
  larger than the earlier ~1e-14/1e-17 numbers at smaller perturbations, since larger h probes
  accumulate marginally more floating-point drift along the log-sum-exp path — not a correctness
  concern).
- **Every origin pair as a genuine simultaneous 2-changed-origin case, every destination, 3 step
  sizes** (72 cases): A+ exactly 0; C+ within 1.1e-13 to 2.8e-13.
- **Winner-transition sweep**: an honest gap, not overclaimed — the h-sweep on the specific draw
  probed only exercised "winner stays" within the tested range for that draw/destination, not an
  actual winner→runner-up switch. Values matched to <1e-8 at every h regardless, but this does
  not itself prove all four listed transition scenarios (incumbent stays / challenger wins /
  runner-up takes over / third-place relevant) — the earlier exhaustive-pair sweep (72+64 cases)
  almost certainly DOES include real winner switches incidentally (large step sizes like h=1.0
  routinely flip winners at D=4), but this was not explicitly isolated/labeled per-transition-type.
- **Numerical range**: 12/12 extreme cases (`h` up to ±20, i.e. `Aod_theta` from `e^-20` to
  `e^20`) stayed finite and matched between Reference and C+ — no NaN/Inf triggered in the tested
  range, so the "both backends agree on which entries are non-finite" fallback path was not
  actually exercised (not a gap in the backends, just means the tested range wasn't extreme
  enough to reach an actual overflow/underflow boundary at D=4/this context's data).

## 11. Compressed-moment interaction

Not a new integration point — `lfix_factorized.jl`/`lfix_factorized_workspace.jl` never import
or call anything in `compressed_moments.jl`/`compressed_live.jl`. Backend C+ is scoped entirely
to the `L_fix` gradient's own base cache (the same scope `LFixBaseCache`/Backend A already have);
the compressed-moment machinery is a separate system (dense/compressed moment-MATRIX construction
for the Hessian callback), consuming `winner`/`wval`-shaped outputs from an entirely different
code path (`winners_from_certificate`/`build_compressed_factual`, not `LFixBaseCacheC`). C+
introduces no new dense `W×402` materialization and no duplicate moment matrix — confirmed by
inspection (no such array appears anywhere in `lfix_factorized_workspace.jl`), not by a new test.

## Next

Section 13 (wire behind a flag, short real outer trajectory — the KNITRO hang fix makes this
newly possible), 14 (cache/concurrency gate reruns with C+ active), 12 (optimized-value
directional checks), 15-17 (decision, fallback flag, final report) — tracked in the todo list,
executed incrementally.
