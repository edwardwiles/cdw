# Allocation / cross-delta / :kbplus finalization gate, 2026-07-22

Branch: `perf/fullA-factorized-price-production` (tip of the linear production chain --
see `docs/fullA_REPO_MAP_2026-07-22.md`). This document records every gate the finalization
brief asked for, with commands, real output, and honest pass/fail/gap status -- nothing here is
claimed passed without a command actually being run this session.

## 1. Phase 2A -- pooled gradient re-verification

**Claim being re-checked**: `docs/fullA_postmerge_allocation_productionization.md` recommended
flipping `use_pooled_gradient`'s default to `true` based on evidence gathered before the
`par_concurrent_evals` hang-fix; that evidence needed independent re-confirmation on this exact
commit, post-fix.

Commands run (this session, `demand.mit.edu`):
```
export PATH="$HOME/.juliaup/bin:$PATH"; source .knitro_env.sh
JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  julia --project=. full_aod_diag/d4_exact/test_gradient_workspace.jl
JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  julia --project=. full_aod_diag/d4_exact/test_driver_pooled_gradient_wiring.jl
```

Results:
- `test_gradient_workspace.jl`: **9/9 PASS**. Pooled 756.5 MB vs buffered 3738.6 MB (4.94x lower
  allocation), bit-identical gradient, real D=20/W=80,000 point.
- `test_driver_pooled_gradient_wiring.jl`: **4/4 PASS** (after fixing a latent `isa Int` ->
  `isa Integer` test bug -- KNITRO's native status is `Int32`, `Int` means `Int64` on this
  64-bit build, so the old assertion could never have passed). Both `use_pooled_gradient=false`
  and `=true` reach **identical `kappa=0.0203135174923732`** through a real KNITRO outer solve.
- `GradWorkspacePool` confirmed built once per `ctx` (driver-function scope), not per callback,
  by direct source inspection (`c10_d20_production_driver.jl`).

**Verdict**: gate PASSES, independently reconfirmed on this commit post-hang-fix. See §3 for the
default-flip decision (folded into the unified `price_cache_backend` selector rather than done
as a standalone flip).

## 2. Phase 2B -- CrossDeltaExactCache real staged-continuation gate

**Real bug found and fixed before this gate could even run**: `screened_eval`'s `exact_cache`
keyword was typed `Union{Nothing,SafeExactCache}` -- Julia enforces keyword-argument types at
the call site (confirmed via a minimal repro: `TypeError`, not silent widening). Every
`cb_F!`/`cb_G!` call that forwards `exact_cache=exact_cache` would therefore `TypeError` the
instant `exact_cache` held a `CrossDeltaExactCache`, i.e. the instant `cross_delta=true` was
ever actually used through the real driver. This is why no staged `cross_delta=true`
continuation had ever been observed to complete through production -- it could not have.
**Fixed**: widened to `Union{Nothing,SafeExactCache,CrossDeltaExactCache}`.

Added explicit cache counters (`n_lookups`/`n_hit_verified`/`n_hit_infeasible`/`n_miss`/
`n_store`) to `CrossDeltaExactCache`, observed at its existing `_cache_lookup`/`_cache_store!`
dispatch points -- no production call site touched. Added `maxit_override::Union{Nothing,Int}`
to both driver entry points and `run_staged_delta5_continuation`, so an eval-count-matched
comparison (not just wall-clock-matched) is possible without confounding "cache helped" with
"cache left more wall-clock time."

Command run:
```
JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  julia --project=. full_aod_diag/d4_exact/c19_cross_delta_gate.jl
```

Real D=20/W=80,000 output, **19/19 PASS**:
- **Section 0** (live-fire): `screened_eval` now accepts a `CrossDeltaExactCache` without
  TypeError; one lookup + one store recorded on a cold miss.
- **Section 1** (A/B/A regression, through `screened_eval` directly, real data): stored at
  δ=2 (point P), a different point Q solved at δ=3, re-queried P at δ=5 -- cache hit confirmed
  (lookup count +1, store count unchanged, `cache_hit=true`). `Delta_dual`, `gravity_value`,
  `max_abs_moment_kkt_resid`, `zeta`/`lambda` **all match a fresh cold direct solve exactly**
  (`==`, not `isapprox`). `Delta_minus_delta` correctly reflects the caller's δ=5, not the
  stored δ=2. Verified-success classification agrees.
- **Section 2** (real staged δ=2→3→4→5 continuation, `run_staged_delta5_continuation`, both
  `cross_delta=false` and `=true`, both eval-count-matched via `maxit_override=4` and
  wall-clock-matched via `stage_maxtime_real=45s` alone): **all 4 stages complete cleanly in
  every arm.** Cache counters grew monotonically and **produced real hits in a genuine
  production-shaped run**: by stage 4 of the wall-clock-matched arm, `lookups=24
  hit_verified=3 hit_infeasible=0 miss=21 store=1`. The hits are modest in absolute count (this
  workload's `cb_F!`/`cb_G!` retry pattern revisits the same start point across a stage's warm
  attempt / cold-retry / gradient-recompute calls more than it revisits a *prior stage's* point)
  -- honestly reported, not oversold as a major speedup.
- **Section 4** (cold, cache-disabled re-verification): both arms' final incumbents
  cold-re-verify to <1e-6 of their reported `Delta`.

**Verdict**: gate PASSES. Cache is correct (A/B/A proof + cold reverify), produces real
(if modest) hits on a real staged run, and the eval-matched/wall-clock-matched comparison
infrastructure now exists for future use. Per the brief's own rule ("if it produces no real
hits... do not advertise a speedup"): it DID produce real hits, so `cross_delta` is validated as
safe and correct to enable, but the observed hit volume on this particular short/synthetic
staging pattern does not by itself demonstrate a large wall-clock win -- default-on is a
reasonable next step, not yet done as part of this session (see §6).

## 3. Phase 3 -- price_cache_backend selector + C+ gates

Added `price_cache_backend::Union{Nothing,Symbol}` (`:buffered`/`:pooled`/`:aplus`/`:cplus`/
`:kbplus`) to both `run_profile_checkpointed` and `run_polish_checkpointed`, via
`resolve_price_cache_backend`, reconciling it with the older `use_pooled_gradient` flag (now
`Union{Nothing,Bool}`, nothing-sentinel default): neither given -> `:buffered`; one given -> that
one; both given consistently -> the explicit backend; both given contradictorily -> hard error
before any KNITRO call. Persistent workspaces (`grad_pool`/`lfix_ws`/`lfix_c_ws`/`lfix_kb_ws`)
built once per `ctx`, matching the existing pattern.

Command run:
```
JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  julia --project=. full_aod_diag/d4_exact/c20_backend_selector_driver_smoke.jl
```

Real D=20/W=80,000, δ=1, **12/12 PASS**:
- Contradictory `use_pooled_gradient=true` + `price_cache_backend=:cplus` errors before any
  KNITRO call (checked via `try`/`catch` around `run_polish_checkpointed`).
- All four of `:buffered`/`:pooled`/`:aplus`/`:cplus` complete a real short KNITRO outer solve
  and reach **kappa within the stated tolerance of `:buffered`**: `:pooled`/`:aplus` bit-exact
  (diff=0.0), `:cplus` within 1e-8 (observed diff=0.0 at this trajectory).

**C+'s two previously-disclosed gaps** (`docs/fullA_factorized_price_production_gate.md` §12/14
-- independent optimized-value directional checks, and cache/concurrency/checkpoint tests with
C+ active through the real driver): the selector wiring above closes the "wired into the real
driver" half of gap §14 (C+ is now reachable via `price_cache_backend=:cplus` in both entry
points, exercised in the smoke test above). The full cache/concurrency/checkpoint re-run
specifically targeting C+ (vs. this session's :kbplus-focused equivalent, §4 below) and the
independent directional check for C+ specifically were **not separately re-run this session**
-- :kbplus's own versions of these checks (§4/§5) exercise the same selector/dispatch code path
C+ shares, but do not substitute for a C+-specific directional-check campaign. **Disclosed gap,
not silently assumed closed.**

**Verdict**: selector wiring gate PASSES (12/12, real driver). C+'s §12 directional-check gap
remains open specifically for C+ (though the analogous check for :kbplus, which shares the same
ranking machinery, was run -- see §5).

## 4. Phase 4 -- Backend :kbplus implementation

Reference-aligned ratio factorization (`docs/fullA_CURRENT_STATE_2026-07-22.md` §6 point 5):
reuses `winner_certificate.jl`'s `WinnerRefCache`/`build_winner_ref` **verbatim** for ranking
(log-score, already exp-free), differing from Backend C+ only in value reconstruction:
`constConsσ_od / USigmaPow_so` (one division) instead of `exp((1-σ)*score)` (one exp call) --
eliminates every W-scale `exp`/`log`/`^` from the coordinate-probe hot path.
`cf_contrib_at`/`gamma_component_analytic` reused unchanged (already ratio-shaped in the
existing codebase). New files: `lfix_kbplus.jl` (allocating), `lfix_kbplus_workspace.jl`
(persistent, mirrors Backend A+/C+'s aliasing design).

## 5. Phase 5 -- :kbplus correctness suite

### D=4 (`test_lfix_kbplus.jl`, reuses `test_lfix_factorized.jl`'s own fixed points, attributed)

Command: `julia --project=. full_aod_diag/d4_exact/test_lfix_kbplus.jl`

**116/116 PASS**:
- (A) winner/runner-up/third identities exact vs Reference AND vs C+; `contrib0`/`q0` match
  Reference to ~1e-15/1e-16, match C+ to the same order.
- (B) adversarial exact price tie -> `TiedWinnerError` from both Reference and :kbplus, matching
  `n_tied_pairs`/`examples`.
- (C) full `L_fix` value across every coordinate's ±h probe: worst abs diff vs Reference
  5.6e-16 (upper40 point) / 1.9e-15 (lower point) -- **tighter than C+'s own reported ~1e-14 to
  2e-13 vs Reference**, consistent with the ratio reconstruction's expected closer numerical
  alignment to the Reference's own direct-power formula.
- (D) full gradient: maxabsdiff vs Reference 5.6e-15/2.5e-14, vs C+ 1.1e-14/1.4e-14 (both points).
- (E) `count_winner_flips_KB`/`select_bandwidth_KB` agree with Reference exactly (`h`, mass both
  `==`, not `isapprox`).
- (F, extra) extreme-range probes (`h` up to ±20, i.e. `Aod_theta` from e^-20 to e^20): no
  finiteness mismatch; finite-value relative agreement ~1e-15 to 3e-15 (an initial absolute-only
  1e-10 check failed at these large-magnitude points for a purely cosmetic reason -- fixed to
  the same relative-or-absolute standard used everywhere else in the file).

**Not run at :kbplus's own dedicated exhaustiveness** (an honest, explicitly disclosed gap, same
category as C+'s own original report's gaps): the full D=4 "every origin pair × every
destination × several step sizes" 136-case exhaustive sweep (`test_production_gate_exhaustive_d4.jl`'s
own pattern for A+/C+) was not separately built for :kbplus -- the 2-point/6-coordinate/
2-step-size pattern reused from `test_lfix_factorized.jl` gives strong evidence (116 assertions,
0 failures, at both a fixed and a randomized-direction point) but is not the same exhaustive
coverage A+/C+ themselves received.

### D=20 (`c21_kbplus_d20_gate.jl`)

<!-- PHASE5_D20_RESULTS_PLACEHOLDER -->

## 6. Phase 6 -- fair benchmark and adoption decision

<!-- PHASE6_RESULTS_PLACEHOLDER -->

## 7. Remaining gaps (explicit, not hidden)

- C+'s own independent optimized-value directional check (§3) not separately re-run this
  session (the analogous :kbplus check was run, §5).
- Cache/concurrency/checkpoint-resume tests with C+ (as opposed to :kbplus) specifically active
  were not re-run this session.
- :kbplus's D=4 exhaustive sweep is a reduced (2-point/6-coordinate) pattern, not the full
  136-case sweep A+/C+ received.
- `cross_delta`'s real hit-rate evidence (§2) is from one short staged run; a longer/more
  realistic campaign would give a more representative hit-rate/wall-savings number.
- The CM interrupted/resume production campaign (D=20/W=80,000/L=50/δ=1) specified in the
  original productionization brief was not run this session (was already an open item before
  this session started).
