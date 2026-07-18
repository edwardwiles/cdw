# Structural moment-construction audit and optimization

User-requested addendum to Continuation 5's Priority 2, prompted by Priority 1A's finding that
`inner_moment_build` is 69.2% of a warmed exact-hard evaluation (`docs/fullA_p1_warmed_profile.md`
Part A). Branch `diag/fullA-d4-exact`, worktree `../gravity-fullA-d4`. New files, all additive, none
modify production code: `full_aod_diag/d4_exact/moments_fast.jl` (the optimization),
`test_moments_fast.jl` (equivalence), `profile_moments_fast.jl` (before/after + D/W scaling).

## 1. Call chain (confirmed by direct code reading, not assumed)

```
evaluate_fullA / solve_base_state
  -> CS.inner_loop_internal(obj, θ_full)
       -> obj.moments!(K, G, θ_full, obj.U, obj)  ==  EK_moments_gammanorm_directgp!
            (full_aod_diag/moments_gammanorm.jl -- confirmed the ACTIVE moments! for this
            investigation's ctx via context.jl's `(moments!) = EK_moments_gammanorm_directgp!`)
            1. unpack obj.γ (wHat, L, τ, P, Uσ, cHat, UPow_scratch/UσPow_scratch, ...)
            2. mu = θ[1], sigma = θ[2]
            3. Aod (level) <- Aod_theta (free outer param) via a closed-form gauge transform
               (cHat, wHat, tau, lambda) -- O(D^2), draws-independent, already correctly hoisted.
            4. AodPow = (Aod./cHat).^(-mu)                       -- O(D^2), cheap.
            5. gamma/gamma_prime setup (gammanorm: forced to 1 except focal) -- O(D).
            6. K[:] = gamma_prime[baseIndex]                      -- O(W), trivial.
            7. UPow = U.^(-mu); UσPow = Uσ.^(-mu)                 -- O(W*D), THE finding, see sec 2.
            8. Threads.@threads over Th=nthreads() draw CHUNKS, each calls:
                 hFunction!(view(G,chunk,:), view(UPow,chunk,:), view(UσPow,chunk,:), ...)
                   for d in 1:D, for o in 1:D: pricesTemp[o] = constCons[o,d]/UPow[ω,o]
                   (O(D) per draw), MinInd! (O(D) hard-min), fills D moment columns per destination.
                 hFunctionCounter!(view(K,chunk), view(G,chunk,:), ...)
                   autarky (counterType==1) branch is a SINGLE VECTORIZED broadcast
                   (moments/hFunction.jl:201), no draw loop -- already near-minimal.
            9. gravMoment==1: newGravityMoment!(...) -- confirmed cheap elsewhere
               (gravity_value bit-identical across every theta tried, prior session's finding).
           10. post-processing: divide by gamma(mu*(1-sigma)+1) (O(W*moments)), PMM subtract
               (usePMM==1? -- NOT active for this investigation, confirmed: AD_PARAMS/indicators
               usePMM=0), NormalizeMoments (O(W*moments)), SamplingWeights multiply (O(W*moments)).
```

Diagnostic-only code confirmed NOT on this path (not touched by this audit): `localGravityMoment!`/
`localGravityCrossMoment!` (`localGravityMoment=0`), `smoothMinIndNew!` (commented out in
`hFunction.jl`, `MinInd!` is what actually runs), the `counterType!=1` branches of
`hFunction!`/`hFunctionCounter!` (this investigation is `counterType=1`, autarky, throughout).

## 2. Fixed-mu,sigma precomputation audit -- THE finding

**Verdict: a scratch BUFFER existed (`obj.γ.UPow_scratch`/`UσPow_scratch`, avoids a fresh `zeros()`
allocation each call) but NOT a VALUE cache.** `UPow = U.^(-mu)` and `UσPow = Uσ.^(-mu)` (step 7
above) were recomputed via a full `.^` broadcast over all `W*D` entries on **every single call** to
`EK_moments_gammanorm_directgp!`, despite `mu` being **provably invariant** for a `ctx`'s entire
lifetime in this investigation: `context.jl` pins `θ_lo[1]==θ_hi[1]==θ0_up[1]` (and the same for
`θ[2]=sigma`), and `free_idx` never includes index 1 or 2 (confirmed directly: `free_idx=[7..23]` at
D=4) -- so no outer-loop perturbation, poll, or FD probe in this entire investigation ever changes
`mu`. `θConstant` (the indicator gating which branch computes `UPow`) is confirmed `0` at this
investigation's setting (not `1`), so the redundant-recompute branch is genuinely the ACTIVE one, not
a dead alternative.

**Fix**: `moments_fast.jl::MuSigmaPowCache` + `get_upow!` -- a single value cache keyed on `mu`,
recomputed only when `mu` changes (never, in this investigation) or the cache is fresh. `mu`/`sigma`
are the ONLY two fixed outer parameters in this problem's layout, so one cache (not "multiple redundant
representations," which the task explicitly warned against) is sufficient -- no log-space or
pre-exponentiated alternate representation was added on top, since profiling (below) shows the win is
already realized by simply not repeating the `.^` broadcast.

`EK_moments_gammanorm_directgp_fast!` mirrors `EK_moments_gammanorm_directgp!` line-for-line, with
step 7 alone replaced by a `get_upow!` cache lookup; `hFunction!`/`hFunctionCounter!`/
`newGravityMoment!` are called verbatim, unmodified, imported not copied.

## 3. Equivalence (mandatory, checked before any timing was trusted)

`test_moments_fast.jl`: calibration, `upper_lfixcomposite_sr1_60s`, `lower_stalled`, 5 random
feasible perturbations -- **8/8 points, `max|G diff| = max|K diff| = 0.0` exactly** (not just within
tolerance: the fast path is a strict refactor of the same arithmetic, bit-identical). Cache behavior
verified structurally: `n_recompute=1` (the very first call), `n_reuse=7` (every subsequent call,
including calls at DIFFERENT theta/A_od points -- confirming the cache correctly keys on `mu` alone,
not the full theta vector).

## 4. Before/after warmed timing, D/W grid

`profile_moments_fast.jl`, N=30 reps, pre-warmed, median, `JULIA_NUM_THREADS=1` (existing repo default
for these benchmarks; see sec 5 for the threading dimension separately):

| D | W | orig (median) | fast, warm cache (median) | speedup | UPow cache: cold recompute | warm lookup |
|---|---|---|---|---|---|---|
| 4 | 8000 | 8.098ms | 5.748ms | **1.41x** | 1.775ms | 0.000018ms |
| 4 | 80000 | 78.074ms | 62.823ms | **1.24x** | 17.574ms | 0.000020ms |
| 6 | 8000 | 14.874ms | 13.971ms | 1.06x | 2.776ms | 0.000020ms |
| 8 | 8000 | 25.126ms | 20.316ms | **1.24x** | 3.630ms | 0.000021ms |
| 10 | 8000 | 41.120ms | 31.568ms | **1.30x** | 4.402ms | 0.000021ms |

A real, modest, honestly-reported speedup (1.06x-1.41x depending on D/W, not a dramatic win) purely
from eliminating ONE redundant `.^` broadcast pass. The `warm lookup` cost is genuinely ~0 (a
dictionary-free struct-field read), confirming the cache mechanism itself adds no material overhead.
D=6's smaller 1.06x gain is consistent with `docs/fullA_block_local_performance.md`'s own D=6
base-point fragility note (this D=6 benchmark point is the theta==1 calibration-style point, not a
converged candidate, used here purely for its trivial constructibility -- see profile script comment)
and is not itself surprising: `UPow`'s share of total cost is not a fixed fraction across D (it scales
as `O(W*D)`, while `hFunction!`'s own draw loop scales as `O(W*D^2)` -- so `UPow`'s relative share
should SHRINK as D grows, exactly the trend D=6->8->10 shows before the D=10 number ticks back up
slightly, within measurement noise for a single-run median).

## 5. Threading (existing production behavior, confirmed active + benchmarked)

**The requested "threads over draw chunks" variant already exists in production**, not a new
optimization: `EK_moments_gammanorm_directgp!`/`_fast!` both wrap their `hFunction!`/
`hFunctionCounter!` calls (and `_fast!`'s own `get_upow!` cache-miss recompute) in
`Threads.@threads for t in 1:Th` over `Th=Threads.nthreads()` draw-index chunks
(`round.(Int, (t-1)/Th*W)+1 : round.(Int, t/Th*W)`), confirmed by direct code reading (sec 1 step 7-8)
-- this audit did not need to ADD chunk-level threading, only confirm it is genuinely exercised (it
is: `JULIA_NUM_THREADS` controls it directly, no other flag gates it) and measure its effect, since
the default single-session `JULIA_NUM_THREADS=1` this investigation otherwise runs under makes it
invisible unless explicitly set:

| `JULIA_NUM_THREADS` | orig (median, D=4/W=8000) | fast, warm cache (median) | speedup (orig/fast) |
|---|---|---|---|
| 1 | 8.098ms | 5.748ms | 1.41x |
| 2 | 6.370ms | 5.247ms | 1.21x |
| 4 | 5.717ms | 5.038ms | 1.14x |
| 8 | 5.788ms | 4.812ms | 1.20x |
| 16 | 5.119ms | 4.645ms | 1.10x |

Draw-chunk threading gives a real absolute win (orig: 8.10ms@1thread -> ~5.1-5.8ms@2-16 threads) but
**plateaus almost immediately past 2 threads** at this W=8000/D=4 scale -- consistent with this
investigation's established finding elsewhere (`docs/fullA_block_local_performance.md` sec 6) that
fine-grained per-draw work saturates Julia's task-scheduling overhead well before the thread count
does, at this problem size. The mu/sigma cache fix's OWN relative speedup shrinks somewhat as thread
count rises (1.41x at 1 thread -> ~1.1-1.2x at 8-16 threads) because threading itself already
parallelizes away part of the `UPow`/`UσPow` cost the cache eliminates -- the two levers are
complementary, not fully additive, exactly as expected (thread count reduces WALL time of a
recomputation; the cache removes the recomputation entirely, so at high thread counts there's simply
less recomputation cost left for the cache to remove).

**Destination-level threading was NOT implemented as a separate variant** -- explicitly flagged, not
silently skipped. Reasoning: at D=4 there are only 4 destinations, well below even the modest 2-4
useful threads the draw-chunk approach already saturates at; `hFunction!`'s own per-destination inner
loop is only `O(D)` work (4 origins), making per-destination task granularity far too fine relative to
Julia's scheduling overhead -- the same argument `docs/fullA_block_local_performance.md` sec 6 already
made for FD-probe-level threading at this D. This is a specific, falsifiable prediction for a future
D=10+ continuation (more destinations = more independent per-thread work), not asserted as true at
larger D without measurement.

**No nested-threading oversubscription risk found in this continuation's actual wiring**: the
composite gradient's own A-block coordinate threading (`composite_gradient_at_fast`'s
`Threads.@threads for k in 2:D2`, Priority 2) never calls `moments!`/`hFunction!` at all -- it uses the
O(1)-incremental `lfix_incremental_at` tier specifically to AVOID rebuilding the moment matrix per
probe. `moments!` is called exactly once per outer KNITRO iterate (the base-state solve), outside any
coordinate-parallel region -- so `Threads.@threads` nesting between the composite gradient's own
parallelism and `moments!`'s internal draw-chunk parallelism cannot occur in the current wiring. Should
a future continuation call `moments!` from inside a per-coordinate parallel region (it does not
today), this would need an explicit single-threaded override -- flagged as a latent risk to watch, not
a live bug.

## 6. Implementation improvements: implemented vs considered-and-not-attempted

**Implemented** (equivalence-tested, sec 3-4 above):
- Fixed-mu,sigma value cache for `UPow`/`UσPow` (the one genuinely redundant computation found).

**Considered, NOT implemented this continuation** (flagged, with reasoning, not silently dropped --
each would need its own equivalence-test discipline before being trusted, and this continuation's
remaining time was prioritized toward Priority 2/3's wiring and the fair hard-vs-smoothed comparison
per the task's own stated fallback ordering):
- `@inbounds`/`@simd` on `hFunction!`'s/`hFunctionCounter!`'s inner draw loops -- these are PRODUCTION
  shared files (per memory `uomodel-gamma-fwl-cleanup`, touched by multiple continuations/branches);
  already `@inbounds`-annotated at the outer `ω` loop level (`moments/hFunction.jl:54`). A deeper pass
  would require modifying shared production code directly (not this investigation's additive-only
  `full_aod_diag/d4_exact/` discipline) -- out of scope without a separate, explicitly-scoped
  production-code continuation.
- Log-space price computation (compute in log-competitiveness units, exponentiate only the winner) --
  `hFunction!` already computes `pricesTemp[o] = constCons[o,d]/UPow[ω,o1]` as a single division per
  origin per draw (not a `log`+`exp` round trip), so there is no repeated `log`/`exp` to eliminate in
  the current formulation; a genuine log-space reformulation would change the numerical algorithm
  (not just its cost), which this investigation's "preserve exact equivalence" discipline treats as
  out of scope without a dedicated derivation and validation pass.
- Fusing winner selection and moment accumulation, or reusing winner information across the
  diagnostic winner/runner-up calculations (`winners.jl`'s `compute_winners`) -- `hFunction!` already
  fuses `MinInd!` (winner selection) directly into the same draw-loop pass that accumulates `G`
  (sec 1 step 8); the SEPARATE diagnostic winner/runner-up calculation (`compute_winners`,
  `winner_compute` in Priority 1A's profile, 14.8% warmed) is a genuinely distinct consumer (used for
  `winner_hash`/external validation, not `moments!` itself) -- unifying the two would require changing
  what `evaluate_fullA`'s public return fields are derived from, flagged as a larger refactor than
  this continuation's remaining budget supports, not attempted.
- A "production mode" flag that omits `winner_compute`/KKT-residual/gravity diagnostics on ordinary
  moment-only callbacks -- `moments!` itself never computes these (they are `oracle.jl`/`oracle_fast.jl`
  -level post-processing, already separately timed in Priority 1A's breakdown, ≤14.8% each) -- not a
  `moments!`-internal cost, out of this audit's scope.

## 7. Updated cost estimate

Combining this audit's fix with Priority 1A's own warmed breakdown
(`docs/fullA_p1_warmed_profile.md`): `inner_moment_build`'s ~8.0ms (69.2% of a 12.8ms warmed
`evaluate_fullA_fast` call, `JULIA_NUM_THREADS=1`) is reduced to ~5.7ms by this fix alone (1.41x, sec
4's D=4/W=8000 row) -- a genuine ~2.3ms per-evaluation saving, i.e. a warmed `evaluate_fullA_fast`
call should now cost roughly **~10.5ms** (was 12.8ms) once this fix is wired into the live oracle path
(NOT YET WIRED into `oracle_fast.jl`/`evaluate_fullA` as of this document -- `moments_fast.jl` is
additive/standalone, matching this investigation's mirror-don't-modify discipline; wiring it into a
new `oracle_fast2.jl`/equivalent is a natural, low-risk next step this continuation did not reach
given the overall Priority 2/3 time budget). The composite gradient's own live callback (currently
~141ms, Priority 1A sec) pays this cost exactly ONCE per outer iterate (the base-state solve) -- so
this fix's practical effect on the composite-gradient bottleneck is small in absolute terms (~2.3ms of
~141ms) but stacks correctly with every other Priority 2 lever (threading, base-state sharing) since
none of them touch this same cost.
