# Continuation 6 handoff — read this FIRST if picking up this investigation

Written mid-session, stopped early due to context limits. Branch `diag/fullA-d4-exact`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`, **HEAD `943655e`** (clean tree). This
continuation was given a large task (`fullA_p2_p3...` era docs + a new "Continuation 6" prompt covering
9 sections: moment-cache wiring, live bandwidth caching, the exact gamma profile, stationarity
reassessment, nested-W, hard-vs-smoothed fair comparison, gated D-scaling, threading discipline,
documentation). **Only the highest-priority item (root-causing and fixing a real bug that blocked the
gamma profile) was completed.** Everything else in that prompt remains open — see "What's left" below.

## Canonical headline (still current, unchanged from Continuation 5)

- Best upper incumbent: `upper_lfixcomposite_sr1_60s`, κ=0.17245688540655113, γ'=0.8926359585,
  Δ=0.9999924058. `EXACT_FEASIBLE_CANDIDATE=true`, `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)=true`,
  `ROBUST_LOCAL_CANDIDATE=false`.
- Best lower incumbent: `lower_lfixcomposite_fast_sr1_300s`, κ=0.005428799948779983, Δ-δ=+8.5e-7.
  Same classification tier as upper.
- `lfix_composite_fast`+SR1 reaches the SAME point as the old method in ~27s not ~43s.
- Smoothed scalar-envelope gradient costs ~44ms warmed, not 3.1s (that was JIT compile).

## What THIS session did: found and fixed the `build_lfix_base_cache` bug

This was Continuation 5's #1 flagged open item. **Root cause found and FIXED, verified, committed**
(commits `4f696b7`, `943655e`):

- **Root cause**: exactly 1 out of 32,000 (draw, destination) pairs at the point that exposed the bug
  had a **bit-exact price tie** between two origins. `MinInd!` (`misc/smoothMinIndNew!.jl`, what
  `hFunction!` actually calls) sets `xInd[i]=1` for every `i` with `x[i] <= xMin` (not `>`) — on a tie,
  **both** origins get winner-share credit. `build_lfix_base_cache`'s winner-finding
  (`min_secondmin_with_idx`'s `isless` scan, and every O(1)/O(D) tier built on it) assumes a **unique**
  winner by design. This is why the self-validation failed with large (not tolerance-sized) errors —
  it was correctly catching a real discrepancy, just for a different reason than the old error message
  ("closed-form derivation has a bug") implied. Verified precisely: `price_and_pTsigma_cell`,
  `aod_pow_cell`, `CONST_d`, and the winner computation are all individually EXACT (0.0 diff against
  the true moment matrix) once the tie's multi-winner contribution is correctly accounted for.
- **Fix** (`full_aod_diag/d4_exact/lfix_incremental.jl`): `detect_price_ties()` scans for ties BEFORE
  the self-validation and throws a new, specific `TiedWinnerError` (not the old misleading message).
  `composite_gradient_at_fast` (`composite_gradient_fast.jl`) catches `TiedWinnerError` specifically
  and falls back to `full_rebuild_gradient_fallback` — a full-rebuild central-FD gradient via
  `fixed_dual_L` (always correct, rebuilds the complete moment matrix so it reflects `MinInd!`'s true
  tie-splitting) — for that ONE point. Slower (32 full moment rebuilds) but always correct, matching
  this file's own "correctness over speed on rare edge cases" precedent (the existing 2-changed-origins
  fallback).
- **Verified**: zero regression at every previously-validated point (`test_lfix_incremental.jl`,
  `test_composite_gradient_fast.jl` both still 100% PASS — the fix is a no-op there since none have
  ties). At the previously-failing point: now cleanly caught, fallback produces a finite gradient
  agreeing in sign with an independent `Delta_dual` FD check.
- **Re-ran `gamma_profile.jl`'s coarse grid** (`results/fullA_d4/4f696b7/gamma_profile_20260718_111444/`):
  dramatically improved — 7/11 points now genuinely KNITRO-optimize (hundreds of real evaluations each,
  vs. only 3/11 before the fix). Only the two extreme theoretical-bound corners (g≈0.8528, g=1.0) still
  trivially fail — that's the OLD, separately-known calibration-adjacent infeasibility corner, unrelated
  to the tie bug.
- **New, NOT-YET-ANALYZED finding**: the fixed profile_Delta(g) is **non-monotonic** beyond the
  incumbent — drops toward g≈0.957 (Δ≈0.002, deep in the feasible interior) then RISES again at
  g≈0.979 (Δ≈0.073). This was never visible before the fix (those points were unoptimized triviality).
  **This needs investigation** — it could mean there are multiple local basins in the A-block
  minimization at different g, which matters for whether `profile_Delta(g)=delta` has a unique crossing.
  Not analyzed this session due to context limits.

## What's left — in priority order

This session did NOT get to any of the Continuation 6 prompt's other 8 sections. Re-read that prompt
in full (it's in the conversation history the user gave; if not visible, ask the user to re-paste it —
it's long and detailed, covering: wiring `MuSigmaPowCache` into the live path, live bandwidth caching
with a real revalidation policy, the full upper/lower gamma profile with adaptive refinement and
multistart, deeper stationarity reassessment, nested-W verification, a fair hard-vs-smoothed
comparison, gated D-scaling, threading discipline, and a final canonical report). Priority order for
picking this back up:

1. **Finish analyzing the gamma-profile's non-monotonicity** (see above) — refine the grid around
   g∈[0.94, 0.99] with more points and multistart per point (the current single-path continuation
   warm-start may be landing in different local basins at different g and reporting whichever it finds
   first, not necessarily the true `min_A Delta(g,A)`). This is now the most likely source of new
   scientific insight.
2. **Wire `MuSigmaPowCache` into the live oracle path** (Continuation 6 prompt section 1) —
   `moments_fast.jl::MuSigmaPowCache`/`EK_moments_gammanorm_directgp_fast!` are equivalence-tested but
   still standalone. `PsiObjectiveBundleImplicit` (`cc_algo/PsiObjectiveBundle.jl:128`) is a
   **mutable struct** — the low-risk wiring path is: build `ctx` normally via `d4_exact_setup()`, then
   `pow_cache = MuSigmaPowCache(ctx.U, ctx.obj.γ.Uσ); ctx.obj.moments! = (K,G,θ,U,obj) ->
   EK_moments_gammanorm_directgp_fast!(K,G,θ,U,obj,pow_cache)` — a simple field mutation, NOT a full
   context-builder mirror. This was identified but not implemented/tested this session. After wiring,
   re-profile `evaluate_fullA_fast` and the live F+G callback to record the REALIZED gain (expected
   modest, ~1.1-1.4x on the moment-build component per `docs/fullA_moment_construction_audit.md`).
3. **Live bandwidth caching** (`h_mode=:cached`) with the strict revalidation policy the Continuation 6
   prompt specifies (periodic forced revalidation, winner-hash/switching-mass triggers, actual-vs-
   predicted reduction checks, revert-to-adaptive on any revalidation failure) — currently only
   equivalence-tested as a kernel, never run live in KNITRO with a real policy.
4. **Nested-W** (W=8000/20000/80000 as prefixes of ONE draw pool, not independent redraws) — still
   fully unattempted across two continuations now.
5. **Full gamma-profile deliverables**: the machine-readable table (solution hashes, cluster IDs, cold
   rechecks), lower-direction profile from g=1, refinement of every `profile_Delta(g)=delta` crossing,
   deeper stationarity reassessment (directional secants at several bandwidths, gravity-tangent-only
   directions, external KKT in original units) at the FINAL candidates.
6. **Gated D-scaling redo** using the NOW-FIXED gradient + (once done) the wired moment cache — the
   D=6 pilot from Continuation 5 is still valid but predates both this session's bug fix and the
   not-yet-done moment-cache wiring.
7. **Fair hard-vs-smoothed comparison**, properly apples-to-apples (same process, back-to-back) — the
   Continuation 5 version reused older smoothed-route numbers.

## NEW USER ADDENDUM — investigate compressed winner-form moment representation

The user has ALSO asked (this session, not yet started) for an investigation into whether the
factual bilateral moments can be represented and used in a **compressed winner form** instead of
materializing the dense `W × D²` moment matrix. Full text below, verbatim, since this is a substantial
new task that should be scoped and started fresh, not summarized lossily:

> Please investigate whether the factual bilateral moments can be represented and used in compressed
> winner form rather than constructing a dense draw-by-bilateral-moment matrix.
>
> For each draw `s` and destination `d`, the factual share moments should have a structure resembling
> `G_{·d,s} = v_{sd}(e_{w_sd} - λ̂_{·d})`, where `w_sd` is the winning origin and `v_sd` is its winning
> CES contribution. Verify the exact production formula and signs from code before relying on this
> expression.
>
> **First profile the current implementation.** Separately time and count allocations for: score
> construction; hard winner search; winning CES-value calculation; zeroing/allocating the output
> moment matrix; writing the winning entries; applying the observed-share centering terms;
> counterfactual moment construction; any matrix reshaping/copying afterward. Determine whether the
> current constructor already exploits the one-winner structure or fills the full (W×D²) matrix
> explicitly.
>
> **Implement a compressed representation** (additive diagnostic only, do not replace the trusted
> dense implementation initially): `winner[s,d]`, `winning_value[s,d]`, separate counterfactual moment
> columns, fixed target-share vectors.
>
> Using the compressed representation, implement the fixed-dual contraction directly. For bilateral
> multipliers `β_od`, use: `Σ_{o,d} β_od G_{od,s} = Σ_d v_sd [β_{w_sd,d} - Σ_o λ̂_od β_od]`.
>
> Verify exact equivalence to dense-matrix multiplication across candidate points, random feasible
> points, all dual vectors tested, and several (D,W) combinations.
>
> **Investigate a compressed CC-inner bundle** (if the scalar contraction succeeds): dual objective,
> dual gradient, primal-weight recovery, moment residuals — without materializing the dense factual
> moment matrix. For the dual gradient, accumulate winner-origin bucket totals and destination totals,
> then apply the target-share correction once per destination. Assess whether exact Hessian-vector
> products can be computed directly from the compressed moments. Do not rewrite the production inner
> solver until the compressed objective/gradient/any Hessian products pass strict equivalence tests.
> Retain a dense-matrix materialization mode for diagnostics/final verification.
>
> **Benchmark** dense vs. compressed at D=4/W=8000, D=4/W=80000, D=6/8/10 W=8000 (where practical).
> Report: moment-construction time; CC objective/gradient/Hessian-or-HVP time; complete warm and cold
> inner-solve time; exact value-call time; peak memory and allocations; `L_fix` base-state time;
> end-to-end short outer-run time. Test draw-level or destination-level threading for ordinary exact
> evaluation, but avoid nested threading when coordinate-level `L_fix` parallelism is active.
>
> Do not change the economic moments, tie-breaking rule, or exact hard estimand. Keep this
> optimization isolated until dense and compressed results agree to machine precision or a documented
> tight tolerance.

**Important connection to this session's own finding**: the compressed representation's assumption of
"one winning origin" is EXACTLY the assumption that just broke in `build_lfix_base_cache` (see above)
— on an exact tie, there are TWO winning origins, not one. Whoever picks up the compressed-moment task
should build the SAME kind of tie detection/fallback in from the start (reuse `detect_price_ties`/
`TiedWinnerError` from `lfix_incremental.jl` directly, don't re-derive) rather than discovering the
same edge case again the hard way. This is a genuinely useful piece of prior art to hand forward.

## Operational notes

- All equivalence-test discipline in this investigation is real and load-bearing — re-run
  `test_oracle.jl`, `test_oracle_profiled.jl`, `test_lfix_incremental.jl`, `test_composite_gradient_fast.jl`,
  `test_moments_fast.jl` before trusting anything, and after ANY change to `lfix_incremental.jl`,
  `composite_gradient_fast.jl`, or `moments_fast.jl` specifically (all touched this session).
- Must run on `demand.mit.edu` (KNITRO license, demand-side only).
- `source .knitro_env.sh` before any Julia invocation that touches KNITRO.
- Set `JULIA_NUM_THREADS` explicitly per run (defaults to 1) — 8 was used for this session's live runs;
  the user says they'll generally provide 20.
- Push new docs/handoffs to Dropbox at
  `dropbox:Gravity robustness/Analysis/Server Output/fullA_d4_continuation6_2026-07-18/` before ending
  a session (established convention) — NOT YET DONE for this session's outputs, next continuation
  should do this or the current one should before fully stopping.
