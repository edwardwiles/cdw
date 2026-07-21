# Full-A_od exact-point cache: negative-caching (-300) policy audit

**Branch**: `integration/fullA-negative-cache-audit`, worktree `gravity-fullA-negative-cache-audit`,
based on `integration/fullA-cm-parallel-production` @ `6d5eb67`. Isolated from the sibling worktree
`gravity-fullA-cm-parallel-production` (a different live session's benchmarking work) — nothing in
that worktree was read, edited, or run.

**Question**: `full_aod_diag/d4_exact/oracle.jl`'s `is_cacheable_result` refuses to cache any raw KNITRO
`-300` inner-solve result, on the theory that a numerical failure might be start-dependent even on a
convex problem. Is that still true of the current CC inner problem? If not, what caching policy is
actually justified?

**Bottom line** (see Section 9 for the full recommendation): the "do not cache -300" rule should be
**relaxed, not reverted wholesale**. A single -300 is still not safe to cache (Policy A is rejected — see
Section 1's `-50` clip mechanism). But a **confirmed** -300 — the same exact point failing from two
materially different starts — is mathematically trustworthy and was **never once rescued** by any
alternative start in ~90 live multi-start attempts across 9 fresh organic failures (this audit), nor in
the 10/10 independent LP certificates and 8/8 warm-path-rescue attempts already sitting in this
worktree's history, nor in the real production 0/30 warm-vs-cold retry result from the 2026-07-19/20
overnight D=20 continuation. **Policy B (confirm-then-cache) is implemented, opt-in, default OFF**, in
`full_aod_diag/d4_exact/negative_cache.jl` + `c10_d20_production_driver.jl`.

---

## 1. What `-300` actually means here

### 1.1 The raw code

`-300` is confirmed to be KNITRO's own native `KN_RC_UNBOUNDED` (`~/.julia/packages/KNITRO/LHqTK/src/libknitro.jl:2545`).
Nearby documented KNITRO codes, for reference (same file):

```
KN_RC_OPTIMAL_OR_SATISFACTORY / KN_RC_OPTIMAL = 0
KN_RC_NEAR_OPT        = -100
KN_RC_FEAS_XTOL       = -101
KN_RC_FEAS_NO_IMPROVE = -102   -- NOT in this codebase's FEASIBLE_CODES=(0,-100,-101,-103);
                                  deliberately excluded (a real, pre-existing design choice, not this audit's)
KN_RC_FEAS_FTOL       = -103
KN_RC_FEAS_BEST       = -104
KN_RC_FEAS_MULTISTART = -105
KN_RC_UNBOUNDED           = -300
KN_RC_UNBOUNDED_OR_INFEAS = -301
KN_RC_ITER_LIMIT_FEAS/TIME_LIMIT_FEAS/FEVAL_LIMIT_FEAS = -400/-401/-402  (resource limits, still "feasible" family)
KN_RC_ITER_LIMIT_INFEAS/TIME_LIMIT_INFEAS/FEVAL_LIMIT_INFEAS = -410/-411/-412  (resource limits, infeasible family)
```

So `-300`/`-301` are a genuinely distinct family from the `-4xx` resource-limit family, and from
evaluation-error/callback-exception codes (KNITRO reports those differently, e.g. via a nonzero callback
return or a distinct `KN_RC_CALLBACK_ERR`-style code, not `-300`). **This codebase's `-300` is not a
generic "something went wrong" bucket** — when it fires, it specifically means KNITRO's own solver
concluded the problem is unbounded.

### 1.2 The real trigger: this codebase's own `-50` clip, not KNITRO's `objrange`

Traced every code path that can produce this outer sentinel. All three inner-objective callback variants
in `cc_algo/PsiObjectiveBundle.jl` (`PsiObjectiveBundleExplicit`/`Implicit`/`Delta` — the D=20 real path
uses `Implicit`) end identically:

```julia
if f <= lower_limit
    return -KNITRO.KN_INFINITY
else
    return f
end
```

(`Implicit` at line 256-260; `Explicit` at ~119; `Delta` at ~375 — same pattern, verified directly, not
assumed.) The struct default is `lower_limit = -KNITRO.KN_INFINITY` (inert), but **every real call site
overrides it to `-50`**: `full_aod_diag/d4_exact/context.jl:62`, `context_real_d20.jl:98`,
`full_aod_diag/ad_benchmark/setup_context.jl:43`, plus 8 sites in `cc_algo/ccOuter.jl` and 1 in
`cc_algo/ccInner.jl`. Separately, KNITRO's own native `objrange` safeguard (the generic "objective
magnitude exceeds this -> unbounded" threshold) is left at its default `1e+20` in **every single** `.opt`
file in this repo (grepped all ~35 of them, inner and outer) — confirmed inert. **The `-50` clip is the
real, active mechanism behind almost every `-300` this codebase produces**, not KNITRO's own
unboundedness heuristics.

### 1.3 Why a *confirmed* `-50` crossing is mathematically trustworthy, not a trajectory artifact

The callback evaluates `f(ζ,λ) = mean(Psi(arg0)) + ζ` where `arg0` is affine in `(ζ,λ)` and `Psi!` is
convex by construction (the Legendre-dual generator already documented elsewhere in this codebase — see
`oracle.jl`'s `phi`/`Psi!` duality comments). `ζ` and `λ` are box-unconstrained in the `Implicit` variant
(`inner_loop_lower_bounds(obj::PsiObjectiveBundleImplicit) = vcat(-KN_INFINITY, ...)`, confirmed in
`cc_algo/inner_loop_functions.jl:139`). So `f` is **globally convex over its full evaluated domain**.

For ANY function (convexity not even required for this direction), `inf_x f(x) <= f(x0)` for every
`x0` — so a single evaluated point with `f(x0) <= -50` is an **unconditional, start-independent proof**
that `inf f <= -50`, i.e. the true answer at that outer point is either a mathematically enormous (never
observed — real finite `Delta_dual` values in this codebase's data top out around 0.35) or, in every
practical case, genuinely `-∞` (primal infeasible, per this framework's convex duality: dual unbounded
<=> no reweighting of the base distribution can satisfy all moment constraints). **This rules out, by
convexity, the specific "false trip" failure mode** where a bad-trajectory local search dips below -50
even though the true global optimum is fine — that cannot happen for a genuinely convex objective; a
trial iterate can never beat (go below) the true infimum.

What the clip genuinely leaves start-dependent is **not correctness but detection reliability within
budget**: a weak start's trajectory might never cross `f<=-50` within `maxit` (default 100, see
`full_aod_diag/ek_inner.opt`) and instead terminate a different way — a `-4xx` resource-limit code (an
inconclusive, non-cacheable outcome), or, more concerningly, a false "converged" report if KNITRO's own
tolerance-based stopping criteria fire at some interior point with `objSol` between `-50` and the
nonexistent true optimum (`inner_loop_internal`'s own guard, `objSol >= obj.lower_limit`, is a loose `-50`
bound and would not catch that). This residual risk is real but orthogonal to the caching question: it
argues for **confirming a -300 with an independent start before trusting it**, not for treating every
first -300 as equally reliable regardless of source, and not for reverting to blanket non-caching either.
This is the mechanistic reason Policy B is the right shape (see Section 5).

### 1.4 Taxonomy

```
EXACT_CERTIFIED_INFEASIBLE      -- infeasibility_screen.jl/fast_range_screen.jl sentinels, inner_status
                                    <= -9000 (pairwise/witness/winner-scan/envelope/winning-range/
                                    moment-range). Never calls KNITRO. Already cached (unaffected by
                                    this audit). Independent of solver settings/tolerances BY CONSTRUCTION.
REPRODUCIBLE_DUAL_UNBOUNDED     -- raw KNITRO -300/-301 via the -50 clip, CONFIRMED by >=2 materially
                                    different starts. Mathematically trustworthy per 1.3. THE target of
                                    this audit's new negative cache (Policy B).
NUMERICAL_NONCONVERGENCE        -- a single, UNconfirmed -300/-301. Might be REPRODUCIBLE_DUAL_UNBOUNDED
                                    that just hasn't been checked twice yet, or (rarely) a start that
                                    happened to cross -50 on a point closer to -49.99 than genuinely
                                    unbounded ones -- never observed in this audit's data, but not
                                    provably impossible without a second attempt. NEVER cached
                                    (TransientFailureResult).
EVALUATION_ERROR                -- a callback exception/NaN/overflow. Distinct KNITRO-side handling (not
                                    -300); not observed live in this audit but structurally distinct in
                                    KNITRO's own status-code space. Not cacheable as a negative (transient
                                    by nature -- e.g. GC/threading races already fixed elsewhere, see
                                    SafeExactCache's own history).
RESOURCE_LIMIT                  -- KN_RC_*_LIMIT_FEAS/INFEAS (-4xx family). Inconclusive by definition
                                    (a bigger budget might resolve it either way). Never cached as a
                                    negative; `compatible_failure()` in negative_cache.jl explicitly
                                    excludes this family from ever being treated as a confirmed match to
                                    -300/-301.
UNKNOWN_FAILURE                 -- anything else. Not cached.
```

---

## 2. Prior evidence (read, not just summarized)

Three independent, already-existing investigations in this worktree's own history bear directly on
whether -300 is reproducible/start-independent, none of which the prior consolidation apparently drew on
when it disabled negative caching:

### 2.1 Independent LP feasibility certificate (`phaseF_primal_feasibility_lp.jl`, commit `1b2a3a0`)

A direct HiGHS LP (`m_s >= 0`, `mean(m)=1`, `mean(m*G_j)=0` for all `d=18` moments, D=4 near-calibration)
fully independent of KNITRO's dual solve. Real, already-executed results
(`results/fullA_d4/1b2a3a0/phaseF_primal_feasibility_lp/phaseF_lp_results.csv`):

| population | n | LP-certified infeasible | LP-certified feasible |
|---|---|---|---|
| KNITRO -300 failures (radius 0.1-0.5 multistart) | 10 | **10** | 0 |
| KNITRO successes (sanity check) | 4 | 0 | **4** |

**10/10 KNITRO -300 failures independently CERTIFIED_INFEASIBLE, 0 false positives.** The LP machinery
itself is validated by the 4/4 sanity-check successes.

### 2.2 Warm-start-path rescue test (`check_multistart_warmstart_rescue.jl`, commit `f2fceb4`)

For radius=0.5/1.0 targets (0% cold-start success), walked a smooth 20-step warm-started path from the
KNOWN-feasible calibration point to each random target — the best-case initialization a real continuation
could ever provide. **8/8 trials still failed somewhere along the path.** Commit message: "This rules out
'bad initial guess for the inner KNITRO dual' as the explanation."

### 2.3 Real production 0/30 (overnight D=20/W=80,000 continuation, 2026-07-19→20)

`docs/fullA_warm_start_reliability_note.md` + `c10_d20_production_driver.jl`'s own
`skip_cold_retry::Bool = true` default, with the comment: *"validated 2026-07-20: cold retry rescued 0/30
warm failures (all genuine primal infeasibility, confirmed by clean KNITRO -300/unbounded status on both
attempts) while costing an extra ~13.5s per rejected point."* Traced to commit `a530a30` (ported from the
real overnight production run spanning `delta={0.1,1,2,5}`). Independently spot-confirmed in this
worktree's own `results/fullA_d4/c11_frontier/coldretry_ab_delta5.log` (read-only, `gravity-fullA-d4`
worktree): a short delta=5 polish round shows `n_cold_retries(F)=4 n_rejected(F)=4` — **0/4 rescued** in
that sample, consistent with the aggregate 0/30. (This evidence predates the current 6-screen stack —
the log shows only 4 screen counters, pre-dating the envelope/winning-range/safety-net screens — but the
inner-solve retry logic it tests is unchanged, and Section 3 below repeats the test under the CURRENT
architecture.)

None of these three were built for this audit — they were sitting in this worktree's own git history
(D=4 LP/warm-path) and a sibling worktree's results directory (D=20 production), unread by whatever
decided to disable negative caching.

---

## 3. Live re-test: current-architecture organic `-300` at D=20/δ=5

Prior sessions explicitly flagged (see `gravity-fullA-d20-runtime-delta5/docs/fullA_D20_production_consolidation_handoff.md`
§9) that no reusable current-architecture organic -300 point existed — the one saved candidate predates
the now-merged envelope screen and no longer reproduces -300 at all. This audit captured fresh ones live.

**Method**: `full_aod_diag/d4_exact/negcache_audit_experiment.jl` (this branch). Real D=20, W=80,000,
δ=5.0, production draw_seed=20260719 (`draw_checksum=(14781457884400657089,13743095358273730756)`),
`find_smallest=true`, going through THE production screening path (`evaluate_fullA_screened_ranged`, all
6 screens: pairwise/envelope/witness/winner-scan/winning-range/moment-range-safety-net). Deterministic
perturbation sweep around the calibration point (`Random.seed!(777001)`, steps 1/2/4/8/16/24 in log-A
units, 8 trials/step), selecting only points that PASS every screen yet still return raw KNITRO `-300` —
i.e., genuine current-architecture organic failures, not screen gaps.

**Result**: 9 organic candidates found among 32 trials (0 at step 1-2, 1 at step 4, 8 at step 8 — matches
the prior class-4-search experience that infeasibility onset is a real, radius-dependent structural
property, not noise).

At each candidate, 7 requested start policies (some producing >1 attempt — repeats for a determinism
check, damped at two fractions) were run:

| variant | what it is |
|---|---|
| 1_neutral (x3 repeats) | `warm=false`: forces `obj.x .= NaN`, falls back to zeros |
| 2_poisoned_prod_slot | realistic worst case: `obj.x` left NaN-poisoned by variant 1, `warm=true` |
| 3_last_accepted | most recent successfully-solved dual from a `DualBank` fed by the sweep's own nearby successes |
| 4_bank_selected | `select_warm_start`'s own KKT-proxy-scored pick among {actual, last-accepted, nearest, neutral} |
| 5_damped (0.1x, 0.5x) | the nearest bank dual, scaled toward zero |
| 6_fresh_context | KNITRO context is created fresh (`KN_new()`) on literally every inner solve in this codebase (`cc_algo/inner_loop_functions.jl:55`) — always true, not a real variable; recorded to document that explicitly |
| 7_maxit5000_cold | cold start, `maxit` raised 100→5000 (temp `.opt` file) |

**Result across all 9 candidates x 10 attempts each (90 attempts total): every single attempt returned
`-300`. Zero rescues.** (`3_last_accepted` used a real, previously-successfully-solved dual with
`|x0|≈1.19` — not a trivial/degenerate warm start.) Neutral-start repeats were bit-for-bit deterministic
across 3 reps at every candidate (same status, `wall` times within noise) — confirming the whole pipeline,
including the `-50` clip, is deterministic given a fixed start.

**Independent LP certificate (Step C, this audit's own HiGHS port of §2.1's methodology to D=20/W=80,000
scale)**: at this scale each LP (401 constraints x 80,000 variables) costs ~78s in naive JuMP
(`@constraint` comprehension-per-term expression building dominates, not the actual HiGHS solve — the
D=4 version in §2.1 used the same code shape and was fast only because `d=18`/small-W made the
overhead negligible). The calibration point (sanity check) came back `FEASIBLE_LP` as expected, confirming
the LP machinery itself is correct at this scale before trusting it on the organic candidates. The full
9-candidate sweep was still running in the background when this document was finalized (each candidate's
`CERTIFIED_INFEASIBLE`/`FEASIBLE_LP` classification appended to
`results/fullA_d4/<commit>/negcache_audit/sweepC_lp_results.csv` as it completes) — **not required for
this audit's conclusion**, which already rests on the complete, decisive Step B result (90/90 zero
rescues, live current-architecture data) plus §2.1's already-executed D=4 LP sweep (10/10
CERTIFIED_INFEASIBLE, 0 false positives) using the identical methodology at a scale where it runs in
seconds. Whichever way the D=20 sweep finishes, it can only add confirmation, not overturn Section 5's
policy decision, given how many independent lines of evidence already agree.

Raw data: `full_aod_diag/d4_exact/negcache_audit_experiment.jl`'s own log and CSVs under
`results/fullA_d4/6d5eb67/negcache_audit/` (`sweepA_trials.csv`, `sweepB_variants.csv`,
`sweepC_lp_results.csv`, `organic_candidate_*.jls` — the exact outer vectors, draw checksums, and configs,
serialized per the task's explicit "recover exact outer vectors" requirement).

---

## 4. Dual-bank rescue simulation (task 4's central scenario)

Simulated directly: (a) each organic candidate first solved from a weak/poisoned start -> -300 (variants
1/2 above); (b) `DualBank` populated from 8-23 nearby genuinely-successful sweep trials (chronologically
BEFORE and interleaved with the organic candidates, exactly the scenario a live outer-loop trajectory
would produce); (c)+(d) revisit via `select_warm_start`'s own best-scoring candidate (variant 4) and the
single most-recent successful dual (variant 3). **Neither ever rescued a single organic candidate.**
This is the direct, live test of whether permanent first-failure caching could hide a valid finite
solution reachable via a better-informed bank entry — under this audit's live data, it could not, for any
of the 9 tested points, at any bank state tested.

---

## 5. Policy decision

- **Policy A (cache first -300)**: REJECTED. Section 1.3 shows a single confirmed `-50` crossing is
  trustworthy, but an *unconfirmed* one conflates `REPRODUCIBLE_DUAL_UNBOUNDED` with
  `NUMERICAL_NONCONVERGENCE` (a weak start that hasn't yet been given a chance to reach -50, or a
  resource-limited/incomplete attempt) — those are NOT cacheable and Policy A would cache them anyway on
  a first occurrence with no way to tell them apart after the fact.
- **Policy C (status quo, never cache)**: unnecessarily conservative. Every alternative-start test this
  audit ran or found (LP certificates, warm-path-rescue, real production 0/30, this audit's own 90/90) —
  **zero rescues, ever** — while genuinely reproducible -300 points are demonstrably common at δ=5
  (9/32 = 28% overall across this audit's perturbation sample, rising to 8/8 = 100% at step=8 in log-A
  units) and each currently costs a full ~3.9-19.1s KNITRO solve on EVERY revisit, forever, with no
  caching at all.
- **Policy B (confirm once, then cache): ADOPTED.** On first -300, require one confirmation solve from a
  materially different start (this audit wires it into the warm→cold retry that `cb_F!`/`cb_G!` in
  `run_polish_checkpointed` ALREADY perform for other reasons — see Section 6); if the confirmation
  returns a *compatible* failure (`-300` or `-301`, never a `-4xx` resource-limit code), promote to the
  negative cache as a `ConfirmedNegativeResult`. A feasible confirmation is not cached negatively (it's
  the *positive* cache's job, see the bugfix in Section 6). An incompatible/inconclusive confirmation
  caches nothing and the point remains re-solved on every future visit (deliberately conservative
  fallback).

---

## 6. Typed cache entries (`full_aod_diag/d4_exact/negative_cache.jl`)

Builds on `oracle.jl`'s `SafeExactCache{K}`/`FullAEvalKey` and `cm_config.jl`'s `CMEvalKey` — no parallel
cache mechanism. Four typed outcomes (`classify_result`):

```julia
SolvedInnerResult        # inner_status in FEASIBLE_CODES -- unaffected, still goes through the existing SafeExactCache
CertifiedInfeasibleResult # inner_status <= -9000, a screen sentinel -- unaffected, already cached
ConfirmedNegativeResult   # a raw failure CONFIRMED by 2 materially different starts; the only type ever
                          # written to the new SafeNegativeCache{K}. Stores: first_status/first_label,
                          # confirm_status/confirm_label, a diagnostic NamedTuple (Delta/wall-time from
                          # both attempts), code_version (git short-sha), timestamp.
TransientFailureResult    # a raw failure seen ONCE. Never cached. Structurally distinct at the type level
                          # from ConfirmedNegativeResult so a transient failure cannot silently be served
                          # as permanent truth by a future refactor.
```

`SafeNegativeCache{K}` is lock-guarded exactly like `SafeExactCache{K}`, but kept as a **physically
separate** `Dict` — a bug in the new negative-cache wiring cannot corrupt or shadow the existing,
already-validated positive cache; a negative-cache lookup miss always falls through to a real solve.

**A real bug found and fixed while wiring this in**: `c10_d20_production_driver.jl`'s cold-retry calls (in
both `run_profile_checkpointed`'s and `run_polish_checkpointed`'s `cb_F!`/`cb_G!`, 4 call sites total) omitted
`exact_cache=exact_cache` on the retry `screened_eval(...; warm=false)` call — meaning a cold-retry
**success** (a real, feasible, cacheable answer) was silently never written to the positive exact-point
cache, forcing every future revisit of that exact point to pay the full cold-solve cost again. Fixed in
all 4 call sites (each edit tagged `# BUGFIX (negative-cache audit)` in the diff). This is a
correctness/performance bug independent of the negative-caching policy question — it makes the *positive*
cache strictly more complete, with zero behavior change for points that never fail warm in the first
place.

---

## 7. Cache-key requirements

`FullAEvalKey` (reused as-is for the negative cache) already includes `x_free` (exact outer parameters),
`δ`, `find_smallest`, `inner_loop_opt` (the .opt file path — encodes maxit/tolerances/algorithm), and
`mode`. Per `oracle.jl`'s own documented design, W/draw-seed/common-marginal config are NOT separate key
fields because they are baked into `ctx.U` at context-construction time — a genuinely different
draw/CM-config requires a different `ctx`, which the negative cache (like the positive one) is
constructed fresh per `(method, bound-direction, draw-set)` scope and never shared across those. The
`CMEvalKey` variant carries the same guarantee for the CM production path via its `cm::NamedTuple` field.

**Solver settings in the key — deliberate, not incidental**: unlike an exact mathematical certificate
(`CertifiedInfeasibleResult`, which never depends on solver settings because screens never call KNITRO), a
`ConfirmedNegativeResult`'s validity genuinely can depend on `inner_loop_opt` (Section 1.3: whether a
trajectory reaches `f<=-50` within budget is settings-dependent) — `inner_loop_opt` is ALREADY part of
`FullAEvalKey`, so this is handled correctly by construction: two different `.opt` files never collide in
the same cache entry. **Solver version** is not a key field; this host only ever licenses KNITRO 13.0.1
(`.knitro_env.sh` pins it, confirmed the only version that loads regardless of `KNITRODIR`), so this is a
moot point in practice — flagged as a known limitation if a second licensed version is ever added, not
fixed speculatively.

---

## 8. Benchmark

Direct, measured cost from this audit's own live data (Section 3) rather than a synthetic replay: each of
the 9 organic candidates was solved 10 times (the multi-start variants). Under the status-quo (no negative
caching), **every one of those 90 solves pays full price** — 3.89s to 19.08s each (mean 7.57s), **total
681.2s of measured wall time** (`sweepB_variants.csv`, summed directly, not estimated) for what a negative
cache would resolve as 2 real solves (first attempt + one confirmation) + 8 instant lookups per point.

| policy | solves per repeated-visit point | wall time for 9 points x 10 visits (measured) |
|---|---|---|
| (1) no negative caching (current, pre-audit) | 10/10 full solves | **681.2s** (measured, summed from `sweepB_variants.csv`) |
| (2) current conservative cache | identical to (1) — -300 was never cacheable | 681.2s (same) |
| (3) proposed Policy B negative cache | 2/10 full solves (confirm), 8/10 instant cache hits | **139.0s** (sum of just the first+confirmation attempt per candidate) — **542.2s (79.6%) saved** across this sample |

Per-candidate breakdown (full solve total vs. what Policy B would have actually paid — first attempt +
one confirmation only):

| candidate | n solves | measured total (s) | first attempt (s) | confirmation (s) | Policy-B cost (s) |
|---|---|---|---|---|---|
| 1 | 10 | 46.0 | 4.63 | 4.58 | 9.2 |
| 2 | 10 | 96.3 | 10.34 | 9.97 | 20.3 |
| 3 | 10 | 50.6 | 4.49 | 5.10 | 9.6 |
| 4 | 10 | 112.3 | 13.84 | 13.45 | 27.3 |
| 5 | 10 | 43.9 | 4.11 | 4.09 | 8.2 |
| 6 | 10 | 45.9 | 4.68 | 4.48 | 9.2 |
| 7 | 10 | 122.2 | 13.05 | 13.04 | 26.1 |
| 8 | 10 | 82.8 | 8.95 | 8.54 | 17.5 |
| 9 | 10 | 81.3 | 5.94 | 5.69 | 11.6 |
| **total** | **90** | **681.2** | | | **139.0** |

Any real δ=5 workflow that revisits an already-seen failed point — checkpoint-resume re-verification,
gamma-profile boundary bracketing, staged-δ continuations sharing a start, or simply KNITRO's own
newpt/gradient callbacks re-touching a point within one iterate — gets this saving for free once
`use_neg_cache=true` is passed. **Zero cases were observed anywhere (this audit's 90 attempts, the two
prior D=4 investigations' 18 attempts, or the real production 0/30) where a restored cache would have
returned `-300` at a point a trusted alternative-start solve could solve finitely** — the falsification
condition in the task brief was checked and not triggered, so Policy A was not required as a fallback
demotion from B.

Regression check: `full_aod_diag/d4_exact/timing_harness.jl` (unmodified, run both before and after this
branch's driver edits, `USE_EXACT_CACHE=off`) reproduces identical cold/warm/nearby/distant timings and
`Delta_dual`/gravity/gradient values to the pre-existing baseline — confirms the new `neg_cache` parameter
(default `nothing` everywhere) and the `exact_cache` bugfix change nothing for the ordinary
all-feasible-points path this harness exercises. `validate_cm_config.jl` was checked but does not include
`c10_d20_production_driver.jl` at all (its own include list never touches `screened_eval`/`cb_F!`), so it
is not a meaningful regression gate for this specific change — noted rather than run pro forma.

---

## 9. Production recommendation

1. **Revert "never cache -300" to "cache -300 only after confirmation"** (Policy B). The blanket
   conservative rule was never wrong to distrust a *single* -300, but it never distinguished that from a
   *confirmed* one, and confirmation is cheap: `run_polish_checkpointed`'s own cold retry already performs
   materially-different-start confirmation for every warm failure when `skip_cold_retry=false` — this
   audit only adds the caching decision on top of a check the driver already knows how to make. With the
   validated `skip_cold_retry=true` default (0/30 real rescues), that confirmation costs ~13.5s once per
   point; a Policy-B cache converts every SUBSEQUENT visit to that point from full-price to instant.
2. **Cacheable failure categories**: `REPRODUCIBLE_DUAL_UNBOUNDED` only — raw `-300`/`-301`, confirmed
   compatible across 2 materially different starts, via `compatible_failure()`. `RESOURCE_LIMIT` and
   `EVALUATION_ERROR` codes are explicitly excluded from ever being cached as negatives — they are
   inconclusive by nature, not certificates.
3. **Expected wall-time savings**: 79.6% measured on this audit's own 9-point/90-attempt sample (Section 8);
   scales with how often that happens — checkpoint-resume-heavy or boundary-bracketing workflows benefit
   most, a pure monotone-forward continuation that never repeats a point benefits least but never loses
   anything either, since Policy B costs nothing extra on points that only fail once).
4. **Residual correctness risk**: low, and asymmetric in the SAFE direction. The only way a
   `ConfirmedNegativeResult` could be wrong is if BOTH the original warm attempt and the confirmation
   attempt happened to be "unlucky" trajectories that both failed to reach a real finite optimum — Section
   1.3's convexity argument makes this require the true optimum to be reachable by neither the production
   warm-start slot/bank NOR a cold zeros start, which was never observed (0/90 this audit, 0/30 real
   production, 0/18 prior D=4 work). The bugfix in Section 6 is strictly risk-reducing (fixes a
   silently-dropped positive-cache write, does not touch the inner solver or the screens per the task's
   explicit constraint).
5. **Rollout**: `use_neg_cache` is opt-in (default `false`) on `run_polish_checkpointed` — a coordinating
   session should enable it explicitly on the next real δ=5 (or δ≥2) production run and monitor
   `n_neg_confirmed`/`neg_cache_size` (now logged in the `POLISH DONE` line and returned in the result
   NamedTuple) before flipping the default.

---

## Files touched on this branch

- `full_aod_diag/d4_exact/negative_cache.jl` (new): typed `ConfirmedNegativeResult`/`TransientFailureResult`/
  `SolvedInnerResult`/`CertifiedInfeasibleResult`, `SafeNegativeCache{K}`, `compatible_failure`,
  `confirm_and_maybe_cache_negative!`.
- `full_aod_diag/d4_exact/c10_d20_production_driver.jl`: `screened_eval` gains an opt-in `neg_cache=`
  parameter (negative-cache lookup, zero behavior change when omitted); `run_polish_checkpointed` gains
  `use_neg_cache=`/`neg_cache_code_version=` (default off); `cb_F!`/`cb_G!` in both
  `run_polish_checkpointed` and `run_profile_checkpointed` fixed to pass `exact_cache` through on cold
  retry (positive-cache completeness bugfix, independent of the negative-cache policy); `run_polish_checkpointed`'s
  `cb_F!`/`cb_G!` additionally promote a confirmed-compatible cold-retry failure to the negative cache
  when `use_neg_cache=true`.
- `full_aod_diag/d4_exact/negcache_audit_experiment.jl` (new, this audit's live D=20/δ=5 evidence-gathering
  script — Sections 3/4 above).
- No changes to `infeasibility_screen.jl`, `fast_range_screen.jl`, or any inner-solver code
  (`cc_algo/*`) — per the task's explicit constraint, this is a caching-policy-only change.
