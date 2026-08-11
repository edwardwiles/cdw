# Why the cross-family searches were starved, and what fixed it

2026-08-11. Branch `feature/ozc-cross-2026-08-09`, worktree `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`.
Companions: `CROSS_8H_UPPER_BOUND_RUNS_2026-08-10.md` (the δ=1 runs this explains),
`HCZ_PROFILE_AND_OPTIMIZATION_2026-08-11.md` (the Hessian-kernel half).

## 0. Summary

The 8-hour δ=1 runs each terminated on a `−102` stall with hours of budget unspent, having produced
only 100 and 28 usable evaluations. This investigation found why, and the cause was not the kernels.

| # | change / finding | measured effect | status |
|---|---|---|---|
| 1 | **`opttol`/`opttol_abs` 1e-12 → 1e-10** | **3.25× usable evaluations**; `−102` class eliminated | **flipped**, gated on both CROSS families |
| 2 | H_CZ `:j_parallel` | 2.8× on that block (~10% of an outer eval) | flipped, gated |
| 3 | `INNER_KEEP_LAST_GOOD_X` (don't NaN a good dual on failure) | cold starts 70%→3%; 1.33× on accepted solves | implemented, **default off** |
| 4 | `use_dual_bank = true` | 4 → 6 accepted solves | **not** flipped (small sample) |
| 5 | `canonical_price_precompute` μ-guard | 0.109 s/call, ≈0.4% of wall | flipped |
| — | support screens A and B | **0 of 18** failures caught | rejected |
| — | dual-bank pre-check ("C") | **0** solves knowable at the warm start | rejected |
| — | `ThresholdAbortState` ("D") | identical to the existing `lower_limit` | rejected |

## 1. The headline: the inner solve was chasing an unreachable tolerance

`full_aod_diag/ek_inner.opt` set `opttol = opttol_abs = 1e-12`. This repo's own memory
(`knitro-minus100-is-opttol-1e-12-vs-achievable-floor`) records the **achievable optimality floor on
this problem as ~1e-11**. So every inner solve was pursuing an accuracy it could not reach, grinding
until KNITRO gave up — which is exactly the observed signature: healthy solves returning `−100`
(`KN_RC_NEAR_OPT`, "stopping tests satisfied within a factor of 100", i.e. it got to ~1e-10) after
~38 Hessian calls, rather than `0`.

A/B through the **real driver**, same start point (the 8h run's own incumbent), same 1200 s budget,
only the tolerance differing:

| | opttol 1e-12 | opttol 1e-10 |
|---|---|---|
| **usable evaluations (`n_eval`)** | **4** | **13 (3.25×)** |
| inner solves | 22 | 40 |
| **`−102` discards** | **5 solves, 646.5 s** | **0** |
| `status 0` (true convergence) | 0 | 12 |
| accepted median wall | 132.8 s | **60.6 s** |
| accepted median Hessian calls | 38 | **20** |

**Accuracy gate**, Δ at eval 1 (same point):

| family | 1e-12 | 1e-10 | relative |
|---|---|---|---|
| OZC-CROSS | 1.0000005161479442 | 1.000000515905411 | **2.425e-10** |
| CM+ZC-CROSS | 0.039528604824169755 | 0.03952860480883394 | **3.880e-10** |

Both sit in the same class as the two flips already accepted this session (BLAS threads 2.734e-10,
H_CZ `:j_parallel` 1.044e-10), i.e. the solver's own tolerance floor. `feastol` is **untouched** at
1e-12, so this changes when KNITRO is satisfied about *optimality*, never about *feasibility*.

`ek_inner.opt` is now 1e-10; the original is preserved as `ek_inner_opttol_1e-12_PRE_2026-08-11.opt`.

> **Scope limit, stated plainly.** `ek_inner.opt` is shared by **all five families and every
> driver**, but this is gated only on OZC-CROSS and CM+ZC-CROSS. `flexible_cm`, `common_frechet` and
> diagonal origin-ZC inherit the change **without evidence**. The mechanism is generic and
> documented, but the magnitude and the Δ agreement are measured for two families only. Gate the
> other three before a production campaign leans on this.

## 2. The `−102` discards were the same bug

Five inner solves per 22 returned `−102` (`KN_RC_FEAS_NO_IMPROVE`), ran to full length (median 39
Hessian calls, 128.8 s), and were then **discarded** — 646.5 s, **50% of all inner-solve wall**,
producing nothing. The inner accept list is `(0, −100, −101, −103)`, which omits `−102`, and
`classify_inner_result` (oracle.jl:322) short-circuits on that whitelist **before reading any
residual**:

```julia
s in (0, -100, -101, -103) || return ConfirmedNumericalNegative
```

so the KKT-residual / primal-dual-gap / `mean_m_resid` gates below it never execute for `−102`. That
looked inconsistent with `knitro_status.jl` (which classes `−102` as `:feasible_approx`,
acceptable = `true`) and with several test files that include `−102` in their feasible sets.

**No accept-list change was needed.** `−102` is specifically "desired dual-feasibility accuracy could
not be achieved" — which is precisely what demanding 1e-12 against a 1e-11 floor produces. At 1e-10
the `−102` class **disappears entirely** (12 solves return `status 0` instead). The rejected-status
problem and the tolerance problem were one bug, and fixing the tolerance dissolved it.

## 3. What does NOT work, and why (each ruled out by measurement)

### 3.1 Support screens on the restriction targets — 0 of 18

Two cheap certificates were built and tested (`test_restriction_support_screens_2026-08-11.jl`):

* **A (marginal)**: `ν_{o,k} ∈ [min_s z_{s,o}^k, max_s z_{s,o}^k]`. The LFD weights are a normalised
  reweighting of the draws, so each target must lie in the convex hull of that feature.
* **B (cross-moment)**: `ν_{o,k1}·ν_{p,k2} ∈ [min_s, max_s]` of the pairwise product — *not* implied
  by A, and exactly the constraint the cross grid adds over the diagonal family.

Both are genuine necessary conditions: a violation *proves* infeasibility, so false positives are
impossible by construction. Cost is negligible (A 12 µs, B 734 µs per point, one-time 5.8 s
precompute).

**Result: A fires on 0/18 failures, B on 0/18, union 0/18.** A positive control confirms both fire on
deliberate violations as small as 0.1%, so this is a real finding and not a broken screen. The reason
is decisive — the tightest margin is **identical** for failures and successes:

| | screen A margin | screen B margin |
|---|---|---|
| failed (n=18) | 0.0044 | 0.0011 |
| succeeded (n=4) | 0.0044 | 0.0011 |

*(as a fraction of interval width)*. Both classes hug the support boundary equally, so no threshold on
these conditions can separate them. Every failing point had all 60 marginal targets **and** all 1710
cross-moment products individually attainable — the system is infeasible **jointly**. No
per-coordinate screen can see that; detecting joint infeasibility is as hard as solving the problem.

**This also refutes a related proposal.** `meanzc_default_nu_bounds` sets the η box to
`(log(lo/4), log(hi*4))` — four times wider than the exact support interval on each side, which in
principle lets KNITRO propose provably-infeasible ν. Measurably it never does: not one proposed ν in
the sample fell outside the exact interval. Tightening the box is harmless hygiene, **not** a
performance win.

### 3.2 A dual-bank pre-check — nothing is knowable up front

Proposal: evaluate the stored dual at the new θ; if `f ≤ lower_limit`, then `f* ≤ f ≤ lower_limit`
and the solve is certain to hit the floor, so reject for the price of one FG evaluation (~0.12 s)
instead of a full solve.

**Measured: 0 of 22 solves were doomed at their first FG call.** Failures crossed the limit only
after 2–45 evaluations. A pre-check on the warm start would catch **none** of them.

Also note `select_warm_start_restricted` picks by *nearest-neighbour in scaled parameter distance* —
it never evaluates the objective — so this check would be an **added** cost, not a free by-product.

### 3.3 `ThresholdAbortState` — the same certificate that already exists

`Delta_dual = -ov.f` (operator_verification.jl:512, sign verified against `oracle.jl`'s
`constr[1] = -f*1e10`). Therefore `f ≤ lower_limit = -10` **is** a threshold abort at `Δ* ≥ 10`, and
it is checked at **every** FG evaluation inside the callback. `ThresholdAbortState` would be the same
certificate in the same place. The only difference available is a tighter threshold — which would
abort points with finite Δ\* slightly above δ, and those are *useful* evaluations (they give the
outer solver a real constraint value). So there is nothing better to be had.

**And the existing abort is already efficient**: when it fires, KNITRO does a median of 3 FG calls
and **2 Hessian calls, 5.8 s** — versus ~130 s for a full solve. It catches the point about as early
as it is knowable.

## 4. The cold-start bug

On a failed inner solve the code did:

```julia
if nStatus ∈ (0,-100,-101,-103); obj.x .= x     # success: keep it
else                             obj.x .= NaN   # failure: DESTROY the last good dual
```

`obj.x` is the warm-start slot (`inner_loop_initial_values = obj.use_cached_x && norm(obj.x) < 1e6 ?
obj.x : zeros(...)`, and `norm(NaN) < 1e6` is false), so a failure forces the **next** solve cold.
It does not merely refuse to store the failed iterate — it discards the last **successful** dual. At
the observed failure rate this meant **18 of 22 solves started at exactly `f = 0`**.

Fixed behind `INNER_KEEP_LAST_GOOD_X` (default `false`), which simply leaves `obj.x` alone on
failure. Safe by construction: `obj.x` is only ever *written* with an accepted solve's `x`, so no
garbage iterate can propagate; verification reads the returned `inner_x`, not `obj.x`; and
`select_warm_start_restricted` already guards with `all(isfinite, obj.x)`.

**Clean A/B, both arms at opttol 1e-10, only the flag differing:**

| | NaN on failure | keep last good dual |
|---|---|---|
| cold starts | 28/40 (70%) | **1/37 (3%)** |
| accepted median Hessian calls | 22 | **16** |
| accepted median wall | 60.6 s | **45.6 s (1.33×)** |
| usable evaluations | 13 | 11 |

The mechanism works exactly as diagnosed, but **the payoff is modest — 1.33×, not the order of
magnitude an earlier contaminated run suggested** (see §6). Throughput did not improve in this
sample (13 → 11 evaluations, read as noise at n≈12). Most wall-clock sits in *rejected* solves and
the outer search, not in shaving iterations off accepted ones.

**One caveat that argues for enabling it anyway.** Warm-start value depends strongly on how far apart
consecutive accepted points are. The per-solve spread for accepted solves at identical configuration:

```
normal search progress (n=11)   n_hess: 0 10 10 15 15 15 17 18 18 18 20
tightly-clustered points (n=4)  n_hess: 0  2  3  3
```

Nearly disjoint — a different regime, not a different draw. And the clustered regime is where long
searches actually end: CM+ZC-CROSS re-evaluated the *identical* point at evals 25, 26 and 28 while
stalling. So this fix is likely worth considerably more than 1.33× in the endgame and ~nothing early.
Combined with the correctness argument — discarding a known-good dual is simply wrong, and it costs
nothing — it is worth enabling. Quote 1.33×, not more.

## 5. The dual bank is off by default

`use_dual_bank::Bool = false` in both drivers, so no `RestrictedDualBank` is ever constructed.
Enabling it (at 1e-12): 22 → 24 solves, **4 → 6 accepted**, accepted median wall 132.8 → 113.1 s.

Memory `restricted-dual-bank-outer-ab-2026-08-01` records verdict *NO_MATERIAL_BENEFIT* — but from a
regime where failures were not constantly wiping the warm state, so it does not settle this case.
**Not flipped**: 4 vs 6 on ~20 solves is suggestive, not conclusive. Note it does not fix the
poisoning (cold starts stayed at 18 in absolute terms); it only supplies a fallback after the fact,
which is why §4 is the better fix.

## 6. Two process failures worth recording

**Data held only in memory.** The first support-screen study completed a 23-minute driver run and
then died in the analysis loop on a Julia soft-scope bug (`nA += 1` at top level creates a new local).
The recorded attempts existed only in RAM and went with the process. Both study scripts now
`serialize` before any analysis touches the data, and the tally runs inside a function.

**A shared config mutated mid-experiment.** `ek_inner.opt` was flipped to 1e-10 while the first
keep-last-good A/B was still running. `KN_load_param_file` is called *per solve*, so that run became
a mixture of both tolerances — visible in its status distribution (`−102` *and* `status 0` in the
same run). It produced a spurious "8.5 s median" that was quoted before being caught. The re-run put
both arms at 1e-10. **Do not edit shared config while experiments are in flight.**

## 7. What this means for the δ=1 bounds

**The κ values stand unchanged.** Every incumbent was feasible and passed `is_verified_success`, and
none of that depends on how long the solver ground before reporting. What changes is the reading of
*why* those runs were starved: "the searches, not the compute, were the binding constraint" now has a
concrete cause — both runs spent most of their budget pursuing an unreachable tolerance, and both
terminated on the `−102` stall that the tolerance produced.

A rerun at 1e-10 should get roughly **3× the evaluations per hour**. Combined with the finding that
single-start searches are not family frontiers (see the nesting write-up: the diagonal family's
multistart more than doubled its own single-start κ), the right next step is a **cross multistart
wave at the new tolerance**, not longer single runs.

## 8. Ranked next steps

1. **Gate the three untested families** on the `opttol` flip (§1) — it is already live for them.
2. **Cross multistart wave** at 1e-10, per the 8h write-up's recommendation.
3. **Decide `INNER_KEEP_LAST_GOOD_X`** (§4) — recommended on, worth 1.33× generally and more in the
   endgame.
4. **Re-test the dual bank** (§5) with a larger sample at 1e-10.
5. `H_EZ_fill` for OZC-CROSS — 21% of its outer solve, same segmented-reduction character
   `H_CZ_prep` had, and still unoptimised.
