# Melitz D20 negative-switch geometry audit (2026-07-30 continuation session)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), HEAD at session start
`2c5c30ab5db2e95675040736694bb968adf4e5e6` ("Melitz reduced-q `NumericalFailure` forensic
audit: root-cause and fix (2026-07-30)"), verified directly via `git log`/`git status` before
any edit (`git status` clean except pre-existing untracked scratch directories inherited from
other, unrelated sessions -- none touched this session). Governing prompt: a narrowly focused
forensic investigation of why a small negative movement along one real-D20 reduced-`q`
direction appears to move the inner problem abruptly from `Delta*~0.483` into the
infeasible/infinite region while positive movements remain finite despite many participation
switches, per the prior forensic session's own disclosed open question ("does a single
participation switch make the finite-QMC moment problem impossible? not established either way
by this session's evidence").

This session read in full, per the governing prompt's own required list:
`src/melitz/CLAUDE.md`, `docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`,
`docs/melitz_post_consolidation_validation_2026-07-28.md`,
`docs/melitz_reduced_q_subspace_search_2026-07-29.md`,
`docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md` (including its embedded
2026-07-30 addendum), `docs/melitz_reduced_q_numericalfailure_forensic_audit_2026-07-30.md`,
and the source behind every named entry point (`cc_bundle.jl`, `inner_screening.jl`,
`inner_session.jl`, `moment_operator.jl`, `log_cutoff_param.jl`, `origin_block_screen.jl`,
`reduced_q_subspace.jl`, `reduced_q_controller.jl`, `affine_cutoff.jl`).

**Disclosed scope decision on the referenced Dropbox archive**: `rclone lsf "dropbox:Gravity
robustness/Analysis/Server Output" --recursive` confirms
`melitz_reducedq_numericalfailure_forensic_audit_2026-07-30.zip` exists there, but every file
it names is already present live in this checkout's `docs/key_results/` (confirmed by name),
matching the immediately-prior session's own identical disclosed reasoning for not fetching
it. Not re-fetched.

## Executive summary

**All five core questions are answered, with runtime evidence, not source inspection alone:**

1. **The abrupt negative-side failure is genuine economic/convex-hull geometry, not a code
   error.** An exact, independently-verified linear-programming certificate (Phase 6) proves
   the finite-QMC moment system for one specific origin becomes infeasible exactly at, and
   immediately after, the first negative-direction participation switch.
2. **It is not specific to the reduced-`q` pathway.** The identical displaced economic state,
   evaluated through both the reduced-`q`/`:logcutoff` pathway and a production-equivalent
   `:logf` pathway, classifies **bit-identically** at every tested bracket point (Phase 3/4).
3. **The exact first negative switch** is at `t=8.4126176915e-03` along the audited dense
   direction: origin `o=14`, destination `d=19`, QMC row `s=13761` (`z=1.3038834644`),
   turning that origin-destination pair's participation **on**.
4. **The point does NOT remain verified finite immediately after it** -- it classifies
   `AboveEvaluationCap` under the default (screen-off) typed classifier, and is **exactly,
   independently LP-certified infeasible** (a true `InfiniteDeltaCertified` condition, Phase
   6) once the pre-existing-but-off-by-default `melitz_origin_block_screen` is applied.
5. **Concrete mechanism (Phase 7)**: origin 14 has 14 of its 20 bilateral-cutoff values
   clustered within roughly 1% of each other in productivity space. One specific interval in
   the compressed origin-block LP (destination `d=16`'s own required-tail-moment interval) is
   supported by **exactly one of the 80,000 QMC draws** at the anchor. All of origin 14's own
   cutoffs move continuously and jointly (they are components of one dense reconstructed
   `q_free` direction via the shared, linear gravity pivot) as the outer coordinate moves --
   even though only the `(o=14,d=19)` cell registers an exact crossing of a specific draw's
   productivity value at this particular `t`, the joint drift of the OTHER nearby cutoffs is
   enough to push that lone draw out of its interval (draw count 1 -> 0 across the bracket),
   destroying the feasibility of destination 16's own tight equality constraint. This is a
   genuine small-sample/clustering fragility of this specific origin's discretized cutoff
   structure, not a general `W`-inadequacy, and not a code defect (the matrix-free operator's
   own row-level bookkeeping was independently verified correct, Phase 5).

**One separate, genuine, now-fixed robustness bug was found (not the cause of the switch
geometry itself)**: `MelitzCCBundle`'s functor (`cc_bundle.jl`) could hand KNITRO a raw `NaN`
objective value on a genuinely non-finite intermediate evaluation (observed live: 5 of 14
raw callback evaluations during the exact D20 minus-side `h=0.5` capped solve that originally
produced the `4.9823e8` certificate) -- never recorded, never guarded, and (had a different
crossing sequence occurred) capable of corrupting a reported weak-duality certificate with
`NaN`/`Inf`. **Fixed**: a new `!isfinite(f)` guard, structurally parallel to the existing
`lower_limit` crossing check, ensures a non-finite raw evaluation always returns
`-KN_INFINITY` to KNITRO and is **never** recorded as `threshold_crossing_bound`.

**The `4.9823e8` number itself was NOT contaminated** -- Phase 1's direct instrumented trace
proves it is a genuine, non-overflowed, finite floating-point evaluation of the smooth CC dual
functor at a KNITRO-chosen (wildly non-optimal) dual point, valid as a weak-duality lower bound
by the same unconditional argument this codebase already relies on elsewhere -- but it is a
**crude, essentially arbitrary** bound (dependent on exactly where KNITRO's own diverging
pursuit happened to be evaluated last before giving up), not a tight characterization of
`Delta*`, and (per finding 4 above) understates the truth: the point is not merely "very large,"
it is exactly, certifiably infinite.

## Provenance

See `docs/key_results/melitz_negswitch_provenance_2026-07-30.txt` (generated alongside this
document) for the full machine-readable record: branch/commits, Julia/KNITRO versions, thread
counts, exact fixture/direction/seed, and every command run.

- Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh` pinned), `OPENBLAS_NUM_THREADS=1`/
  `OMP_NUM_THREADS=1` throughout. `-t 20` for every D20 diagnostic script (per
  `src/melitz/CLAUDE.md`), `-t 1` for the test suite.
- Anchor: `noah_D20` real data (`real_data/noah_D20`), focal country `fra` (index 2),
  `sigma=2.5`, `theta_star=:estimate`, seed=1, `W=80,000`, `target=0.5` (`theta0` loaded from
  `docs/key_results/melitz_qbw_phase3_theta_q_2026-07-29.csv`, row
  `("realD20_seed1_W80000", 0.5)` -- the exact Gate 3B/forensic-session fixture).
  `Delta0=0.4832764950468883` (matches every prior session's own documented value exactly).
- Direction/basis: `melitz_build_reduced_q_stage(theta0, x0, ctx, obj, 1;
  bandwidth_policy=PowerScaledQBandwidth(1e-3, 80_000, 0.5), target_switches=100)`, exactly the
  Gate 3B recipe -- `r_basis=|b_q|=0.000357`, crossings at `s=+-1`: `(+117,-100)`, matching the
  original doc's own reported values exactly.
- Policy: `CappedEvaluation(10.0)` throughout.

## Phase 0: exact state recovery and tri-backend reproduction

`scripts/melitz_negswitch_phase0_recover_anchor_2026-07-30.jl` (the exact script run).

Verified the anchor reproduces identically through all three routes the governing prompt
requires:

1. **Reduced-`q` state map** (`melitz_reduced_full_theta` at `s=0`): `max|theta_via_reduced -
   theta0| = 0.0` -- exact by construction, not merely close.
2. **Direct full-state evaluator** (`melitz_recover_lfd(obj, theta0)` on the `:logcutoff`
   bundle): `Delta0=0.4832764950468883`, `lfd_ok=true`, `nStatus=0`.
3. **Production `(A,f)` evaluator**: `theta0` -> `(A,f,gamma'_j)` via `melitz_expand_theta`,
   round-tripped into `:logf` via `melitz_reduce_theta` (relative discrepancy in the
   round-tripped `(A,f,gamma'_j)`: `0.0` / `9.99e-15` / `0.0`), then evaluated via a fresh
   `:logf`-parameterized bundle: `Delta0=0.4832764950468883`. **`max|Delta(reduced/logcutoff)
   - Delta(:logf production-equivalent)| = 0.0`** (bit-identical). The typed classifier
   (`solve_melitz_delta!`) at the anchor, through both the `mode=:delta` (`:logcutoff`) bundle
   and a freshly-built `mode=:implicit` bundle (`build_melitz_implicit_bundle`, production's
   own construction path), both return `FiniteSolved` for the anchor.

**All three backends agree.** No state-reconstruction fix was needed.

## Phase 1: exact provenance of the `4.9823e8` number

Instrumented **every** raw `MelitzCCBundle` functor objective evaluation
(`MELITZ_OBJECTIVE_TRACE`/`MELITZ_OBJECTIVE_TRACE_ENABLED`, new opt-in diagnostic,
`backend_config.jl`/`cc_bundle.jl`, disabled by default, zero cost when off) for the exact
`h=0.5` minus-side capped solve that originally produced this number
(`docs/melitz_reduced_q_numericalfailure_forensic_audit_2026-07-30.md` Phase 7).

**Raw callback trace, all 14 evaluations** (`docs/key_results/melitz_negswitch_phase1_objective_trace_2026-07-30.csv`):

| call | `f` | `f<=lower_limit(-10)`? |
|---:|---:|---|
| 1,2 | 0.0 | false (clean zero-dual start) |
| 3 | -7.8529e7 | **true** (first crossing) |
| 4 | -3.9264e7 | true |
| **5-9** | **NaN (all 5)** | false (NaN never satisfies `<=`) |
| 10,11 | -0.1601 | false |
| 12,13 | -0.3437 | false |
| 14 | **-4.9823190135746557e8** | true (LAST call) |

- **File/line**: `src/melitz/cc_bundle.jl`, the `MelitzCCBundle` functor,
  `f = sum(Q.arg1)/Q.M + zeta` (line ~293), certificate recorded at
  `Q.threshold_crossing_bound[] = -f` inside the `f<=Q.lower_limit` branch.
- **Formula**: `f(zeta,mu) = sum_s Psi(arg0_s)/M + zeta`, `arg0 = mul_G!(zeta,mu)` (the matrix-
  free CC dual functor, `Psi(x)=exp(x)-1` for `x<=1`, `Psi(x)=0.5*e*x^2-1` for `x>1`) -- an
  ordinary, un-transformed dual-objective evaluation, not a KNITRO terminal objective, dual-ray
  extrapolation, or transformed sentinel.
- **Not a sentinel/overflow artifact**: the reported value is exactly `-f` from call **14**,
  the literal LAST functor evaluation before KNITRO's own native `nStatus=-300` ("problem
  appears unbounded") exit -- `-f(last call) == certified_lower_bound`: **verified true**,
  both pre- and post-fix. `f=-4.9823190135746557e8` is a genuine, finite Float64 -- not
  `floatmax`, not `-floatmax`, not `Inf`. Mechanistically: at this call's dual iterate,
  `zeta approx -4.98e8` while the `mu`-block pushes almost every one of the 80,000 draws'
  `arg0` deeply NEGATIVE, so `Psi(arg0) approx exp(arg0)-1 approx -1` for nearly every draw
  (the smooth, UNDERFLOWING -- never overflowing -- branch of `Psi`), making
  `sum(Psi)/M approx -1`, and `f approx zeta - 1 approx zeta`. This is exactly why the reported
  value is safe from overflow in this specific direction of divergence: the dominant
  contribution is the (very large but ordinary, finite) `zeta` coordinate itself, computed via
  arithmetic that cannot overflow on this branch.
- **first callback satisfying `f<=-10`**: call 3 (`f=-7.85e7`).
- **value returned to KNITRO** at every crossing call: `-KNITRO.KN_INFINITY` (unconditional,
  mode-independent, the prior session's own 2026-07-30 fix).
- **`threshold_crossed` becomes true**: at call 3 first, then again at 4, then (crucially) at
  14 -- `threshold_crossing_bound` is **overwritten on every crossing call**, so the reported
  certificate is always the LAST one, not the first.
- **raw terminal KNITRO status**: `nStatus=-300` ("problem appears unbounded").
- **before or after termination**: the value is computed and recorded **during** the solve (at
  the last accepted callback invocation), then simply read back by the classifier **after**
  `KN_solve` returns -- there is no separate post-termination extrapolation step.
- **No `floatmax`/`-floatmax`/`Inf`/overflow value entered the REPORTED certificate's own
  calculation.**

**A genuinely new, disclosed finding**: **5 of the 14 raw callback evaluations (calls 5-9) were
literal `NaN`**, not merely large -- confirmed directly via the trace, not inferred. This is a
qualitatively different divergence direction than calls 3/4/14 (which cleanly saturate via
`Psi`'s bounded-below `exp()` branch): here, KNITRO's own line search evaluated a trial point
extreme enough that `mul_G!`'s internal arithmetic overflowed and two infinite terms of
opposite sign cancelled to `NaN` (`u[s] += const_o; u[s] -= z_power*cum[...]`, both terms
individually overflowing). Since `NaN <= Q.lower_limit` evaluates **false** in IEEE754/Julia,
these calls fell to the functor's `else` branch and were returned to KNITRO **raw, as literal
`NaN`, completely unrecorded** -- a genuine robustness gap, independent of (and not the cause
of) the reported `4.9823e8` value, since KNITRO's own subsequent behavior (backing off to the
small, well-behaved values at calls 10-13) happened, in this specific trajectory, not to depend
on what was returned during the `NaN` episode. Relying on that being true by chance rather than
by construction is exactly the gap Phase 9 closes (below).

## Phase 2/3: exact negative-side switch-threshold enumeration and bracketing

**Exact (not bisected) enumeration**, promoted to source
(`src/melitz/reduced_q_switch_geometry.jl`, `melitz_q_direction_exact_switches`), exploiting
that with `(g,A_free)` held fixed, `expand_free_theta_logcutoff`'s full `D x D` log-cutoff
matrix `q_full(t)` is an EXACT LINEAR function of `t` for every cell -- including the
analytically-reconstructed gravity-pivot cell (`build_q_gravity_offset`'s own `g0_q` depends
only on the fixed `q_jj`/`g`, and `pivot_expand` is linear). **Verified directly, not merely
assumed**: `max|slope(0.2->0.6) - slope(0.6->1.0)|` for `q_full(t)` at this real-D20 anchor is
`3.19e-15` (both signs) -- floating-point noise, confirming exact affinity over the whole
tested range.

**First 10 exact minus-direction switch thresholds**
(`docs/key_results/melitz_negswitch_phase2_minus_switches_2026-07-30.csv`):

| step | `t` | origin `o` | dest `d` | sorted pos `k` | QMC row `s` | on/off | `z` |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 8.4126176915e-03 | 14 | 19 | 72152 | 13761 | on | 1.303883 |
| 2 | 1.0530457182e-02 | 14 | 16 | 72152 | 13761 | off | 1.303883 |
| 3 | 1.4665291629e-02 | 2 | 14 | 69630 | 30241 | on | 1.262945 |
| 4 | 3.1171519313e-02 | 18 | 7 | 71437 | 65179 | off | 1.290791 |
| 5 | 5.9225422667e-02 | 14 | 16 | 72153 | 26324 | off | 1.303897 |
| 6 | 6.6223067109e-02 | 15 | 16 | 72828 | 7614 | on | 1.317284 |
| 7 | 6.7099698874e-02 | 12 | 13 | 73455 | 19461 | off | 1.331022 |
| 8 | 6.9433533783e-02 | 8 | 18 | 69957 | 77556 | on | 1.267616 |
| 9 | 8.0756407099e-02 | 14 | 19 | 72151 | 65491 | on | 1.303864 |
| 10 | 8.1972684097e-02 | 5 | 8 | 72188 | 9252 | on | 1.304498 |

**Cross-check against the existing (independently implemented) two-sided crossing-count
infrastructure** (`melitz_q_direction_two_sided_crossings`, pre-existing source): total
crossing count just below/above each threshold matches `k-1`/`k` **exactly** at 9 of the first
10 thresholds (`k=1..6,8,9,10`); one single-count discrepancy at `k=7` (crossing count reads
one higher than expected on both sides of that threshold) that this session did **not** root-
cause -- disclosed as an open, minor residual, not affecting the first-5-switch analysis this
document's own findings rest on (all of which passed the cross-check exactly).

**No such analysis previously existed for the exact first switch** -- the prior forensic
session's own Phase 7 replayed only the `h=0.5` endpoint (~52 switches away, per Gate 3B's own
crossing counts), never the exact first one.

### Bracketing (Phase 3) and cross-backend replay (Phase 4)

For each of the first 5 distinct minus-direction thresholds, evaluated `below`/`at`/`above`
(`eps` = 15% of the smaller neighboring gap) through **both** the reduced-`q`/`:logcutoff`
pathway and a production-equivalent `:logf` pathway (identical displaced economic state,
converted via `melitz_expand_theta` -> `MelitzPrimitives` -> `melitz_reduce_theta`), fresh
`MelitzInnerSession` each time (`docs/key_results/melitz_negswitch_phase34_minus_bracket_2026-07-30.csv`):

| k | label | `t` | reduced-`q` result | production-equivalent result | match |
|---:|---|---:|---|---|---|
| 1 | below | 8.0949e-3 | `FiniteSolved`, `Delta=0.483277` | `FiniteSolved`, `Delta=0.483277` | **yes** |
| 1 | at | 8.4126e-3 | `AboveEvaluationCap`, `cert=367267` | `AboveEvaluationCap`, `cert=367267` | **yes** |
| 1 | above | 8.7303e-3 | `AboveEvaluationCap`, `cert=107281` | `AboveEvaluationCap`, `cert=107281` | **yes** |
| 2 | below/at/above | ... | `AboveEvaluationCap` (80210/618310/2.9168e6) | identical | **yes** |
| 3 | below/at/above | ... | `AboveEvaluationCap` (212612/466768/292834) | identical | **yes** |
| 4 | below/at/above | ... | `AboveEvaluationCap` (258141/568316/177212) | identical | **yes** |
| 5 | below/at/above | ... | `AboveEvaluationCap` (227384/3.07781e6/170474) | identical | **yes** |

**The transition from `FiniteSolved` to `AboveEvaluationCap` happens EXACTLY at the first
switch (`k=1`), and nowhere else in the first five** -- `k=2..5` are all already
`AboveEvaluationCap` on both sides of their own threshold (the point never "re-enters"
`FiniteSolved`). **Every one of the 15 bracket points produces a bit-identical classification
and, where applicable, a bit-identical `certified_lower_bound`, between the reduced-`q` and
production-equivalent pathways.** This directly answers core questions 1 and 2: the phenomenon
is real geometry (not a code artifact) and is **not** reduced-`q`-specific.

## Phase 5: independent affected-row verification

For the exact switching row (`o=14`, `d=19`, QMC row `s=13761`), computed activity
INDEPENDENTLY via the raw `melitz_firm` primitive (no shared low-level moment-operator helper)
and compared against the production matrix-free operator's own `op.bin[row,o] >= op.rank[d,o]`
lookup, both just below and just above the switch:

| side | raw `melitz_firm` active | `op.bin>=rank` active | match |
|---|---|---|---|
| below (`t=7.15e-3`) | false | false | **yes** (`bin=9,rank=10`) |
| above (`t=8.73e-3`) | true | true | **yes** (`bin=10,rank=10`) |

**Exact agreement.** The matrix-free operator's row/cell indexing is correct at the switching
row -- the participation switch is a genuine economic event, not an indexing bug. (This
diagnostic ran with ordinary `@inbounds`-enabled production code; the pre-existing, disclosed,
unrelated `mul_G!` segfault documented by the prior two sessions -- see "Test suite" below --
occurs only inside Gate 2's own "Method A" KNITRO-driven code path, never reached by this
targeted row check.)

## Phase 6: direct finite-support feasibility test

Used the **pre-existing** exact origin-block feasibility infrastructure
(`src/melitz/origin_block_screen.jl`: `melitz_origin_block_lp`, a compressed `O(D)`-variable
LP; `melitz_origin_block_lp_reference`, an UNCOMPRESSED `O(W)`-variable LP over the raw draws,
built as an independent validation reference; `melitz_origin_block_monotonicity_check`, a cheap
necessary-only condition) directly at the first-switch bracket, for the affected origin
(`o=14`) (`docs/key_results/melitz_negswitch_phase6_feasibility_2026-07-30.csv`):

| side | `t` | compressed LP | full-`W` reference LP | monotonicity | full 20-origin screen |
|---|---:|---|---|---|---|
| below | 7.1507e-3 | **feasible** | **feasible** | ok | PASS (no infeasibility anywhere) |
| above | 8.7303e-3 | **INFEASIBLE** | **INFEASIBLE** | ok (necessary-only, doesn't catch this) | **INFEASIBLE (origin 14)** |

**The compressed LP and the independent full-`W`-row reference LP agree exactly** (both
feasible below, both infeasible above) -- confirming the compressed reformulation is exact, not
merely a heuristic proxy, at this point. **This is a genuine, LP-certified, EXACT infeasibility
of origin 14's own finite-QMC trade-share moment system, occurring exactly at the first
negative participation switch.**

**Direct answer to the central question this whole investigation exists to resolve**: *does
the first minus-side participation switch make the finite-QMC moment problem impossible?*
**Yes -- for origin 14's own moment block, proven by an exact LP certificate, independently
confirmed by an uncompressed full-draw reference LP.** This is a materially stronger and more
precise answer than the prior forensic session's own "not established either way" -- that
session replayed only the `h=0.5` endpoint (~52 switches downstream) and never ran the
origin-block LP at all; this session ran it exactly at the first switch and got a clean,
independently-cross-checked yes.

**Note on classification vs. certification**: the DEFAULT typed classifier
(`solve_melitz_delta!`, `origin_block_screen=false` by default, unchanged this session, per the
2026-07-28 consolidation's own disclosed decision not to re-benchmark it) reports this point as
`AboveEvaluationCap` (a valid but crude weak-duality bound), **not** `InfiniteDeltaCertified` --
even though the stronger, exact certificate is available and, when explicitly invoked
(`melitz_origin_block_screen`), correctly returns `InfiniteDeltaCertified(o=14, ...,
:origin_block)`. See Phase 9 for the disclosed, deliberately-not-unilaterally-changed
implication.

## Phase 7: concrete convex-hull geometry diagnosis

**Not "high-dimensional geometry" -- one specific, identifiable interval.** Directly inspecting
the compressed LP's own inputs (`docs/key_results/melitz_negswitch_phase7_geometry.log`,
archived alongside this document) at the bracket around the first switch:

- Origin 14's own 20 bilateral-cutoff breakpoints span `[1.097, 1.350]`, but **14 of the 20**
  are packed into the narrow sub-band `[1.297, 1.309]` -- roughly a 1% relative spread.
- **Destination `d=16`'s own required tail-moment interval (rank position 9 in the sorted
  breakpoint list) is supported by exactly ONE of the 80,000 QMC draws at the pre-switch
  bracket** (`ndraw=1`, `ymin=ymax=1.48887` -- a single draw, `ymin==ymax` because there is
  only one). **After the bracket, that same interval has ZERO draws**
  (`ndraw=0, ymin=Inf, ymax=-Inf`).
- **The mechanism**: with `(g,A_free)` fixed, ALL of origin 14's own bilateral cutoffs are
  components of the SAME dense reconstructed `q_free` vector (via the shared, linear gravity
  pivot) -- so as the outer coordinate `t` increases, EVERY one of origin 14's cutoffs drifts
  continuously and simultaneously, not just the one (`o=14,d=19`) cell whose cutoff happens to
  cross an exact QMC draw value at this particular `t` (this session's own binary "switch"
  definition, Phase 2). The joint drift of the OTHER nearby cutoffs (in particular the pair of
  breakpoints bracketing destination 16's own thin interval) is enough, over this small
  bracket, to push the single occupant draw of that interval out of it entirely. With that
  interval's only supporting mass gone, destination 16's own tight equality constraint
  (`H[16]=0.17618690`, an empirical trade-share target that must be matched EXACTLY by some
  nonnegative probability distribution over the 80,000 draws) can no longer be satisfied by
  ANY feasible allocation -- not merely "hard to hit," but structurally impossible, exactly
  the compressed LP's own verdict.
- **Which of the task's own candidate explanations fits**: **a single, uniquely essential
  draw** (the lone occupant of interval 9 -- with it gone, no feasible representation exists
  for destination 16's equation) **whose loss is CAUSED by gravity-pivot-induced, joint
  multi-cell movement** (several origin-14 cutoffs shift together, not just the literal
  "switching" cell) -- both categories jointly, not a single simple story.
- **Base LFD weight on the switched row**: not separately re-derived this session (the switched
  ROW, `s=13761`, is a distinct draw from the interval-9 draw that actually disappears; a full
  weight decomposition of the origin's own optimal LFD both sides of the switch was not run,
  a disclosed scope reduction -- the LP feasibility verdict itself, independently
  cross-validated by the uncompressed reference LP, is the decisive evidence this phase
  required).
- **The focal free-entry link plays no role**: origin 14 is not the focal country (`focal=2`,
  `"fra"`); `melitz_origin_block_lp`'s own focal-link row is only added `if o==target_country`,
  structurally inapplicable here (Core question 10, answered: **no**).

## Phase 8: plus-vs-minus asymmetry

Repeated the identical bracketing protocol for the first 5 exact POSITIVE-direction switches
(`docs/key_results/melitz_negswitch_phase8_plus_bracket_2026-07-30.csv`):

| k | `t` (first switch) | origin/dest | dir | classification (below/above) | `Delta` |
|---:|---:|---|---|---|---|
| 1 | 1.4773e-02 | 18/12 | on | `FiniteSolved`/`FiniteSolved` | 0.483276 -> 0.483250 |
| 2 | 2.2966e-02 | 10/13 | on | `FiniteSolved`/`FiniteSolved` | 0.483250 -> 0.483249 |
| 3 | 3.2777e-02 | 8/16 | on | `FiniteSolved`/`FiniteSolved` | 0.483249 -> 0.483248 |
| 4 | 4.3267e-02 | 14/19 | off | `FiniteSolved`/`FiniteSolved` | 0.483248 -> 0.483236 |
| 5 | 5.2892e-02 | 4/10 | off | `FiniteSolved`/`FiniteSolved` | 0.483236 -> 0.483230 |

**Every one of the first 5 positive-direction switches remains `FiniteSolved`**, `Delta`
drifting only in the fifth decimal place -- no infeasibility of any kind, consistent with the
Gate 3B table's own documented plus-side robustness (verified finite through 1-50 switches).

**Why the asymmetry**: the mechanism identified in Phase 7 is inherently DIRECTIONAL --moving
`t` in the plus direction moves the SAME cluster of tightly-packed origin-14 cutoffs the
OPPOSITE way. For this specific dense direction and anchor, that opposite movement does not
happen to drain any single-draw interval to zero within the tested range (the fragile
interval-9 draw identified on the minus side is a MINUS-side-specific casualty; the plus
direction's own first few switches involve different origin/destination cells entirely --
`o=18,d=12`; `o=10,d=13`; `o=8,d=16`; `o=14,d=19` again, but this time turning OFF, at
`k=4` -- none of which happens to coincide with draining a razor-thin interval this session
identified as empty). **This is genuine geometry, not a code bug**: the SAME dense direction,
run in two opposite signs, is not expected to encounter the same fragile intervals
symmetrically, since the underlying cutoff clustering is itself asymmetric around the anchor.
A full symmetric geometric dump for the plus side (mirroring Phase 7's breakpoint/interval
table) was not independently re-run this session -- a disclosed scope reduction; the
observed classification outcome (uniformly `FiniteSolved`) is the evidence this phase rests
on, not an assumed symmetry argument.

## Phase 9: repair

**Fixed** (`src/melitz/cc_bundle.jl`, `src/melitz/backend_config.jl`): the NaN-objective
robustness gap found in Phase 1. `MelitzCCBundle`'s functor now checks `!isfinite(f)` BEFORE
the existing `f<=Q.lower_limit` check and, if true, returns `-KNITRO.KN_INFINITY` (the same
"back off" sentinel already used for a genuine crossing) **without** ever touching
`threshold_crossed`/`threshold_crossing_bound`/`threshold_crossing_x` -- a non-finite raw
evaluation can now never become a reported weak-duality certificate. Verified safe: re-running
the exact instrumented `h=0.5` minus-side trace post-fix reproduces the **identical** 14-call
trajectory and the **identical** `certified_lower_bound=4.9823190135746557e8` (this specific
trajectory was insensitive to what was returned during the `NaN` episode; the fix closes a
latent risk, not a live misclassification in this particular case).

**No new exact screen was added** -- `melitz_origin_block_screen` (the tool that correctly
upgrades this exact family of points from `AboveEvaluationCap` to `InfiniteDeltaCertified`,
Phase 6) already existed, pre-dating this session, and is directly demonstrated (this session,
live) to work correctly on the exact point this investigation is about. **This session
deliberately does NOT flip `origin_block_screen`'s default to `true`** for reduced-`q`
production evaluations -- that is a broader production-behavior/performance-tuning decision
(one `D`-origin LP solve per trial-point evaluation) explicitly out of this session's bounded
forensic scope (matching the 2026-07-28 consolidation's own disclosed reasoning for leaving
this same default alone: "this session did not re-benchmark it"). **Recommendation for a
future session**: benchmark `origin_block_screen=true` as the reduced-`q` default (or as an
early, cheap pre-screen ahead of every KNITRO trial-point evaluation, mirroring the existing
Phase 7 cap-screen pattern in `reduced_q_subspace.jl`) -- the evidence in this document shows
it correctly and cheaply distinguishes "certifiably infinite" from "merely enormous" exactly
where the current default cannot.

**Genuine infeasibility is established** for origin 14 immediately after the first negative
switch (Phase 6/7) -- this is not itself upgraded to a live production default this session,
per the above.

Promoted the exact switch-threshold enumeration to source
(`src/melitz/reduced_q_switch_geometry.jl`, `melitz_q_direction_exact_switches`, wired into
`include_melitz.jl`) -- Rule 10 (no second, script-only implementation): every diagnostic
script in this session's own package, and the new regression tests below, call this one
function.

## Required tests

New testset `"D20 negative-switch geometry audit (2026-07-30)"`
(`test/melitz/runtests.jl`), D4 (`FIXTURE`, the repo's own standard fast test fixture) --
real-D20-scale findings are validated by the one-off scripts above (this repo's own
established D20/test-suite split), not replayed at D20 cost inside the fast suite:

1. **Item 2**: exact first-switch locations (both signs) match the pre-existing
   `melitz_q_direction_two_sided_crossings` crossing-count infrastructure exactly at every
   tested threshold, for a controlled D4 fixture.
2. **Item 1**: reduced-`q` state map, `:logf`-converted production-equivalent state, and the
   direct full-state evaluator agree (`(A,f,gamma')` bit/near-bit-identical; `Delta` agrees to
   `rtol=1e-6`) at `s=0` and at a displaced point.
3. **Item 7**: an identical displaced state, bracketed around the first D4 switch, classifies
   identically (`nameof(typeof(r))` matches) through the reduced-`q` and production-equivalent
   `:logf` wrappers, both below and above the threshold.
4. **Item 8**: the compressed origin-block LP and the independent full-`W` reference LP agree
   exactly (feasible/infeasible) at the first D4 switch's own bracket.
5. **Item 5/9**: `AboveEvaluationCap.certified_lower_bound` is `isfinite` (never `NaN`/`Inf`)
   across a batch of 20 bracket points (both signs, first 5 switches each); zero
   `NumericalFailure` across the same batch (Item 10).
6. **Item 5 (functor-level)**: directly exercises the new `!isfinite(f)` guard by forcing a
   `NaN` raw objective via a `NaN` dual coordinate -- confirms the functor returns
   `-KNITRO.KN_INFINITY` (never `NaN`) and leaves `threshold_crossed` untouched.
7. **Item 11**: `MELITZ_DENSE_G_MATERIALIZATIONS[]` is unchanged across the exact
   switch-geometry enumeration and an origin-block LP call -- no dense `G` in this session's
   own new code paths.

**52/52 assertions passed** in an isolated standalone run
(`docs/key_results/melitz_negswitch_standalone_test_2026-07-30.log`). See "Test suite" below
for the full-suite run's own (pre-existing, unrelated) limitation.

## Test suite

`julia --project=. -t 1 test/melitz/runtests.jl`: **reproduced the SAME pre-existing,
documented `SIGSEGV`** the immediately-prior two sessions both independently hit and disclosed
(`mul_G!`, `moment_operator.jl:281`, inside the pre-existing "Gate 2/3 matched-effort +
threaded direction infrastructure (2026-07-29 validation session)" testset's own "Method A"
KNITRO-driven code path -- exact same file/line, same trigger testset, same
`SIGSEGV`/`getindex`/`@inbounds`-block signature the 2026-07-30 forensic session's own doc
already root-cause-narrowed and left open). **Confirmed unrelated to this session's own
changes**: the crash occurs BEFORE this session's own new testset is ever reached (appended at
the very end of the file, after the crashing testset), and neither of this session's two
modified files (`cc_bundle.jl`'s new `isfinite` guard, `backend_config.jl`'s new trace globals)
is anywhere near `mul_G!`/`moment_operator.jl`. Verified, not merely asserted: this session's
own new testset, run in complete process isolation, passes 52/52; the immediately-adjacent
pre-existing testsets most directly touching the code this session modified ("Section 11
(architecture consolidation): static scan for unsafe defaults/direct low-level calls" and
"`NumericalFailure`/`lower_limit` forensic audit (2026-07-30): `threshold_crossed` is
mode-independent", lines 6579-7386), extracted and run together in an isolated process, also
pass cleanly (see `docs/key_results/melitz_negswitch_adjacent_testsets_2026-07-30.log`).
A genuinely clean single monolithic 7500-line run could not be obtained this session, matching
(not regressing from) both immediately-prior sessions' own identical, already-disclosed
limitation for the exact same pre-existing testset.

**Disclosed extraction artifact**: the isolated adjacent-testsets run's own log
(`melitz_negswitch_adjacent_testsets_2026-07-30.log`) shows "Section 11" passing 6/6 cleanly,
and the following testset ("Governing prompt 2026-07-29 (A_q separation...)") passing 46/47
with one `UndefVarError: KNITRO_AVAILABLE not defined` -- this is an artifact of this session's
own naive line-range extraction (that testset references a top-level `const KNITRO_AVAILABLE`
defined much earlier in the full file, at line 1371, outside the copied range), not a real
assertion failure or a consequence of this session's own source changes; every assertion that
DID run in that testset passed.

## Files changed

```
Modified:
  src/melitz/backend_config.jl   (new opt-in objective trace globals)
  src/melitz/cc_bundle.jl        (Phase 1's NaN trace hook; Phase 9's isfinite guard)
  src/melitz/include_melitz.jl   (new include line)
  test/melitz/runtests.jl        (new testset, "D20 negative-switch geometry audit (2026-07-30)")

New:
  src/melitz/reduced_q_switch_geometry.jl   (Phase 2/3, promoted exact switch enumeration)
  docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md   (this document)
  docs/key_results/melitz_negswitch_*_2026-07-30.{csv,log,txt,jl}
```

**Zero diff in `cc_algo/`** or any other Ricardian path (confirmed via `git status`).

## Required output files

- `docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md` -- this document.
- `docs/key_results/melitz_negswitch_phase1_objective_trace_2026-07-30.csv` -- Phase 1, all 14
  raw callback values.
- `docs/key_results/melitz_negswitch_phase2_minus_switches_2026-07-30.csv` /
  `..._phase2_plus_switches_2026-07-30.csv` -- Phase 2, first 15 exact switch thresholds each
  sign.
- `docs/key_results/melitz_negswitch_phase34_minus_bracket_2026-07-30.csv` -- Phase 3/4,
  cross-backend classification at every bracket point.
- `docs/key_results/melitz_negswitch_phase6_feasibility_2026-07-30.csv` -- Phase 6, LP
  feasibility verdicts.
- `docs/key_results/melitz_negswitch_phase7_geometry_2026-07-30.log` -- Phase 7, full
  breakpoint/interval/`H`-target dump, both sides of the first switch.
- `docs/key_results/melitz_negswitch_phase8_plus_bracket_2026-07-30.csv` -- Phase 8, plus-side
  comparison.
- `docs/key_results/melitz_negswitch_standalone_test_2026-07-30.log` -- new regression testset,
  isolated run (52/52 passed).
- `docs/key_results/melitz_negswitch_adjacent_testsets_2026-07-30.log` -- adjacent pre-existing
  testsets, isolated run.
- `docs/key_results/melitz_negswitch_provenance_2026-07-30.txt` -- full provenance record.
- Scripts (all reproduce the exact numbers in this document, `julia --project=. -t 20
  <script>.jl` after `source .knitro_env.sh; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`):
  - `scripts/melitz_negswitch_phase0_recover_anchor_2026-07-30.jl`
  - `scripts/melitz_negswitch_phase1_objective_trace_2026-07-30.jl`
  - `scripts/melitz_negswitch_phase2_switch_thresholds_2026-07-30.jl`
  - `scripts/melitz_negswitch_phase34568_bracket_checks_2026-07-30.jl`
  - `scripts/melitz_negswitch_phase7_geometry_2026-07-30.jl`
- Source: `src/melitz/reduced_q_switch_geometry.jl` (new); `src/melitz/backend_config.jl`,
  `src/melitz/cc_bundle.jl`, `src/melitz/include_melitz.jl`, `test/melitz/runtests.jl`
  (additive edits).

## Final report answers

1. **Is the negative-switch anomaly specific to reduced `q`?** No -- bit-identical
   classification and `certified_lower_bound` through both the reduced-`q` and
   production-equivalent `:logf` pathways at every one of 15 tested bracket points (Phase 4).
2. **Do production and reduced wrappers evaluate the same displaced state identically?** Yes,
   exactly (Phase 0, Phase 4).
3. **What is the exact first negative participation switch?** `t=8.4126176915e-03`, cell
   `(o=14, d=19)`, QMC row `s=13761` (`z=1.3038834644`), turning on.
4. **Does the point remain verified finite immediately after it?** No --
   `AboveEvaluationCap` under the default classifier.
5. **If not, is it `AboveEvaluationCap` or `InfiniteDeltaCertified`?** `AboveEvaluationCap`
   under the DEFAULT (screen-off) classifier; genuinely, exactly `InfiniteDeltaCertified`
   (origin 14) once `melitz_origin_block_screen` is applied -- both true simultaneously,
   depending on which tool is asked.
6. **What independent certificate supports that result?** An exact origin-block LP
   (compressed AND an independent full-`W`-row reference LP, both agreeing), Phase 6.
7. **How can one or a few switches matter with `W=80,000`?** Origin 14's own bilateral
   cutoffs are unusually tightly clustered (14 of 20 within ~1% of each other), creating a
   razor-thin interval supported by exactly 1 of 80,000 draws -- a small joint movement of
   several such cutoffs (not literally the switching cell alone) drains it, Phase 7.
8. **Are the switched rows uniquely essential?** The row that actually matters (the lone
   occupant of interval 9) is uniquely essential for destination 16's own tight equality --
   yes.
9. **Does the gravity pivot cause additional hidden switches?** Not additional discrete
   switches, but additional CONTINUOUS co-movement of several cutoffs simultaneously (they
   share one linear reconstruction) -- yes, and this joint drift is the proximate mechanism.
10. **Is the focal free-entry link responsible?** No -- origin 14 is not the focal country.
11. **Does the matrix-free operator agree with an independent reference calculation?** Yes,
    exactly, at the switching row (Phase 5).
12. **Was the reported `4.9823e8` number valid?** Yes, as a genuine (non-overflowed, finite)
    weak-duality lower bound -- but a crude, non-tight one; the true answer for this point is
    exact infinity (Phase 6).
13. **Exactly how was that number generated?** The last of 14 raw CC-dual functor
    evaluations before KNITRO's own `nStatus=-300` exit, `-f` at that call, Phase 1.
14. **Why is the positive direction finite with dozens of switches?** The same clustering
    mechanism, run in the opposite direction, does not happen to drain any single-draw
    interval within the tested range -- genuine directional asymmetry in the underlying
    geometry, Phase 8.
15. **Is the asymmetry genuine geometry or a code bug?** Genuine geometry, independently
    confirmed via exact LP certificates and independent row-level verification -- not a code
    bug in the switch geometry itself. (A separate, unrelated raw-`NaN`-to-KNITRO robustness
    gap WAS a real bug, found and fixed, but did not cause or explain the switch geometry.)
16. **What was fixed or what new exact screen was added?** Fixed: the `!isfinite(f)` functor
    guard (Phase 9). No new exact screen was built -- the pre-existing
    `melitz_origin_block_screen` is demonstrated to already be exactly the right tool,
    recommended (not unilaterally defaulted-on) for a future session's benchmarked rollout.
