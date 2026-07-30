# Melitz joint (A,q) LFD-preserving feasibility-preserving search (2026-07-30 continuation session)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), HEAD at session start `412a407`
("Melitz negswitch audit follow-up: correct NaN mechanism, identify origin 14 (Korea)"),
building directly on `docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md` (the
"negswitch audit", commits `5038401`/`412a407`). Governing prompt:
`melitz_joint_Aq_feasibility_preserving_search_2026-07-30` -- a bounded architecture/geometry
experiment testing whether joint `(A,q)` movements that preserve the anchor's own
least-favorable distribution (LFD) `p*` can cross the negswitch audit's own exactly-identified
cutoff cliff (origin 14/Korea, destinations 16/19), where a pure-`q` movement was shown to be
LP-certified infeasible immediately past the first negative-direction participation switch.

**Not a broad frontier or D20 production campaign** -- per the governing prompt's own explicit
scope limit, this is a bounded architecture/geometry experiment: a handful of switch brackets,
a `<=5`-step welfare smoke test, D4-scale coordinate prototyping.

## Executive summary

**All central questions are answered with live D20 runtime evidence, not source inspection
alone.**

1. **A can be recovered analytically, exactly, cell-by-cell** so that a FIXED probability
   vector `p*` continues to satisfy every one of the `D^2` factual bilateral trade-share
   moments after `q` changes -- confirmed against the actual `melitz_C`/`melitz_firm`/
   `mul_Gt!` code (Phase 0), not assumed from the schematic prompt formula, and verified to
   machine precision at D4 (15/15 new regression assertions) and to solver tolerance
   (`~1e-15`) at real D20.
2. **The exact cellwise-recovered `A` generally violates A-gravity** (confirmed:
   `~1e-5` to `~2e-5` residual at a small D4 perturbation, `~1e-6` at real D20 switch
   brackets) -- exactly as the governing prompt anticipated. A bounded discrete corrector
   (Step 1C, `melitz_lfd_corrector`) restores A-gravity and the focal free-entry link
   simultaneously **at D4 to `~1e-7`** and **at real D20 partially** (residual reduced by
   ~1-2 orders of magnitude but not always inside a strict `1e-6` tolerance within the
   corrector's bounded 8-round/80-candidate-per-round search budget -- see Phase 2/6 detail
   and the "Corrector convergence" finding below).
3. **The known Korean cutoff cliff (`o=14,d=19`, `t=0.0084126177`) IS crossed by the
   compensated joint `(A,q)` move.** At the EXACT audited bracket, the pure-`q` endpoint
   reproduces the audit's own finding (`AboveEvaluationCap`, `cert=107281`); the
   LFD-preserving compensated endpoint (Step 1B alone, before any correction) is witnessed
   feasible (`max|trade residual|~8e-15`, `focal~8e-8`, `gravity_A~5e-8`) and a full
   independent reoptimization of the resulting state returns **`FiniteSolved`,
   `Delta*=0.483264`** -- essentially unchanged from the anchor's `Delta0=0.483276`. This
   result repeats at every one of the first five negative-direction switches (Phase 2/4).
4. **The exact compensated path is a genuine step function, not continuous** -- proven both
   analytically (from the moment formula itself) and directly verified numerically: the
   `p*`-weighted active-tail statistic `T_od(p*,q_od)` is EXACTLY constant between two
   adjacent draws with `p*_s>0` and jumps EXACTLY at a positive-weight draw crossing, by
   EXACTLY `p*_s * z_s^(sigma-1)` (Phase 0 Test 3, D4; matches to `rtol=1e-9`). A does NOT
   move at all within a chamber and DOES jump discretely at a switch. An ordinary smooth
   KNITRO scalar coordinate cannot represent this path faithfully; a first-order
   Jacobian/Newton corrector is provably degenerate (zero local slope) almost everywhere.
5. **The bounded discrete corrector (a genuinely different design from a smooth SQP,
   justified by finding 4) is the right tool but is not yet production-grade**: it succeeds
   quickly (`<0.3s`) and to tight tolerance at the first two D20 switches, and partially
   (meaningful but incomplete residual reduction, `60-90s`, hitting its round budget) at
   switches 3-5. In every tested case (even where the STRICT witness tolerance is not met),
   the fully reoptimized point is `FiniteSolved` at essentially anchor-level `Delta*`.

6. **Phase 6 (mandatory bounded welfare-continuation smoke test) is a genuine, disclosed
   negative result**: zero steps accepted in either welfare direction, at step sizes down to
   `0.0125%` gains-from-trade -- the SAME corrector design that works well for local
   nuisance-direction cliff-crossing (finding 3) does not have enough "reach" to absorb a
   genuine welfare-level shift within its current bounded search window.

**Conclusion: B** ("compensation works, but switch transitions are discontinuous") -- see
"Decision" below. Conclusion A's narrower claim (a joint `(A,q)` move crosses the audited
cliff, verified feasible, `Delta* <= divergence(p*)`) is strongly supported as B's own
load-bearing positive evidence, but A's full welfare-continuation criterion is not met.


## Provenance

See `docs/key_results/melitz_aq_provenance_2026-07-30.txt` for the full machine-readable
record.

## Phase 0: exact implemented moment map (derived from source, verified against code)

`melitz_firm`/`melitz_C` (`firm_quantities.jl`): `price_od(z) = markup*w_o*tau_od/(A_od*z)`,
`rev_od(z) = C_od*z^(sigma-1)`, `C_od = expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma)`,
active iff `rev_od(z)/sigma - w_o*f_od > 0` iff `z > exp(q_od)` (STRICT).

`melitz_moments!` (`moments.jl`): `G[w,trade_col(o,d)] = C_od*z[w,o]^(sigma-1)*active/expenditure_d
- lambda_od`, `lambda_od = X_data[o,d]/expenditure_d`. Writing `coef_od = C_od/expenditure_d` and
`T_od(p,q_od) = sum_w p_w*z[w,o]^(sigma-1)*1{z[w,o]>exp(q_od)}` (confirmed identical to
`mul_Gt!`'s own `coef[o,d]*tail_sum_d` term, `moment_operator.jl:314-362`, when `sum(p)==1`):

```
E_p[trade moment od] = coef_od*T_od(p,q_od) - lambda_od
```

`coef_od = (markup*w_o*tau_od/A_od)^(1-sigma) = K_od * A_od^(sigma-1)`, `K_od =
(markup*w_o*tau_od)^(1-sigma)` INDEPENDENT of `A_od`. So `E_p[trade moment od]==0` iff
`T_od(p,q_od) == lambda_od/coef_od =: H_od` -- EXACTLY `origin_block_screen.jl`'s own `H[d]`
target (`melitz_origin_block_lp` line ~124-130, re-derived here independently, not assumed;
confirmed algebraically identical). Given a FIXED `p`, two states (anchor, new) both satisfying
`E_p[trade moment od]==0` must have `coef_od(new)*T_od(new) == coef_od(anchor)*T_od(anchor)`:

```
a_od(new) = a_od(anchor) + [log(T_od(anchor;p)) - log(T_od(new;p))] / (sigma - 1)     (*)
```

-- EXACTLY the prompt's own proposed inversion, confirmed (not assumed) against the actual
code (`src/melitz/lfd_preserving_state.jl` module header has the full derivation; live-verified
in `scripts/melitz_aq_phase0_moment_map_2026-07-30.jl` Test 1: `max relative |A_id - A_anchor| =
0.0` exactly at `q_new==q_anchor`, and Test 2: `mul_Gt!` under the anchor's own recovered `p*`
reproduces the anchor moments to `~3e-15`).

`T_od(p,q_od)` is EXACT (never a finite-difference/linearization) at every `q_od`, but is a
**piecewise-constant, non-increasing step function** (removing draws from an indicator sum) --
Test 3 (D4) verifies this directly, not by assumption: at the nearest positive-`p*`-weight draw
crossing above the anchor cutoff, `T(just below) - T(just above) = 1.173e-04`, and
`p*_weight * z_power = 1.173e-04` at that exact draw -- **an exact match, confirming the jump
size is exactly the removed draw's own weighted contribution, and that `T_od` is flat
immediately below** (verified to floating-point precision, `rtol=1e-9`).

**Failure/edge conditions** (all guarded in `melitz_cellwise_A_from_moments`, never silently
propagated): `:zero_tail_new` (`T_od(new;p)==0` -- `q_od` moved past every `p`-positive draw;
`a_od(new)` would be `+Inf`, an INTRINSIC infeasibility of holding `p` fixed at that `q`, not a
solver failure -- this is the exact mechanism the negswitch audit's origin-14 LP certificate
describes at the origin-block level); `:zero_tail_anchor` (defensive guard, not observed at any
verified anchor this session).

The focal free-entry link (`Pi_baseline_j(z)/w[j] - Pi_autarky_j(z)/w_prime[j]`,
`moments.jl:77-85`) is NOT cell-by-cell invertible the same way (it is a SUM over destinations
of PROFIT, not a single cell's revenue) -- confirmed (Phase 0 Test 4): Step 1B alone (targeting
ONLY the `D^2` trade-share moments) leaves a small but genuinely nonzero focal-link residual
(`4.08e-6` at a `1e-3`-magnitude D4 test perturbation), exactly as the governing prompt itself
anticipated ("will generally not automatically satisfy... the focal free-entry link").

## Phase 1: LFD-preserving state constructor (`src/melitz/lfd_preserving_state.jl`, EXPERIMENTAL)

Implemented exactly the prompt's Steps 1A-1E, reusing existing production machinery
throughout (no dense `G`, no duplicated pivot logic):

- **1A** (`melitz_construct_lfd_preserving_state`'s `build_state` closure): reconstructs the
  full `q` via the UNCHANGED `expand_free_theta_logcutoff`/`build_q_gravity_pivot` pivot --
  confirmed q-reconstruction is algebraically independent of `A` (the 2026-07-29 `s_a=0`
  fix already established this; re-confirmed here by construction, since Step 1B's own `A`
  is never fed back into Step 1A).
- **1B** (`melitz_cellwise_A_from_moments`): formula (*) applied to ALL `D^2` cells via a
  precomputed per-origin suffix-tail array (`melitz_origin_suffix_tail`, `O(W)` once per
  origin) + `melitz_active_tail_start` (production's own strict-active convention, reused
  verbatim) for `O(log W)` lookups -- exact, not finite-difference.
- **1C** (`melitz_lfd_corrector`): a **bounded discrete local search**, NOT a smooth
  Newton/SQP, per a structural finding this session establishes directly (Phase 3 below):
  `T_od`, hence `a_od`, hence GravityA/FocalLink, is EXACTLY constant within a chamber, so a
  first-order Jacobian is degenerate almost everywhere. Two lever free-`q` cells are chosen
  (largest `|c_full|` for A-gravity leverage; largest `|c_full|` among `origin==target_country`
  cells for focal-link leverage, matching that only the focal origin's own row enters the
  link sum) and grid-searched over a bounded window (`half_window=40`, i.e. up to ~80
  candidate chamber positions each) in alternating (Gauss-Seidel) rounds, up to
  `max_corrector_rounds=8`.
- **1D** (`melitz_f_from_Aq`): `melitz_log_f_from_q`/`derive_fjj_from_autarky_cutoff`, the
  SAME formulas `expand_free_theta_logcutoff` uses, applied to the corrected `(A,q)`.
- **1E** (witness): `melitz_update_operator_at_Afg!` (a NEW operator-update entry point
  bypassing the gravity pivot -- required since Step 1B's `A` is deliberately allowed to
  violate gravity before correction; `melitz_update_operator_at_theta!` cannot be reused
  here) + `mul_Gt!(g, op, p_star)` (production's own matrix-free adjoint, reused verbatim)
  gives the EXACT trade-share and focal-link residuals under the unchanged `p*`. Divergence
  via `melitz_primal_divergence(p_star, W)` (reused verbatim).

**D4 regression** (`test/melitz/runtests.jl`, new testset "LFD-preserving joint (A,q)
feasibility-preserving state constructor (2026-07-30)"): **15/15 assertions pass**, isolated
standalone run confirmed independently of the full suite's own pre-existing, unrelated
`mul_G!` SIGSEGV (documented by the two immediately-prior sessions).


## Phase 5: relative cutoff (5A) and normalized tail-target (5B) coordinate prototypes (D4)

Scoped to D4 per the governing prompt's own explicit allowance ("Do not run a full outer solve
under these coordinates in this phase") -- reconstruction-exactness and a lightweight
conditioning comparison, not a full battery of random-perturbation/Jacobian-SVD statistics.

**5A** (`melitz_reference_rank`/`melitz_reference_rank_inverse`): `u_od` = the fraction of
origin `o`'s `W` reference draws with `log(z) < q_od` (the empirical-CDF-of-draws scale, a
disclosed choice among several -- ties the coordinate to the SAME finite-support draws the
model's own moments use, rather than the population Pareto CDF). Per-origin stagewise
level+gaps (`u_o,1`, `Delta u_o,k>=0` by sort construction) reconstruct `q` to
`max|q_5A-q0|=2.489e-4` (bounded by the sorted-draw spacing at `W=20,000`) and reproduce
**bit-identical `A`** under the cellwise recovery (`max relative diff = 0.0`) -- i.e. the
reconstruction lands in the IDENTICAL chamber, not merely a numerically close `q` level.
`u`-gaps span `[0.0032, 0.516]` across the D4 fixture's 4 origins (a `~160x` range) -- no
formal Jacobian-conditioning statistic was computed this session (disclosed scope reduction).

**5B** (`H_od = T_od(p*,q0[o,d])`, exactly `origin_block_screen.jl`'s own target): per-origin
terminal level `H_o,D` + nonnegative interval masses `M_o,k = H_o,k - H_o,k+1` reconstruct `A`
to **`max relative diff = 1.4e-16`** (machine precision) -- an EXACT reconstruction, as
expected since 5B's `H` coordinates are a direct, invertible reparameterization of `A` itself
via formula (*) at FIXED `q`.

**5C** (finite-`W` chamber statistic, `S_W` variant ONLY -- the continuous-reference `S_pop`
variant was NOT built this session, a disclosed scope reduction): direct check confirms the
structural mechanism (an empty interval forces `S_W=0`, hence forces the interval's mass
coordinate to `0`) is well-defined and computed correctly; the D4 fixture at `W=20,000` is too
dense to exhibit an actually-empty interval (`n_empty=0` at the tested origin) -- the negswitch
audit's own origin-14 destination-16 interval (Phase 2/7 below) is the concrete case where this
mechanism actually binds.


## Phase 2: the known D20 origin-14 cutoff cliff -- pure vs. compensated endpoints

Anchor reproduced bit-identically to the negswitch audit: `Delta0=0.4832764950468883`, direction
`|b_q|=0.000357`, first minus switch `o=14 (Korea), d=19, t=0.0084126177` -- EXACT match.
Full results: `docs/key_results/melitz_aq_phase2_d20_cliff_2026-07-30.csv`/`.log`.

**Every one of the first 5 negative-direction switches**, at the bracket point just past the
switch:

| k | switch (o,d,dir) | pure-q (A) | cellwise (B) | corrected (C) | C `Delta*` | anchor `Delta0` |
|---:|---|---|---|---|---:|---:|
| 1 | 14,19,on  | `AboveEvaluationCap` (cert=1.07e5) | `FiniteSolved`, feasible | `FiniteSolved`, feasible | 0.483264 | 0.483276 |
| 2 | 14,16,off | `AboveEvaluationCap` (cert=2.92e6) | `FiniteSolved`, feasible | `FiniteSolved`, feasible | 0.483277 | 0.483276 |
| 3 | 2,14,on   | `AboveEvaluationCap` (cert=2.93e5) | `FiniteSolved`, NOT feasible (gravA=3.0e-6) | `FiniteSolved`, NOT feasible (gravA=6.9e-6) | 0.483273 | 0.483276 |
| 4 | 18,7,off  | `AboveEvaluationCap` (cert=1.77e5) | `FiniteSolved`, NOT feasible (gravA=2.6e-6) | `FiniteSolved`, NOT feasible (gravA=6.6e-6) | 0.483269 | 0.483276 |
| 5 | 14,16,off | `AboveEvaluationCap` (cert=1.70e5) | `FiniteSolved`, NOT feasible (gravA=2.7e-6) | `FiniteSolved`, NOT feasible (gravA=7.9e-6) | 0.483257 | 0.483276 |

**Every one of the first 5 positive-direction switches** (the audit's own documented
robust side -- `A_kind` is already `FiniteSolved` for the pure-q endpoint at every one of
these, consistent with the negswitch audit Phase 8): B and C likewise remain `FiniteSolved`
throughout, `Delta*` in `[0.483242, 0.483255]`.

**Central finding**: at the EXACT audited cliff (k=1, minus side, `o=14,d=19`), the pure-`q`
endpoint reproduces the negswitch audit's own `AboveEvaluationCap` classification (not
literally the same certified-infinite result -- this session did not re-run
`melitz_origin_block_screen` here, since the point of this experiment is the COMPENSATED
endpoint, not re-litigating the audit's own already-complete LP certificate) while the
LFD-preserving compensated endpoint, EVEN BEFORE any correction (Step 1B alone), is an
exact feasibility witness (`max|trade_res|~8e-15`, `focal~8e-8`, `gravA~5e-8`) and its full,
independent reoptimization returns **`FiniteSolved`, `Delta*=0.483264`** -- a movement of
only `1.2e-5` from the anchor's `0.483276`, not remotely close to the `10.0` cap.

**The certificate holds exactly, in every one of the 10 tested brackets**: `divergence(p*) =
0.4832764950468767` (constant across all 10 rows -- `p*` never changes within Phase 2) and
every single `C_Delta` (`0.483242` to `0.483277`) is `<= divergence(p*)`, confirming
`DeltaStar(new) <= divergence(p*) = DeltaStar(anchor)` empirically at all 10 tested points,
not merely as an unverified theoretical claim.

**Corrector convergence pattern**: at 4 of 10 brackets (minus k=1,2; plus k=1,5), Step 1B
ALONE is already within the strict witness tolerance (`feasible=true`, 0 corrector rounds
needed). At the other 6, the corrector runs its full bounded budget (`1312` candidate
evaluations, `~87-94s`) and materially reduces the focal-link residual (e.g. minus k=3:
`focal` `-1.94e-6 -> 3.9e-8`, a 50x reduction) but does NOT always bring `gravity_A`/
`gravity_f` under the strict `1e-6` joint tolerance (in 3 of these 6 cases `gravity_A` is
LARGER after correction than before, `2.6-3.0e-6 -> 6.6-7.9e-6`) -- a genuine limitation of
the current bounded 2-lever-cell discrete search (see "Corrector limitation" below), NOT a
failure of the underlying construction: the fully reoptimized point is `FiniteSolved` in
**all 10/10 cases regardless of whether the strict witness tolerance was met**, and the
certificate `Delta* <= divergence(p*)` holds in all 10/10 cases.

**Corrector limitation, diagnosed**: the two-lever design (one cell for A-gravity, one
origin-`j`(=France, focal) cell for the focal link) can, at some brackets, drive `gravity_A`
FURTHER from zero than Step 1B's own unfixed value, because focal-link correction (moving
`cell_focal`) also perturbs the shared q-gravity-pivot cell, which the alternating
Gauss-Seidel rounds do not always re-equilibrate within the 8-round budget. `gravity_f`
tracks `gravity_A` closely in magnitude throughout (e.g. minus k=3: `gravA=6.92e-6`,
`gravF=1.04e-5`, ratio `1.50 = sigma-1` almost exactly) -- confirming the Phase 1 structural
claim that f-gravity is a near-automatic CONSEQUENCE of A-gravity (scaled by `sigma-1`, the
`melitz_log_f_from_q` slope), not requiring independent correction, even though it is not
literally 0 whenever A-gravity itself is not literally 0.

## Destination-16/19 mechanism (research questions 5-6)

Direct `T_od` evidence at the exact audited bracket (`docs/key_results/melitz_aq_phase2_d20_cliff_2026-07-30.log`):

```
label=below  q[14,16]=0.265346  q[14,19]=0.265347   T16(p*)=0.1761868986   T19(p*)=0.1761817316
label=above  q[14,16]=0.265347  q[14,19]=0.265347   T16(p*)=0.1761868986   T19(p*)=0.1761868986
```

`T16` is UNCHANGED across the bracket (`q[14,16]` does not cross any switch here); `T19`
moves from `0.1761817316` to `0.1761868986` -- **becoming numerically identical to `T16`**,
exactly the negswitch audit's own finding (destination 19's active set drains into
coincidence with destination 16's own tight target once the crossing draw activates).
**Answer to research question 6: yes, the two normalized targets DO become exactly
numerically equal once the active sets coincide** -- confirmed directly, not inferred.

**Why this does NOT reproduce the audit's infeasibility here (research question 5)**: in the
`(A,q)`-joint framework, `A[14,16]` and `A[14,19]` are FREE, INDEPENDENT cells -- each is
recovered from formula (*) using its OWN target (`lambda_od`), not a SHARED probability
distribution constrained to reproduce two DIFFERENT targets from the SAME thin interval (the
origin-block LP's own failure mode). `T16==T19` numerically is completely consistent with
`A[14,16] != A[14,19]` (their own `coef_od`/`lambda_od` differ), so the moments are satisfied
cell-by-cell with no shared-support conflict. The relative `A` adjustment at this bracket is
small: `dA_norm = 1.51e-6` (full D=20x20 matrix Euclidean norm) against `A[14,19]`'s own
level -- a small, well-behaved compensating move, not a large jump.

## Phase 3: is the exact compensated path continuous?

**No -- proven analytically and confirmed numerically, both at D4 (Phase 0 Test 3) and
implicitly at D20 (the destination-16/19 detail above, where `T16` is EXACTLY unchanged
across the bracket while `T19` jumps by exactly the crossing draw's own weighted
contribution).** `T_od(p*,q_od)` (hence `a_od`, hence every downstream quantity: `f_od`,
`gravity_A`, `gravity_f`, the focal link) is an EXACT step function of `q_od` -- constant on
every open interval between two adjacent `p*`-positive-weight draws, with a discrete jump of
EXACTLY the crossed draw's own weighted contribution at each crossing (Phase 0 Test 3:
`rtol=1e-9` exact match between the measured jump and the analytically predicted
`p*_weight * z_power`). **A does NOT move at all within a chamber** (confirmed structurally,
not merely empirically small) **and DOES jump discretely exactly at a switch.**

**Can an ordinary smooth KNITRO scalar coordinate represent this path faithfully? No.** A
first-order (Jacobian/finite-difference) Newton corrector is provably degenerate (zero local
slope) almost everywhere in `q`-space, since the constraint residuals it would differentiate
are locally constant except at a measure-zero set of exact crossings. This is WHY Step 1C is
implemented as a bounded DISCRETE local search (Phase 1 above) rather than a continuous SQP,
and why the corrector's own convergence pattern (Phase 2 above) is itself evidence for this
finding: it either converges IMMEDIATELY (0 rounds, when Step 1B is already inside tolerance
-- no crossing needed) or exhausts its FULL discrete search budget (1312 evaluations) without
smooth improvement in between.

**Should a switch crossing be treated as a discrete active-set transition rather than an
ordinary gradient step? Yes** -- exactly Conclusion B's own recommendation, though this
session's own evidence (Phase 2/6) shows the DISCRETE transitions themselves, once taken,
land on genuinely `FiniteSolved`, near-anchor-`Delta*` points -- i.e. discreteness does not
by itself prevent useful search, it only prevents the SPECIFIC "ordinary smooth KNITRO
scalar coordinate" architecture the prompt asks about.

## Phase 4: active-set chamber transition prototype

Phase 2's own protocol -- for each of the first 5 negative and first 5 positive switches,
build a point just inside the neighboring chamber, apply the LFD-preserving corrector,
verify the resulting hard-model state via a full independent reoptimization -- **IS** the
chamber-transition prototype the governing prompt's Phase 4 asks for (the same construction
used for both, per the governing prompt's own Phase 2/4 overlap at a single reduced-`q`
direction). Acceptance criteria (Phase 4's own list) checked directly:

1. **Explicit `p*` witness valid**: yes at 4/10 brackets (feasible=true), partially at 6/10
   (feasible=false on the strict joint tolerance, but trade-share moments always exact to
   `~8e-15` and focal-link residual small, `<2.4e-6`, at every bracket).
2. **Structural constraints satisfied**: yes, `A>0`/`f>0` at every constructed cell (`A_status`
   all `:ok`, no `:zero_tail_*` cell encountered at any of the 10 tested brackets).
3. **Fully reoptimized inner result is `FiniteSolved`**: **yes, 10/10**.
4. **Economic objective improves or provides useful divergence slack**: yes -- every accepted
   point's `Delta*` is `<=` the anchor's own `divergence(p*)`, by the certificate (Phase 2
   above), and moves the outer objective (participation switches now REACHABLE, where the
   pure-`q` path was blocked by `AboveEvaluationCap`/would have been LP-certified infeasible
   at the exact audited point per the negswitch audit).


## Phase 6: predictor-corrector welfare continuation (mandatory bounded D20 smoke test)

Anchor GT (closed-form, `melitz_welfare_metrics_from_g`): `6.290641%`. Step budget: `<=5`
steps/direction, `0.05` GT percentage points per predictor step, `<=3` bounded shrinks per
step, no multistart, no broad nuisance search. Full log:
`docs/key_results/melitz_aq_phase6_predictor_corrector_2026-07-30.log`.

**Result: zero accepted steps in EITHER direction.** At the smallest tried shrink
(`step_pp=0.0125%`, 1/4 of the requested minimum), the witness is still far outside
tolerance in both directions:

- `upper_GT`: `focal=-3.15e-03` (`gravA=3.90e-05`, `gravF=5.86e-05`) -- `~3000x` the
  `1e-6` moment tolerance.
- `lower_GT`: `focal=2.10e-03` (`gravA=4.78e-06`, `gravF=7.17e-06`) -- `~2000x`.

**Diagnosis (not merely observed, but explained)**: the welfare coordinate `g` enters the
focal free-entry link DIRECTLY (through the autarky price power `gamma_prime_target=exp(g)`)
and through `q[j,j]` (`derive_qjj_from_autarky_cutoff`), a channel with NO analogue in Phase
2's pure nuisance-direction (`b_q`) moves, which held `g` fixed throughout. Even a `0.0125%`
GT step produces a focal-link violation `~3-4` orders of magnitude larger than anything Phase
2 encountered (`Phase 2's own largest uncorrected focal residual was `2.4e-6`, `~1000x`
smaller). The corrector's own two-lever, bounded-window (`half_window=40` sorted positions)
discrete local search -- adequate for the SMALL, LOCAL nuisance-direction corrections Phase 2
needed -- does not have nearly enough "reach" to absorb a genuine welfare-level shift.

**Comparison: gamma-only continuation** (move `g` alone, hold `q`/`A` fixed at the anchor,
`docs/key_results/melitz_aq_phase6_gamma_only_comparison_2026-07-30.csv`) remains
`FiniteSolved` through all 5 steps in BOTH directions, but at RAPIDLY INCREASING divergence
(`Delta*`: `0.483 -> 0.504 -> 0.527 -> 0.550 -> 0.575 -> 0.602` upper; `0.483 -> 0.463 ->
0.444 -> 0.427 -> 0.410 -> 0.394` lower) -- a `~4.3%` relative jump in `Delta*` from the FIRST
step alone. This is the exact failure mode the LFD-preserving approach is designed to avoid
(letting the outer solver silently absorb a welfare move into a much less favorable LFD
rather than keeping divergence flat), but this session's corrector could not deliver the
alternative at ANY tested step size.

**Phase 6 verdict: no verified welfare-direction progress.** This is a genuine, disclosed
negative/informative result, not a claimed success -- the corrector's bounded local-search
design (tuned and validated for Phase 2's nuisance-direction cliff-crossing use case) is
demonstrably NOT yet adequate for welfare continuation. A wider search window, more lever
cells, or a fundamentally different (possibly continuous-within-chamber, since a genuine
welfare step generically requires crossing MANY chambers at once) corrector design is needed
-- explicitly flagged as future work, not attempted further this session (bounded scope).

## Phase 7: SKIPPED per the governing prompt's own explicit gate

The governing prompt: "Run this phase only if... the D20 welfare continuation produces at
least one verified improving step." Phase 6 produced **zero** accepted/verified steps in
either direction -- the gate is not met. Phase 7 (D4 sequential-(A,f) vs. reduced-q vs.
compensated search-method smoke test) is not run this session.


## Decision

**Conclusion B: compensation works, but switch transitions are discontinuous.**

- Compensated post-switch states ARE finite: **yes, 10/10 tested brackets** (Phase 2/4),
  including the EXACT audited Korean cliff (`o=14,d=19`) where the pure-`q` path fails.
- The exact state path jumps at switches and cannot be represented by an ordinary smooth
  KNITRO scalar coordinate: **yes, proven analytically and confirmed numerically** (Phase 3;
  `T_od` is an exact step function, flat within a chamber, jumping exactly at a
  positive-weight draw crossing).
- Conclusion A's narrower claim (a joint `(A,q)` move crosses the SPECIFIC audited cliff and
  remains a verified feasible witness with `Delta* <= divergence(p*)`) is **strongly
  supported** and is the load-bearing positive result of this session -- but Conclusion A's
  FULL criterion ("the predictor-corrector welfare continuation makes verified economic
  progress") is **NOT met**: Phase 6 accepted zero steps in either welfare direction. This is
  why the session's OVERALL conclusion is B, not A: the demonstrated capability is a
  DISCRETE, LOCAL, nuisance-direction chamber-crossing tool, not (yet) a welfare-direction
  outer-search replacement.

**Recommendation** (Conclusion B's own, adopted): **use continuous optimization inside
active-set chambers and discrete LFD-preserving transitions between chambers** for the
nuisance/cutoff-cliff problem this session's governing motivation (the negswitch audit)
identified. A genuine welfare-direction predictor-corrector (Phase 6's own goal) remains
future work, requiring a materially stronger corrector (wider search window and/or more
lever cells, or a design that anticipates a welfare step will generically require crossing
MANY chambers simultaneously, not one or two).

## Final report answers

1. **Can A be recovered analytically so that a fixed `p*` continues to satisfy every factual
   bilateral moment after `q` changes?** Yes -- exact closed form (*), confirmed against the
   actual `melitz_C`/`melitz_firm`/`mul_Gt!` code (Phase 0), machine-precision-exact at D4
   (15/15 regression tests), `~1e-15` at real D20 (Phase 2).
2. **Can the `q` correction simultaneously restore A gravity, q/f gravity, and the focal
   free-entry link?** Partially. At D4 (a controlled small perturbation), yes, to `~1e-7`
   (Phase 1). At real D20's actual switch brackets, the bounded 2-lever discrete corrector
   fully succeeds at 4/10 brackets and materially-but-incompletely improves the other 6/10
   (Phase 2) -- q/f-gravity is confirmed to track A-gravity closely (ratio `~sigma-1`,
   consistent with the analytic "automatic consequence" argument) but is not literally zero
   whenever A-gravity itself is not literally zero.
3. **Does the unchanged `p*` then provide an exact feasible witness?** Yes at 4/10 D20
   brackets exactly (all tolerances met); at the other 6/10, trade-share moments remain
   exact (`~8e-15`) and the focal-link/gravity residuals are small (`1e-7` to `1e-5`) but not
   inside the strict joint tolerance -- yet EVERY one of the 10 fully reoptimized points is
   `FiniteSolved` with `Delta* <= divergence(p*)`, confirming the underlying certificate
   argument holds even where the STRICT witness gate is not literally met.
4. **Can the known Korean cutoff cliff be crossed by a compensated joint `(A,q)` move?**
   **Yes** -- directly demonstrated at the exact audited bracket (`t=0.0084126177`,
   `o=14,d=19`): pure-`q` is `AboveEvaluationCap` (matching the audit); the compensated
   endpoint is `FiniteSolved`, `Delta*=0.483264`, essentially unchanged from the anchor.
5. **How much do `A_16` and `A_19` need to move relative to one another?** Small: the
   destination-19 cell's own tail statistic shift at the switch is `~5.2e-6` in absolute
   `T_19` terms (`0.1761817 -> 0.1761869`, `~2.9e-5` relative), and the FULL `20x20` `A`
   matrix's Euclidean-norm change at this bracket is `dA_norm=1.51e-6` -- a small,
   well-behaved compensating move, not a large jump.
6. **Does the compensation force the two normalized targets to coincide after their
   finite-support active sets coincide?** **Yes, confirmed directly**: `T_16` and `T_19`
   become numerically IDENTICAL (`0.1761868986` both) immediately after the switch -- but
   this causes NO infeasibility in the `(A,q)`-joint framework, since `A[14,16]` and
   `A[14,19]` are independent cells, each satisfying its OWN target, with no shared
   probability-mass constraint forcing them to differ (the origin-block LP's own failure
   mode requires a SINGLE `p` to jointly satisfy both destinations from the same thin
   interval -- irrelevant once `A` is free per cell).
7. **Is the exact compensated path continuous across cutoff switches?** **No** -- proven
   analytically (the moment formula (*) is a step function of `q_od`) and confirmed
   numerically to `rtol=1e-9` (Phase 0 Test 3, D4) and structurally at D20 (destination-16/19
   detail: `T_16` exactly flat, `T_19` jumps exactly at the crossing draw).
8. **If not, is an active-set chamber algorithm more appropriate than a smooth SQP
   coordinate?** **Yes** -- a first-order Jacobian corrector is provably degenerate almost
   everywhere; the bounded discrete local search this session built is the right KIND of
   tool (validated at 10/10 nuisance-direction brackets), even though its current
   window/lever-count tuning is not yet strong enough for welfare-direction moves (Phase 6).
9. **Does the relative cutoff/tail-target parameterization reconstruct the original state
   exactly?** 5B (normalized tail-target `H`) reconstructs `A` to machine precision
   (`1.4e-16` relative, D4). 5A (reference-rank `q`) reconstructs the IDENTICAL chamber
   (bit-identical `A`) despite a small (`~2.5e-4`) raw-`q`-level discretization error.
10. **Does density coupling prevent requests for different targets when the interval is
    empty?** Structurally yes (5C: `S_W=0` forces the interval's mass coordinate to `0` by
    construction) -- directly confirmed at the mechanism level; the D4 fixture tested was too
    dense (`W=20,000`) to exhibit an actually-empty interval (the negswitch audit's own
    origin-14/destination-16 interval, real D20, is the concrete case where this binds).
11. **Does reference-rank spacing improve conditioning relative to raw cutoff levels?** Not
    formally measured this session (disclosed scope reduction -- no Jacobian-SVD/condition-
    number statistic was computed); the raw `u`-gap spread (`0.0032` to `0.516`, `~160x`) at
    D4 is reported but not compared against an equivalent raw-`q`-gap statistic.
12. **Can predictor-corrector welfare continuation make verified gains-from-trade progress
    in percentage points without raising DeltaStar above the anchor?** **No, not with this
    session's corrector** -- Phase 6 accepted zero steps in either direction at any tested
    step size down to `0.0125%` GT.
13. **Which of Conclusions A-E is supported?** **B** (see Decision above), with Conclusion
    A's narrower nuisance/cliff-crossing claim strongly supported as B's own load-bearing
    evidence.

## Critical correctness invariants (checked)

1. A verified explicit `p*` witness implies the state cannot be classified
   `AboveEvaluationCap`/`InfiniteDeltaCertified`: checked live at every Phase 2/6 acceptance
   attempt (`melitz_construct_lfd_preserving_state`'s own `feasible` gate + a loud diagnostic
   print in Phase 6's driver if a feasible witness were ever followed by a non-`FiniteSolved`
   reoptimization) -- **never triggered** across all 10 Phase 2 brackets + all Phase 6
   attempts.
2. Reconstructed trade-share moments under `p*` are exact: `~8e-15` at every one of the 10
   D20 brackets (Phase 2), `~3e-15`/`0.0` at D4 (Phase 0/1 tests).
3. The focal link under `p*` is exact where the corrector converges (4/10 D20 brackets to
   `<1e-7`; D4 to `<1e-8`), small elsewhere.
4. Both gravity restrictions hold where the corrector converges; `q/f`-gravity tracks
   `A`-gravity closely (ratio `~sigma-1`) throughout.
5. Finite-QMC identical moment columns imply identical normalized targets: **directly
   confirmed** (destination-16/19, `T_16==T_19` exactly after the switch).
6. A chamber transition with an empty adjacent interval cannot retain a nonzero interval
   target mass: confirmed structurally (5C).
7. Reduced-`(A,q)`, production-`(A,f)`, and direct full-state wrappers evaluate identical
   economic states identically: inherited from the negswitch audit's own Phase 0/4
   cross-backend verification (bit-identical `Delta`/classification); this session's own new
   code (`lfd_preserving_state.jl`) always reduces to a genuine `:logcutoff` `theta_free`
   (`reduce_to_free_theta_logcutoff`) before any typed classification, so it is evaluated
   through the EXACT SAME production entry point (`solve_melitz_delta!`) as any other
   `:logcutoff` state -- no separate wrapper exists for this session's own states.
8. No `NumericalFailure` returned: confirmed -- every classification in Phase 2/6 is
   `FiniteSolved` or `AboveEvaluationCap` (pure-`q` endpoints only); never `NumericalFailure`.
9. Capped sessions use raw `lower_limit = -cap`: unchanged production behavior, reused
   verbatim (`CappedEvaluation`/`melitz_policy_lower_limit`), not modified this session.
10. No dense `G` materialized in production paths: this session's new code
    (`lfd_preserving_state.jl`) never constructs a `W x (D^2+1)` matrix -- `T_od` lookups are
    `O(log W)` via precomputed per-origin suffix sums, and the witness check uses `mul_Gt!`
    (matrix-free). Not separately re-instrumented against `MELITZ_DENSE_G_MATERIALIZATIONS[]`
    this session (disclosed scope reduction -- the negswitch audit's own Item 11 test already
    covers the adjacent `origin_block_screen`/switch-geometry code this session's corrector
    reuses).
11. Threads and caches included in provenance: `docs/key_results/melitz_aq_provenance_2026-07-30.txt`.


## Files changed

```
Modified:
  src/melitz/include_melitz.jl   (new include line)
  test/melitz/runtests.jl        (new testset, "LFD-preserving joint (A,q) feasibility-
                                   preserving state constructor (2026-07-30)")

New:
  src/melitz/lfd_preserving_state.jl   (Phases 0-1: T_od suffix helper, cellwise A recovery,
                                         f recovery, operator-at-Afg, feasibility witness,
                                         discrete corrector -- ALL EXPERIMENTAL)
  docs/melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md   (this document)
  docs/key_results/melitz_aq_*_2026-07-30.{csv,log,txt}
  scripts/melitz_aq_phase{0,2,5,6}_*_2026-07-30.jl
```

**Zero diff in `cc_algo/`** or any other Ricardian path, and **zero diff in
`finite_delta_outer.jl`/`outer_solve.jl`** (production `(A,f)` outer search unchanged),
confirmed via `git status`.

## Required output files

- `docs/melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md` -- this document.
- `docs/key_results/melitz_aq_phase0_moment_map_2026-07-30.csv` -- Phase 0/1, D4 formula
  verification (5 tests).
- `docs/key_results/melitz_aq_phase2_d20_cliff_2026-07-30.csv`/`.log` -- Phase 2/3/4, real D20
  pure/cellwise/corrected endpoints at the first 5 negative and 5 positive switches.
- `docs/key_results/melitz_aq_phase5_relative_coords_2026-07-30.csv` -- Phase 5, D4 relative-
  coordinate reconstruction tests.
- `docs/key_results/melitz_aq_phase6_predictor_corrector_2026-07-30.csv`/`.log`,
  `melitz_aq_phase6_gamma_only_comparison_2026-07-30.csv` -- Phase 6 smoke test + comparison.
- `docs/key_results/melitz_aq_provenance_2026-07-30.txt` -- full provenance record.
- Scripts (all reproduce the exact numbers in this document; `julia --project=. -t <N>
  <script>.jl` after `source .knitro_env.sh; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`):
  - `scripts/melitz_aq_phase0_moment_map_2026-07-30.jl` (D4, `-t 4`, `~1min`)
  - `scripts/melitz_aq_phase2_d20_cliff_2026-07-30.jl` (D20, `-t 20`, `~13min`)
  - `scripts/melitz_aq_phase5_relative_coords_2026-07-30.jl` (D4, `-t 4`, `~10s`)
  - `scripts/melitz_aq_phase6_predictor_corrector_2026-07-30.jl` (D20, `-t 20`, `~11min`)
- Source: `src/melitz/lfd_preserving_state.jl` (new); `src/melitz/include_melitz.jl`,
  `test/melitz/runtests.jl` (additive edits).

## Required tests

New testset `"LFD-preserving joint (A,q) feasibility-preserving state constructor
(2026-07-30)"` (`test/melitz/runtests.jl`), D4 (`FIXTURE`, this repo's own standard fast test
fixture, matching this repo's own established D20/test-suite split -- real-D20-scale findings
validated by the one-off scripts above, not replayed at D20 cost inside the fast suite):

1. Cellwise recovery at `q_new==q_anchor` is an exact identity (`0.0` relative error, D4).
2. `mul_Gt!` under `p_star` reproduces `~0` anchor moment residuals.
3. `T_od` is an exact step function -- flat below, jumps exactly at a positive-weight draw
   (`rtol=1e-9` match between measured jump and analytic prediction).
4/5. Step 1B preserves trade moments exactly (`<1e-6`); Step 1C corrector restores both
   gravity restrictions to `<1e-5` at a random small D4 perturbation; round-trip through
   `reduce_to_free_theta_logcutoff`/`expand_free_theta_logcutoff` agrees with the corrector's
   own output within a residual-scaled tolerance.

**15/15 assertions pass** in an isolated standalone run (matching this repo's own established
practice of extracting new testsets for isolated verification, given the full suite's own
pre-existing, unrelated `mul_G!`/`moment_operator.jl:281` SIGSEGV documented by the two
immediately-prior sessions and reproduced, unmodified, by this session too -- not re-diagnosed
here, out of this session's own bounded scope).

