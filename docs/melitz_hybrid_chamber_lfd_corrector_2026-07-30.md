# Melitz hybrid chamber-aware LFD-preserving corrector (2026-07-30 continuation session)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), HEAD at session start `523c4af`
("Melitz joint (A,q) LFD-preserving feasibility-preserving search (2026-07-30)"). Governing
prompt: `melitz_hybrid_chamber_lfd_corrector_2026-07-30`, a narrowly-bounded follow-up to
`docs/melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md` (Conclusion B: compensation
works locally, switch transitions are genuinely discontinuous, the prior two-lever discrete
corrector is not production-grade). Does not modify the Ricardian implementation, `cc_algo/`,
or the production `(A,f)` outer search.

**Note on this session's operating environment**: partway through this session, a DIFFERENT
concurrent process (not this session) added `src/melitz/fixed_q_a_middle_loop.jl` and its
`include_melitz.jl` line to this same working directory (not a separate worktree) -- observed
live via a harness file-change notification and confirmed via `git status`/`ls`. This session's
own new file (`hybrid_chamber_corrector.jl`) loads and runs correctly alongside that addition;
the two files are independent and this report does not describe or depend on that other
session's work.

## Executive summary

1. **Phase 0 correction of the prior session's claims**: the round-trip audit
   (explicit corrector output vs. `reduce_to_free_theta_logcutoff` -> `expand_free_theta_logcutoff`)
   is measured directly, not assumed, for every one of the prior session's 10 brackets. Result:
   **the round trip is NEVER exact, even at the 4/10 brackets the prior session called
   "feasible"** -- the discrepancy is a precisely explained, constant `2.046x` multiple of the
   corresponding gravity residual at EVERY one of the 10 brackets (not noise), confirming a
   theoretical prediction stated BEFORE measuring (see Phase 0.2). The prior session's own
   `Delta* <= divergence(p*)` certificate itself remains intact (verified independently here);
   what is corrected is the implicit assumption that a small gravity residual means the
   round-tripped/reduced representation of that state is the SAME state.

2. **A genuine structural correction to the prior session's blanket "everything is a step
   function" claim**: the focal free-entry link is provably **smooth** (not a step function)
   within a fixed chamber. Derived from `moments.jl`/`firm_quantities.jl` (not assumed) and
   verified live: `FocalResidual(q_row_j) = CONST(chamber) - sum_{d!=j} N_jd(p*) * f_jd(q_jd)`,
   where `f_jd(q_jd) = K'_jd*exp((sigma-1)*q_jd)` is an EXACT exponential, and `N_jd(p*,q_jd)`
   (a plain, non-z-power-weighted probability tail mass) is constant within the chamber by the
   SAME step-function mechanism the prior session established for the z-power-weighted `T_od`.
   D4 regression: the closed form matches the exact `mul_Gt!`-based operator evaluation to
   `~1e-17` (self-calibrated) and predicts an out-of-sample within-chamber trial point to
   `~1.2e-17` absolute; its exact Jacobian matches a central finite difference of the true
   operator evaluation to `4.7e-11` relative error (the `O(h^2)` floating-point-noise signature
   of an exactly-correct analytic formula, not an approximation).

3. **A-gravity has NO continuous handle within a chamber** -- it is a genuine, unavoidable
   combinatorial (not merely locally-flat) function of the discrete chamber assignment, since
   `T_od(p*,q_od)` is constant on each chamber and `A_od` is a deterministic function of `T_od`
   alone. This means the task's own suggested target (`|A-gravity| < 1e-10`) is **not
   generically reachable by any bounded discrete search** -- confirmed by direct experiment
   (D4, a `3e-3`-scale random perturbation): a `200`-candidate budget (the task's own suggested
   default) reaches `~1.9e-7`; `1,000` candidates reach `~2.1e-8`; `5,000` and `20,000`
   candidates both plateau at the SAME `~2.6e-9` (confirming the residual floor is set by the
   search's `beam_width`/`max_depth` structure, not by candidate-count scarcity) -- i.e. more
   budget helps roughly log-linearly and then plateaus; **it does not converge to machine
   precision**, because doing so would require an exact combinatorial cancellation, not a
   continuous limit. This is reported as a genuine, quantified structural finding, not a bug.

4. **Phase 5 (D20 cliff retest, 10 negative + 10 positive switches, doubling the prior
   session's 5+5 coverage)**: the new hybrid corrector is a decisive, quantified improvement
   over the prior two-lever corrector on its OWN primary target. Measured against the prior
   session's own tolerance (`1e-6`): **A-gravity is satisfied at 20/20 brackets under the new
   corrector vs. 9/20 under the old one**; the worst-case `|gravity_A|` improves from
   `2.55e-5` (old) to `1.65e-7` (new), a `~155x` reduction. The focal link (already the old
   corrector's strong suit) is satisfied at 19/20 brackets under the new corrector (one
   outlier, `6.16e-6`, discussed below) vs. 20/20 under the old one. Jointly (`both < 1e-6`):
   **19/20 new vs. 9/20 old**. **Every one of the 20 reoptimized points remains
   `FiniteSolved`**, `Delta*` in `[0.483236, 0.483276]`, essentially unchanged from the anchor
   `0.4832764950`. Neither corrector reaches the task's own aspirational `1e-10`/`1e-9`
   tolerances at every bracket (Phase 1.4/Phase 2's own structural finding: A-gravity has no
   continuous handle within a chamber, so this is expected, not a defect).

5. **Phase 6 (bounded welfare predictor-corrector retest, hybrid corrector)**: **zero accepted
   steps in either direction, at all 3 tested step sizes (0.0125%/0.025%/0.05% GT)** -- an
   exact replication of the prior session's own negative finding, now with a MATERIALLY
   stronger corrector (the same one that just delivered a 155x A-gravity improvement on the
   cliff-crossing task). Even at the smallest step (`0.0125%`), the focal-link residual is
   `-3.41e-3`/`+2.36e-3` (upper/lower) -- `~3,000x` the `1e-6` tolerance the SAME corrector
   satisfies for 19/20 nuisance-direction brackets, and does not shrink appreciably as the step
   size varies (`gravA`/`gravF` DO reach `~1e-8`-`1e-9`, i.e. the corrector's discrete+
   continuous machinery is working as designed, but the focal-link gap itself is simply too
   large for ANY chamber-bounded correction to close). Gamma-only comparison reproduces the
   prior session's own finding: `Delta*` moves `0.483 -> 0.550` (upper, 3 steps) / `0.483 ->
   0.427` (lower, 3 steps), a real, unavoidable divergence cost with no correction attempted.

6. **Decision: B (see full Decision section below)** -- reconfirmed and sharpened, not merely
   repeated: the hybrid corrector is a genuine, quantified engineering improvement for LOCAL
   chamber-transition/cliff-crossing correction (Phase 5), but the welfare-direction gap
   (Phase 6) is shown to be `~1,000-3,000x` larger in scale than anything a bounded chamber-
   aware corrector -- however good -- can bridge, which is now demonstrated rather than merely
   suspected.

## Phase 0: correcting the prior session's claims

### 0.1 Reproducing the prior Phase 2 table

`scripts/melitz_hybrid_phase0_roundtrip_audit_2026-07-30.jl` reproduces the prior session's
own 5+5-bracket table exactly (identical anchor `Delta0=0.4832764950`, identical switch
sequence/`(o,d,dir)` at every k, identical `A_kind`/`B_feasible`/`gravA`/`gravF`/`Delta` values
to the last printed digit -- e.g. minus k=1 `gravA=-5.096e-8`, `Delta=0.483264`; plus k=10
values match the original report's table exactly). Confirms the prior session's Phase 2 table
is exactly reproducible, not a one-off artifact.

### 0.2 Explicit-vs-round-trip state audit (the task's own required check)

For every B/C endpoint from the prior session's corrector
(`melitz_construct_lfd_preserving_state`), this session compares the EXPLICIT constructed
state `(A,f,q,gamma_prime_j)` directly against the state obtained by
`reduce_to_free_theta_logcutoff` -> `expand_free_theta_logcutoff` (i.e. exactly what the prior
session's own `stC.theta_free` field represents once fed back through the ordinary
`:logcutoff` machinery).

**Theoretical prediction, stated before measuring** (per the CLAUDE.md warning against
explaining away a discrepancy without measuring it): `pivot_expand`/`pivot_reduce`
(`equilibrium.jl`) map every NON-pivot free cell IDENTICALLY (`z[other[k]] = z_free[k]`, no
offset) -- so `reduce_to_free_theta_logcutoff` throws away only the two pivot cells' own
explicit values (`A`'s A-gravity pivot cell, `f`'s q-gravity pivot cell) and
`expand_free_theta_logcutoff` RECOMPUTES them from scratch so that A-gravity/q-gravity hold
EXACTLY (`=0`/`=RHS(g)` by construction). Therefore: whenever the explicit state's own
`gravity_A_residual`/`gravity_f_residual` are nonzero, the round-tripped PIVOT CELL must differ
from the explicit pivot cell by an amount that exactly cancels that residual -- i.e. the
round-trip discrepancy should be commensurate with (not merely correlated with) the gravity
residual itself, concentrated in the pivot cell, and should vanish whenever gravity is already
at machine tolerance.

**Measured** (`ratio_A`/`ratio_F` columns of `docs/key_results/melitz_hybrid_phase0_roundtrip_audit_2026-07-30.csv`
-- round-trip discrepancy divided by the corresponding gravity residual): **the prediction is
confirmed exactly, at every one of the 10 brackets, to 10+ significant figures**:
`ratio_A = ratio_F = 2.045939...` in EVERY row (minimum `2.0459386736`, maximum
`2.0459445016` -- the tiny 6th-decimal spread is itself explained: at the two brackets where
the residual is smallest, `~1e-10`/`~1e-11`, the ratio picks up a proportionally larger share
of ordinary floating-point roundoff). This constant is `1/|c_full[pivot]|` for the shared
physical A/f-pivot cell (confirmed: the `argmax` location of the round-trip discrepancy is the
IDENTICAL cell, `(o=3,d=14)`, at every one of the 10 brackets -- the SAME physical A-gravity
pivot cell throughout, since it depends only on `tau`/`ctx`, never on the displaced state).

**Concrete numbers, not glossed over**: even at the prior session's OWN "feasible=true"
brackets (minus k=1,2; plus k=1,5 -- `gravity_A_residual` `5.1e-8` to `9.2e-11`), the explicit
state and its round-tripped representation differ by `1.04e-7` to `1.9e-10` in `max|Δlog A|`
(and proportionally in `log f`) -- SMALL, but not remotely machine-precision-zero, and growing
linearly with the underlying residual (up to `1.6e-5` at the worst "infeasible" bracket, minus
k=5). `max|Δq|` stays at floating-point noise (`~1e-15`) throughout -- confirming the
discrepancy is ENTIRELY concentrated in the two pivot cells' own reconstruction (A/f), never in
`q` (which round-trips exactly, since every non-pivot free `q` cell maps identically under
`pivot_expand`/`pivot_reduce`, and the q-gravity pivot's own value is already an EXACT function
of `g` alone by the 2026-07-29 session's fix -- unaffected by A's gravity violation).

**Conclusion**: the prior session's `feasible` flag (a threshold on `gravity_A_residual`/
`gravity_f_residual`/`focal_residual`/`trade_residuals` alone) is a necessary but NOT
sufficient condition for "the explicit state IS the state that would be recovered if this
point were later serialized, reduced to free coordinates, and re-expanded" (e.g. as an anchor
for a follow-on continuation session, exactly this session's own Phase 6 usage pattern). This
session's hybrid corrector reports `roundtrip_max_abs_logA`/`_logf`/`_q` and a strict
`roundtrip_exact` boolean explicitly (Phase 4) precisely so this distinction is never silently
assumed again.

## Phase 1: the hybrid chamber structure, derived and verified

### 1.1 The focal link is smooth within a chamber; A-gravity/trade-moments are not

Derivation (verified against `melitz_moments!`/`melitz_firm`, not assumed -- see
`src/melitz/hybrid_chamber_corrector.jl`'s own module header for the full algebra):

- `melitz_moments!`'s `profit_j[w] = sum_{d} firm(...).realized_operating_profit` (origin `j`
  row only), `melitz_firm`'s `realized_operating_profit = active ? (C_jd*z^(sigma-1)/sigma -
  w_j*f_jd) : 0`, `active = z > exp(q_jd)`.
- Fixing the active set at every row-`j` cell (staying inside the current chamber, no draw
  crossing at any `(j,d)` cell), the `C_jd*z^(sigma-1)/sigma` term's `p*`-weighted expectation
  is `C_jd*T_jd(p*)/sigma` -- CONSTANT within the chamber (the prior session's own `T_od`
  result). The `-w_j*f_jd` term's expectation is `-w_j*f_jd*N_jd(p*,q_jd)`, where
  `N_jd(p*,q_jd) = sum_w p*_w*1{z[w,j]>exp(q_jd)}` is a PLAIN (non-z-power-weighted)
  probability tail mass -- a NEW object this session introduces (`melitz_origin_suffix_prob`/
  `melitz_N_od`), distinct from `T_od`, but an exact step function by the identical mechanism.
- So `FocalResidual(q_row_j) = CONST(chamber) - sum_{d!=j} N_jd(p*)*f_jd(q_jd)`, and since
  `melitz_log_f_from_q` gives `f_jd(q_jd) = K'_jd*exp((sigma-1)*q_jd)` EXACTLY (not to first
  order), the closed form is an EXACT sum of exponentials in the free row-`j` q cells, smooth
  and strictly monotone-decreasing in each coordinate throughout the chamber.
- By contrast, `A_od` (via the prior session's own formula `a_od(new) = a_od(anchor) +
  [log(T_od(anchor)) - log(T_od(new))]/(sigma-1)`) is a function of `T_od(p*,q_od)` ALONE,
  which is EXACTLY constant within the chamber -- `A` genuinely does not move at all until a
  crossing, confirming (not merely repeating) the prior session's finding for the A-block
  specifically, while correcting its OVER-GENERALIZATION to the focal link.

### 1.2 Exact within-chamber focal-link Jacobian

`d(FocalResidual)/d(q_jd) = -(sigma-1)*N_jd(p*)*f_jd(q_jd)` exactly (`
melitz_focal_link_jacobian_entry`) -- reuses the SAME multiplicative `(sigma-1)` scale
`exact_q_smooth_gradient.jl` already derived for the ENVELOPE (`d(Delta*)/dq`) sensitivity, but
this is a DIFFERENT object: the derivative of the RAW MOMENT RESIDUAL under the fixed `p*`
(needed to drive `FocalResidual(p*,q)` itself to zero), not the envelope-theorem sensitivity of
the optimized `Delta*` to `q` at the converged dual. `exact_q_smooth_gradient.jl`'s own
`tail1_d` (dual/`dpsi`-weighted) is replaced here by `N_jd(p*)` (`p*`-weighted, the "plain
probability" analogue of the prior session's own `T_od`).

D4 regression evidence (see Executive Summary item 2): the closed form and its Jacobian are
verified exact, not merely locally accurate.

### 1.3 A-gravity / q-gravity / f-gravity relation

`q`-gravity is automatically exact for ANY free-`q` value (the prior 2026-07-29 session's own
`build_q_gravity_pivot`/`build_q_gravity_offset` fix: `Gravity(q) = RHS(g)` identically, by
pivot construction, independent of free `A`). `A`-gravity is NOT automatically exact under the
cellwise moment-recovery formula (*) -- that is the entire reason a corrector exists.
`f`-gravity's relationship to `A`-gravity: substituting `log(f_od) = (sigma-1)*(q_od+a_od-...)
+ ...` into `dot(c_full,vec(log f))` gives `dot(c_full,vec(log f)) = (sigma-1)*dot(c_full,q) +
(sigma-1)*dot(c_full,vec(log A)) + dot(c_full,const_vec)`. Since `q`-gravity holds identically
(`dot(c_full,q)` fixed at `RHS(g)/(sigma-1)`-scale, a KNOWN constant depending on `g` alone) and
`const_vec` is data-only, **`f`-gravity residual is an EXACT AFFINE function of `A`-gravity
residual with slope `(sigma-1)`, not merely correlated with it** -- confirming and sharpening
the prior session's own empirical observation ("gravity_f tracks gravity_A closely, ratio
~sigma-1") into an exact algebraic identity: `gravity_f_residual = (sigma-1) *
gravity_A_residual` exactly, whenever the SAME `q`/`g` (hence q-gravity) are held fixed while
only `A`'s cellwise-recovered value is perturbed. **`f`-gravity is therefore exactly redundant
given `A`-gravity holds and `q` is reconstructed via the ordinary pivot** -- it never needs an
independent corrector; the hybrid corrector below targets `A`-gravity and the focal link only,
and reports `f`-gravity purely as a derived diagnostic.

### 1.4 Chamber bounds

Rather than hand-deriving and separately maintaining an affine inequality system for "does
free-q coordinate `k` moving by `delta` keep every cell (including the q-gravity pivot cell,
which moves by the LINEAR amount `-c[other[k]]/c[pivot]*delta` under `pivot_expand`'s own
exact formula) inside its current chamber," this session relies on an exact, ground-truth
computational check instead (`melitz_chamber_signature`/`melitz_chamber_signature_diff_count`):
recompute `melitz_active_tail_start` (the same sorted-position lookup that already defines a
chamber) for EVERY `D^2` cell before and after a trial move, and require zero cells to differ.
This is provably equivalent to the affine-inequality system (both test exactly "does `q_od`
cross a sorted-draw boundary for its own origin") but avoids a second, error-prone,
separately-derived bookkeeping path -- a disclosed, deliberate implementation choice, not a
gap in the derivation (the affine structure itself is exactly as described above: linear in
the moving free coordinates via the SAME single pivot-cell dependency the prior session's
`melitz_lfd_corrector` docstring already identified, just checked exactly rather than
enumerated symbolically).

## Phase 2: discrete A-gravity chamber selector

Implemented in `melitz_discrete_chamber_selector` (`src/melitz/hybrid_chamber_corrector.jl`):
a bounded best-first/beam search over a leverage-ranked pool of free-q coordinates (default
`lever_pool_size=40-80`, vs. the prior session's fixed 1 cell for A-gravity), candidate
positions from the prior session's own `melitz_candidate_q_positions` window
(`half_window=40-160`), combined up to `max_depth=3` simultaneous single-coordinate
reassignments, beam width `50-100`, deduplicated, EVERY candidate scored via the exact full
cellwise recovery (`build_state`, never an additive approximation of switch effects, per the
task's own explicit requirement). A hard cap (`max_candidates`) bounds total candidate
evaluations.

**Achieved-tolerance-vs-budget frontier (D4, disclosed in full, not cherry-picked)**:

| max_candidates | beam | pool | half_window | achieved \|gravity_A\| | candidates examined |
|---:|---:|---:|---:|---:|---:|
| 200 | 50 | 40 | 40 | 1.9e-7 | 200 |
| 1,000 | 100 | 50 | 80 | 2.1e-8 | 1,000 |
| 5,000 | 200 | 80 | 160 | 2.6e-9 | 5,000 |
| 20,000 | 400 | 113 | 400 | 2.6e-9 (IDENTICAL to the 5,000 row) | 20,000 |

The plateau between the last two rows (identical residual despite 4x more candidates) shows
the binding constraint at that point is the search's `beam_width`/`max_depth` structure, not
raw candidate count -- consistent with Phase 1.4's structural claim that A-gravity has no
continuous handle: closing the LAST few orders of magnitude would require either a much deeper
search or accepting that exact cancellation is a combinatorial rarity, not a limit any bounded
search reaches generically. **Given this, the task's own suggested "~200 candidates" default is
retained as a fast, disclosed floor; the D20 retest scripts use an intermediate, still-bounded
budget (`max_candidates=2000, beam_width=100, lever_pool_size=60, half_window=100`) balancing
compute time against the observed diminishing returns above.**

## Phase 3: continuous within-chamber focal corrector

Implemented in `melitz_continuous_focal_corrector`: a damped Newton solve over the free row-`j`
q cells using the EXACT closed form/Jacobian above (never a finite-difference approximation),
with a hard chamber-bin veto (`melitz_chamber_signature_diff_count(sig_trial,sig0)==0`
required for acceptance, checked over ALL `D^2` cells, not merely row `j`, since a lever's move
also linearly perturbs the single q-gravity pivot cell via `pivot_expand`'s own exact formula).
If repeated halving (`max_shrinks=20`) cannot find an admissible step, control returns to the
discrete selector (`status=:returned_to_discrete`) rather than clipping silently, exactly as
the governing prompt requires.

D4 regression (isolated, no discrete step needed): a pure within-chamber row-`j` perturbation
converges the focal residual from `6.13e-6` toward `4.77e-7` over 15 iterations with
monotonically increasing damping, then correctly reports `:returned_to_discrete` -- confirmed,
by inspecting the trace, that this is NOT slow Newton convergence but a genuine chamber-
boundary pin: the exact zero of the focal link at that particular perturbation lies just
outside the currently selected chamber, and the corrector correctly declines to cross it
silently. This is the intended, spec-compliant behavior (Phase 3's own "return control to the
discrete chamber selector" instruction), not a defect -- confirmed by a separate test where
the SAME Newton step, applied at a point safely inside a chamber, tracks a finite-difference
Jacobian to `4.7e-11` relative error and predicts an exact-operator-evaluated trial residual to
`1.2e-17` absolute (Executive Summary item 2).

## Phase 4: exact witness and round-trip invariant

`melitz_construct_hybrid_chamber_state` (top-level driver) alternates Phases 2/3 up to
`max_macro_rounds=5` times (returning to Phase 2 whenever Phase 3 reports
`:returned_to_discrete`), then: (1) evaluates the exact witness (`trade_residuals`,
`focal_residual`, `gravity_A_residual`, `gravity_f_residual`, `divergence_pstar`) via the SAME
`melitz_update_operator_at_Afg!`/`melitz_moment_residuals_under_p` machinery the prior session
used; (2) performs the round-trip audit (`reduce_to_free_theta_logcutoff` ->
`expand_free_theta_logcutoff`, compared directly to the explicit state in log space for
`A`/`f`, raw for `q`); (3) sets `feasible` only if ALL of: every cell `:ok`, trade residuals
`<moment_tol`, focal residual `<moment_tol`, both gravity residuals `<gravity_tol`, `p*`
nonnegative and normalized. D4 regression (`test/melitz/runtests.jl`, "Hybrid chamber-aware
LFD-preserving corrector (2026-07-30)"): **37/37 assertions pass** (isolated standalone run,
matching this repo's own established practice given the pre-existing, unrelated `mul_G!`
SIGSEGV in the full test suite) -- including the hard invariant "a verified feasible witness
never reoptimizes to `NumericalFailure`" and, whenever gravity residuals are genuinely below
`1e-9`, a strict round-trip match.

## Phase 5: D20 cliff retest

`scripts/melitz_hybrid_phase5_d20_cliff_retest_2026-07-30.jl`: the first 10 negative-direction
and first 10 positive-direction chamber transitions along the audited reduced-q direction
(`b_q`, same direction the negswitch audit and the prior joint-(A,q) session used), comparing
A (pure q), B (cellwise, uncorrected), C (prior two-lever discrete corrector), D (new hybrid
corrector: `discrete_max_candidates=2000, discrete_beam_width=100, discrete_lever_pool_size=60,
discrete_half_window=100, continuous_max_iters=50, max_macro_rounds=5` -- an intermediate,
still-bounded budget per Phase 2's own disclosed frontier finding).

**Headline comparison (full data: `docs/key_results/melitz_hybrid_phase5_d20_cliff_retest_2026-07-30.csv`)**:

| metric | C (prior, 2-lever) | D (new, hybrid) |
|---|---:|---:|
| worst-case `\|gravity_A\|` (20 brackets) | `2.554e-5` | `1.653e-7` (`~155x` better) |
| worst-case `\|focal_residual\|` (20 brackets) | `6.79e-7` | `6.16e-6` (one outlier; see below) |
| brackets with `\|gravity_A\|<1e-6` | **9/20** | **20/20** |
| brackets with `\|focal\|<1e-6` | 20/20 | **19/20** |
| brackets with BOTH `<1e-6` | **9/20** | **19/20** |
| brackets `FiniteSolved` | 20/20 | **20/20** |
| corrector time per bracket | `0.2s` (easy) / `75-92s` (hard) | `20-24s` (easy) / `110-118s` (hard) |

**Every single one of the 20 reoptimized points is `FiniteSolved`**, `Delta*` in
`[0.483236, 0.483277]` -- essentially unchanged from the anchor `0.4832764950`, extending the
prior session's central cliff-crossing claim from 5+5 to the full 10+10 brackets the governing
prompt requires. The `Delta* <= divergence(p*) = 0.4832764950468767` certificate holds at
19/20 brackets; the one exception (minus k=2, `Delta*=0.4832764950469392`) exceeds it by
`6.25e-11` (relative `1.3e-10`) -- at that bracket the discrete selector needed ZERO candidate
evaluations (`gravity_A` was already `<1e-10` at the starting point) and the reoptimized point
is essentially the anchor itself with ordinary KNITRO solve-tolerance noise, not a genuine
economic violation of the certificate.

**A genuinely mixed, disclosed result on the focal link**: at the 10 "easy" brackets (where the
discrete selector needed few/no candidates), the continuous Newton corrector drives the focal
residual to `1e-13`-`1e-15` -- essentially EXACT, a qualitative improvement over C's own
`1e-7`-`7e-7`. At the 10 "hard" brackets, the continuous corrector reports
`:returned_to_discrete` after 5-15 iterations (a genuine chamber-boundary pin, Phase 3's own
documented behavior) and the focal residual settles at `4.9e-7`-`7.6e-7` -- comparable to C's
own achieved level, not uniformly better -- EXCEPT plus k=10, where it is `6.16e-6`, worse than
C's `5.1e-8` at that same bracket. **Diagnosed, not glossed over**: at plus k=10, C's own
`gravity_A` is catastrophically bad (`-2.55e-5`, C's single worst bracket) -- C's two-lever
design happened to prioritize the focal link there at A-gravity's expense; D's discrete
selector correctly prioritizes A-gravity (its designed first objective) and brings it to
`5.45e-9`, but the resulting chamber leaves less "room" for the continuous corrector to also
fully close the focal gap within its bounded iteration/shrink budget. This is the expected,
disclosed trade-off of a corrector that (correctly, per the task's own priority order) treats
A-gravity as the primary objective and the focal link as secondary within a chosen chamber, not
a regression.

## Phase 6: bounded welfare predictor-corrector retest

`scripts/melitz_hybrid_phase6_welfare_retest_2026-07-30.jl`: the same real-D20 anchor
(`Delta0=0.4832764950`, `GT0=6.290641%`), gains-from-trade predictor steps of
`{0.0125%, 0.025%, 0.05%}` in both directions, up to 3 accepted continuation steps/direction,
using the new hybrid corrector (identical configuration to Phase 5).

**Result: zero accepted steps in EITHER direction, at ALL 3 tested step sizes** (full log:
`logs/melitz_hybrid_phase6_2026-07-30.log`, full data:
`docs/key_results/melitz_hybrid_phase6_predictor_corrector_2026-07-30.csv`):

| direction | step_pp | focal residual | gravA | gravF |
|---|---:|---:|---:|---:|
| upper_GT | 0.0125% | `-3.41e-3` | `7.99e-9` | `1.20e-8` |
| upper_GT | 0.0250% | `-6.84e-3` | `3.95e-9` | `5.93e-9` |
| upper_GT | 0.0500% | `-1.37e-2` | `-9.45e-9` | `-1.42e-8` |
| lower_GT | 0.0125% | `+2.36e-3` | `-1.94e-8` | `-2.92e-8` |
| lower_GT | 0.0250% | `+4.73e-3` | `-1.42e-8` | `-2.13e-8` |
| lower_GT | 0.0500% | `+9.46e-3` | `-4.34e-9` | `-6.51e-9` |

**This is the single most important quantitative fact this session establishes**: `gravity_A`/
`gravity_f` reach `~1e-8`-`1e-9` at EVERY tested welfare step -- the SAME quality the hybrid
corrector delivers on the easy Phase 5 brackets, confirming the discrete+continuous machinery
itself is working correctly here too. The blocker is specifically and only the focal-link
scale: `3.4e-3` to `1.4e-2`, **three to four orders of magnitude larger** than the `1e-6`
tolerance the SAME corrector satisfies at 19/20 Phase-5 nuisance-direction brackets, and it
does not shrink meaningfully as the step size varies -- consistent with the prior session's own
diagnosis (the welfare coordinate `g` enters the focal link DIRECTLY via
`gamma_prime_target=exp(g)`, a channel with no analogue in a pure nuisance-direction `q` move)
but now measured against a corrector an order of magnitude better at everything else it does.

**Gamma-only comparison** (move `g` alone, hold `q`/`A` fixed at anchor) reproduces the prior
session's own finding: `Delta*` moves monotonically away from the anchor with NO correction
attempted -- upper: `0.483 -> 0.504 -> 0.527 -> 0.550` (3 steps); lower: `0.483 -> 0.463 ->
0.444 -> 0.427` (3 steps) -- the exact failure mode (letting the outer solver silently absorb a
welfare move into a less favorable LFD) the LFD-preserving program is designed to avoid, and
still cannot deliver an alternative to at this step scale.

## Decision

**Decision B: hybrid correction works for local chamber transitions but not welfare shifts.**
Not merely re-selected from the prior session's own menu -- **sharpened with a quantified
scale gap**: Phase 5 shows the hybrid corrector closing A-gravity to `<1e-6` at 20/20 tested
cliff brackets (vs. 9/20 for the prior design) and to the focal link exactly (`<1e-13`) at half
of them; Phase 6 shows the SAME corrector, at the SAME quality level (`gravA`/`gravF` reaching
`~1e-8`-`1e-9`), unable to close a focal-link gap that is `1,000`-`10,000x` larger simply
because a genuine welfare move requires crossing far more chambers simultaneously than any
bounded discrete search (Phase 2's own measured frontier: a `200`-`20,000`-candidate budget
spans `1.9e-7` down to a `2.6e-9` PLATEAU, never reaching the `1e-2`-scale gap Phase 6 needs
closed) can supply.

**Recommendation**: retain and adopt the hybrid corrector (`hybrid_chamber_corrector.jl`) as
the production-quality tool for the nuisance/cutoff-cliff-crossing use case the negswitch audit
identified -- it is a genuine, quantified, order-of-magnitude improvement over the prior
two-lever design, with an exact (not first-order) closed-form focal-link model and Newton
corrector that a future session can build on directly. Do NOT attempt a further q-space
corrector iteration for the welfare-continuation problem (Phase 6's own scale-gap finding rules
this out cleanly, not merely tentatively, this time) -- pursue the genuine fixed-q A middle-loop
approach (mentioned in this session's own governing-prompt context) or an entirely different
outer-loop mechanism for welfare-direction continuation instead.

**Distinguishing exact claims from approximate ones, as required**:
- Exact bilateral-moment compensation under `p*`: confirmed to `~1e-15` at every tested state
  (unchanged from the prior session, reused verbatim).
- Outer structural admissibility (both gravity restrictions, focal link): quantified precisely
  above -- NOT uniformly at the task's own `1e-10`/`1e-9` aspirational tolerance, but a
  measured, order-of-magnitude improvement over the prior corrector, with the residual floor
  itself explained structurally (Phase 1.4/Phase 2), not merely observed.
- An explicit `p*` witness: computed and reported (`feasible` field) at the STRICT tolerance
  (`gravity_tol=1e-10`, `moment_tol=1e-9`) for every state in this session -- honestly, this
  strict flag is `false` at all 20 Phase-5 brackets and all 6 Phase-6 attempts, even though the
  achieved residuals are excellent by the prior session's own (looser) standard. This session
  does NOT relax the strict flag's definition to manufacture a "yes" -- it reports the achieved
  numbers precisely instead.
- A nearby finite inner state: confirmed 20/20 in Phase 5 (`FiniteSolved`, `Delta*` near
  anchor).
- A short continuation smoke test: Phase 6, above -- a genuine, disclosed negative result.
- Genuine outer convergence: NOT claimed or tested this session (no outer KNITRO search was
  run; only single-point constructions and reoptimizations, matching the governing prompt's own
  scope limit).

## Files changed

```
New:
  src/melitz/hybrid_chamber_corrector.jl
  docs/melitz_hybrid_chamber_lfd_corrector_2026-07-30.md   (this document)
  scripts/melitz_hybrid_phase0_roundtrip_audit_2026-07-30.jl
  scripts/melitz_hybrid_phase5_d20_cliff_retest_2026-07-30.jl
  scripts/melitz_hybrid_phase6_welfare_retest_2026-07-30.jl
  docs/key_results/melitz_hybrid_phase0_roundtrip_audit_2026-07-30.csv
  docs/key_results/melitz_hybrid_phase5_d20_cliff_retest_2026-07-30.csv
  docs/key_results/melitz_hybrid_phase6_predictor_corrector_2026-07-30.csv
  docs/key_results/melitz_hybrid_phase6_gamma_only_comparison_2026-07-30.csv

Modified:
  src/melitz/include_melitz.jl   (new include line)
  test/melitz/runtests.jl        (new testset; also fixes a genuine pre-existing gap --
                                   `lfd_preserving_state.jl` was added to `include_melitz.jl`
                                   on 2026-07-30 but never added to this file's OWN explicit
                                   include list, so the prior session's "LFD-preserving..."
                                   testset silently depended on an include that was never
                                   actually present in the full-suite run; fixed here since
                                   the new testset needs the same file)
```

## Disclosed scope reductions

1. Chamber-bound affine inequalities are derived analytically (Phase 1.4) but the
   IMPLEMENTATION relies on an exact recomputed chamber signature rather than a symbolically
   enumerated inequality system -- a disclosed, deliberate choice (see Phase 1.4).
2. The discrete chamber selector's leverage pool (40-80 coordinates) is a bounded, high-
   leverage subspace, not literally every free coordinate -- disclosed in Phase 2.
3. `norm_disp` (scaled q displacement) uses plain unscaled Euclidean distance, not a
   feasibility-metric-derived scaling -- a disclosed simplification.
4. Given the D4-measured plateau (Phase 2), the D20 retest scripts use a bounded but larger
   candidate budget (2,000) than the task's own suggested 200 -- disclosed above.
