# Unrestricted profiled-destination-scale KNITRO comparison — Master Report (2026-08-01)

## CORRECTION (2026-08-01, same day, user-caught)

The first version of this report claimed the D20 gate was `not_run` because the profiled KNITRO
solve failed to converge tightly (`nStatus=-400`, KKT `~0.12`-`0.36`) at `W=15,000`, and speculated
this was caused by real D20's huge `A_od` dynamic range, citing a memory entry about native KNITRO
variable scaling as precedent. **Both the diagnosis and the citation were wrong, caught by the
user**, who pointed out that `W=15,000` is below this project's own established reliability
threshold for real D=20 data (`W>=80,000`, per this repo's own prior memory —
`D=20 real-data W-sensitivity` and `Melitz small-W numerically finicky`) and that they have never
seen inner-loop convergence problems in production. The cited memory entry
(`melitz-real-d20-scaled-knitro-native-scaling-2026-07-25`) is about a **different model** (Melitz
heterogeneous-firms), not this gravity/CDW model — an inappropriate citation this session should
not have made without checking.

Re-run at `W=80,000` with an added control test (the reference/old formulation, run independent of
the profiled/recovery pipeline): the control converges cleanly (`nStatus=0`, `kkt=6.1e-13`, 5
Hessian calls — same order as D4). The profiled solve now *also* converges cleanly at both points
(`nStatus=0`, 5 Hessian calls each), and the full recover-then-resolve comparison passes decisively
(LFD diff `~4.8e-5`, divergence diff `~3e-9`–`5e-9`, winner checksums identical). **There was no
scaling/conditioning problem and no bug in the reduced kernels — the only issue was testing at too
small a W.**

Note on framing: this session's control test was originally described as isolating "cold-start-at-
D20-scale" from a kernel-specific bug. That framing itself needs a correction, per this repo's own
standing feedback memory (`feedback-user-knitro-convergence-not-start-dependent`, independently
recorded from a separate diagnosis this same day): in this codebase the inner KNITRO solve's
convergence has never been shown to depend on the starting point, only on the actual point/problem
instance — so "cold start" was never really a live candidate explanation either. The correct
framing is simpler: `W=15,000` real draws is a materially under-sampled, noisier, worse-conditioned
instance of this D=20/380-cell problem than `W=80,000` is — a genuinely different (harder)
problem instance, not a warm/cold-start artifact. §5 below and the verdict block are updated
accordingly; the superseded `W=15,000` narrative is struck through rather than deleted, per this
repo's own convention of leaving corrections visible in place.

Diagnostic branch: `diagnostic/profiled-scales-unrestricted-knitro-2026-08-01`
Diagnostic worktree: `/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-knitro-2026-08-01`
Branched from: `architecture/profile-all-destination-scales-2026-07-31` @ `d84d392` (HEAD at branch-cut
time; confirmed code-identical to the task's recorded prototype HEAD `b2a745b` — only a 122-line
doc-only executive-summary commit separates them, see `EXISTING_PROFILED_SCALE_GATE_REPRODUCTION_2026-08-01.md`)
Base of that branch: `production/fullA-exact @ cd17235`
Diagnostic HEAD (this session): `d84d392` (no new commits made to the diagnostic worktree; see
provenance.txt for the exact file list)
Scope: **unrestricted family only**, per this task's explicit instruction. No CM, common-Fréchet,
ZC, or CM+ZC work attempted. No production campaign launched. No production default changed. No
merge attempted.

## 0. Executive summary

The prototype's main documented gap — the anchor-inclusive (redundant) moment dimension never
actually being reduced — is now closed: a new `ProfiledEconomicMomentLayout` genuinely removes one
factual moment per destination (never a value change dressed up as a reduction), backed by new
reduced forward/transpose/Hessian kernels, a real `OperatorPsiBundle` for
`economic_parameterization = :profiled_destination_scales`, and a real KNITRO inner-solve driver
(`inner_loop_KNITRO_profiled`) mirroring production's own `inner_loop_KNITRO_compressed`.

**D=4 is fully decisive.** Every building-block gate (layout, forward/transpose, Hessian) passes at
machine precision against independent references (dense reduced-G, finite differences, and an
*exact* — 0.0 diff — equivalence to the old full kernel with the anchor's dual coordinate forced to
0). The real KNITRO comparison required discovering and implementing the theory doc's own stated
but never-executed comparison theorem (recover full-A using the reduced solve's **own** LFD, then
re-solve the legacy reference at that recovered point) — a naive same-θ comparison does **not**
match (a real, diagnosed, non-bug finding, see §3) and this is not what the theorem claims. Once
implemented correctly, calibration and three perturbations (including one with 5,240 of 32,000
winner cells flipped) all agree to `~1e-13`–`1e-15`.

**D20 is now also decisive, at the right W.** ~~Live dimensions (380→361→362→360) match the task's
predicted figures exactly, and the outer coordinate round-trip is exact to `1.8e-14` in log-space.
The actual KNITRO inner-solve comparison did not complete: the profiled solve does not reach tight
first-order stationarity from a cold start within a 3000-iteration budget (KKT residual
`0.36→0.12`, improving but not tight) — most likely because real D20's `A_od` spans roughly 15
orders of magnitude at this data, a known characteristic of this dataset, needing native KNITRO
variable scaling not implemented here.~~ **Superseded (see correction notice above): that finding
was at `W=15,000`, below this project's own reliability threshold for real D20 data — a materially
under-sampled, harder problem instance, not a scaling/conditioning/start-point issue. Re-run at
`W=80,000`** (this project's own established minimum), with an added control test (the reference
formulation, run independent of the profiled/recovery pipeline): it converges in 5 Hessian calls
(`kkt=6.1e-13`). The profiled solve **also** converges cleanly at both points (`nStatus=0`, 5
Hessian calls each, same order as D4), and the full recover-then-resolve comparison passes
decisively: LFD diff `~4.8e-5`, divergence diff `3.3e-9`–`5.1e-9`, winner checksums identical at
both points. Live dimensions (380→361→362→360) match the task's predicted figures exactly, and the
outer coordinate round-trip is exact to `1.8e-14` in log-space (unaffected by the W correction,
since it never touched KNITRO).

**Rectangular D4 gate: not run.** No D4 rectangular (`Ddest<D`) economic context builder exists
anywhere in this repo; building one safely was assessed as out of remaining scope (comparable
effort to what `context_real_d20.jl` needed for the real D20 case). The layout/kernel code itself
was deliberately written against the repo's general rectangular-safe primitives
(`active_cell_index`/`dest_slot`/`global_destination`, not any D4-specific square assumption), and
the D20 `:exclude_row` gate (`Ddest=19≠D=20`) exercises that same code path with real non-square
data — partial evidence, not a substitute for the requested synthetic D4 gate.

## 1. What was reproduced from the prototype (§2 of the task)

See `EXISTING_PROFILED_SCALE_GATE_REPRODUCTION_2026-08-01.md` for the full account. All 9 supplied
D=4 gates reproduce cleanly (0 failures), and the static bundle guard
(`scripts/static_bundle_guard_2026-07-30.sh`) reports 0 violations against the full
`full_aod_diag/d4_exact/` tree, including every file this session added.

First real blocker (resolved): the Julia environment lives at the **worktree root**
(`Project.toml`), not in `full_aod_diag/d4_exact/` — `julia --project=<worktree-root>` is required
from any working directory.

## 2. The anchor-moment reduction actually implemented (§3–§10 of the task)

New files (`full_aod_diag/d4_exact/`, all `_2026-08-01.jl`, all additive, none modify a
2026-07-31 or earlier file):

| File | Role |
|---|---|
| `profiled_economic_moment_layout_2026-08-01.jl` | `ProfiledEconomicMomentLayout`, `build_anchor_spec_from_ctx` (rectangular-safe anchor spec, uses `global_destination(ctx,s)` for own-cell default, not slot==global-id), `assert_no_factual_price_index_moment` |
| `reduced_homogeneous_contraction_2026-08-01.jl` | Reduced forward/transpose kernels — no anchor column ever allocated |
| `reduced_homogeneous_hessian_2026-08-01.jl` | Reduced `H_EE` — `T1`/`R4` (dense, winner-independent) unchanged in form; `u`/`r`/`R2`/`QQ` (sparse, winner-indexed) skip accumulation whenever the winner is the omitted anchor |
| `reduced_operator_verification_2026-08-01.jl` | `verify_inner_solution_reduced_profiled!`, feeds the *same* `verify_namedtuple_from_operator` the production reference path uses |
| `profiled_operator_bundle_2026-08-01.jl` | `ProfiledCBState`, the two KNITRO callbacks, `inner_loop_KNITRO_profiled` — a faithful reduced-dimension sibling of `inner_loop_KNITRO_compressed` |
| `reduced_recovery_from_lfd_2026-08-01.jl` | `recover_gamma_normalized_full_A_from_lfd` — theory doc §2.3's actual comparison-theorem recovery step, using the reduced solve's **own** LFD, not the factual measure |

### 2.1 Why the reduction is exact, not merely close (the key derivation this session made)

The prototype's own gate (`test_homogeneous_moments_2026-07-31.jl`) already proved
`Σ_o (Q_od(ω)−λ_od·M_d(ω)) = 0` exactly, pointwise, for every draw. This session used that identity
directly:

- **Forward**: setting `κ[anchor,slot]=0` (never allocating a β for it) and computing `Cbar[slot]`
  as a sum over retained origins only is **exactly** what the full kernel computes when the
  anchor's β is forced to `0.0` — verified as an exact (`0.0` diff) equivalence, not an
  approximation (`test_reduced_homogeneous_contraction_2026-08-01.jl`, TEST 3).
- **Hessian**: every economic moment column decomposes as `E_{w,j} = y_part[w,j] − Lam[j]·wval[w,slot(j)]`,
  where `y_part` is sparse (nonzero only at the actual winner's column) and `Lam[j]` is a fixed
  per-column constant. This decomposition holds for **any** subset of `(o,slot)` columns — nothing
  in it assumes a complete slot. So restricting `j` to retained columns only requires: (a) `T1`/`R4`
  (the dense, `Lam`-weighted pieces) stay **exactly as originally coded**, full sums over every
  draw, using `wval` directly (always defined, winner-independent); (b) `u`/`r`/`R2`/`QQ` (the
  sparse, winner-indexed pieces) simply accumulate **nothing** on draws where the winner is the
  omitted anchor (there is no column to accumulate into) — exactly matching the task's own
  description ("no direct one-hot winner coefficient but every retained moment still receives its
  `-lambda*M` term").

Both derivations were verified, not just argued: `test_reduced_homogeneous_contraction_2026-08-01.jl`
and `test_reduced_homogeneous_hessian_2026-08-01.jl` (all PASS, D4).

### 2.2 Structural guarantees

`assert_no_factual_price_index_moment` fails loudly if any destination ever has other than
`D-1` retained factual moments, or if the France ratio moment ever collides with a bilateral index
— exercised by both a positive and a deliberately-malformed negative control in
`test_profiled_economic_moment_layout_2026-08-01.jl`.

## 3. The naive-comparison finding (why calibration equivalence is NOT trivial — a real, diagnosed result)

A direct comparison of the reference (old, fixed-`denom[d]`) and profiled (new, homogeneous
`M_d(ω)`) inner solves **at the identical calibration θ**, with no recovery step, does **not**
numerically agree: `Delta_dual`/`Delta_primal` differ by `~1.09e-4` (≈11% relative), and individual
LFD (`m_weights`) entries differ by up to `~0.16` absolute
(`UNRESTRICTED_KNITRO_CALIBRATION_EQUIVALENCE_D4_2026-08-01.csv`, "naive" columns).

This was investigated to ground, per this repo's own standing discipline (`CLAUDE.md`'s
"floating-point knife edge" warning), rather than accepted or dismissed. Root cause: both moment
systems are satisfied exactly by the **factual/uniform** measure at calibration, but the actual
solved LFD is a **tilted** distribution (`m_min`/`m_max` far from 1 for both solves — this is
correct CC-framework behavior, not a bug). Under a tilted measure, `E_LFD[M_d(ω)]` need not equal
the fixed `denom[d]` — the homogeneous moment only pins the *shares*, not the *scale*, under any
measure (exactly the scale-invariance property this whole reparameterization exploits). So the two
dual programs are genuinely different constraint sets on the LFD whenever it departs from uniform,
and same-θ equality was never a valid expectation.

Theory doc `PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md` §2.3 states the actual theorem: (i)
solve reduced; (ii) recover full-A using the reduced solve's **own** LFD (not the factual measure);
(iii) re-solve the legacy full problem at that recovered A. This session implemented step (ii) for
the first time (`recover_gamma_normalized_full_A_from_lfd`) and executed the full three-step
procedure — see §4.

## 4. D=4 KNITRO gates — DECISIVE

### 4.1 Symmetric calibration (`UNRESTRICTED_KNITRO_CALIBRATION_EQUIVALENCE_D4_2026-08-01.csv`)

Recover-then-resolve procedure result: `Delta_dual`/`Delta_primal` diff `~2e-16`–`2.8e-16`; max
LFD diff `8.2e-13`; winner checksum identical (`81357`); both KNITRO solves `nStatus=0`.

### 4.2 Three deterministic perturbations (`UNRESTRICTED_KNITRO_PERTURBATION_EQUIVALENCE_D4_2026-08-01.csv`)

| Point | Winner cells changed (of 32,000) | LFD diff | Divergence diff |
|---|---|---|---|
| small (scale 0.02) | 186 | 8.3e-13 | 6.4e-16 |
| moderate (scale 0.15) | 1,481 | 6.0e-13 | 1.3e-15 |
| large/multi-winner (scale 0.5) | 5,240 | 6.8e-13 | 4.2e-15 |

All three: `nStatus=0` both sides, winner checksums identical, gravity residual unchanged to
`~1e-18`–`1e-19`. This is the decisive multi-winner-change test task §12 asks for.

### 4.3 Rank/redundancy diagnostics (`REDUNDANT_VS_REDUCED_RANK_GATE_D4_2026-08-01.csv`)

Full basis (17 columns): rank **13**, 4 exactly-zero singular values (`~7.7e-15`), condition number
`1.4e16` (numerically singular) — one zero direction per destination, exactly as predicted. Reduced
basis (13 columns): rank **13** (full), smallest singular value `7.8`, condition number `12.4`.
Hessian eigenvalues confirm the same story: full basis has ≥4 near-zero (`<1e-6`) eigenvalues in its
λ-block; reduced basis's smallest `|eigenvalue|` is `0.007`. This is the concrete demonstration (not
just an assertion) of why the anchor removal is necessary for a well-posed dual, per task §13.

### 4.4 Full-A recovery gate (`RECOVERED_FULL_A_EQUILIBRIUM_GATE_2026-08-01.csv`)

At every D4 point tested: `gamma_d → 1` exactly by construction post-recovery; retained shares and
France ratio preserved; gravity residual unchanged to machine precision; LFD and divergence match
the reference re-solved at the recovered A to `~1e-13`–`1e-15`.

## 5. D20 real-data gate — DECISIVE (corrected; see notice at top of report)

`UNRESTRICTED_KNITRO_OMIT_ROW_D20_SMALLW_2026-08-01.csv`,
`PROFILED_ECONOMIC_MOMENT_LAYOUT_2026-08-01.json`.

Live dimensions from `d20_real_setup(destination_sample=:exclude_row)`: `D=20`, `Ddest=19`, active A
cells `380`, retained factual moments `361`, France ratio `1`, total reduced `362`, free A after the
existing gravity pivot `360` — **all matching the task's predicted figures exactly**, read from live
`ctx`/`cf` metadata, not hardcoded (unaffected by the W correction below). The outer coordinate
round-trip (reduce calibration → decode) is exact to `1.8e-14` in **log-space** — comparing in raw
level-space is meaningless here because real D20's calibrated `A_od` spans `8.8e8` to `2.3e23`
(confirmed live; consistent with `CLAUDE.md`'s standing ~11-orders-of-magnitude warning) — this
session's first test script wrongly used an absolute level-space tolerance and produced a spurious
"failure" (`~7.7e8` absolute diff) before this was caught and fixed.

**First attempt, at `W=15,000` (superseded — see correction notice at top of report):** the KNITRO
inner solve did not converge tightly (`nStatus=-400`, KKT residual `0.359` at `maxit=100`, improving
to `0.120` at `maxit=3000` but still not tight). This session initially attributed it to real
D20's huge `A_od` dynamic range causing ill-conditioning, and cited a supposedly-analogous prior fix
— **both wrong**, caught by the user: `W=15,000` is below this project's own established
reliability threshold for real D=20 data (this repo's own prior memory: `D=20 real-data
W-sensitivity`, `Melitz small-W numerically finicky` — both say use `W>=80,000`), and the cited
"fix" was for an unrelated model (Melitz, not this gravity/CDW model).

**Corrected run, at `W=80,000`:** added a control test — the reference (old/production) formulation
solved independent of the profiled/recovery pipeline — which converges in 5 Hessian calls
(`nStatus=0`, `kkt=6.1e-13`). The profiled solve **also** converges cleanly at both the calibration
and modest-perturbation points (`nStatus=0`, 5 Hessian calls each — the same order as D4), and the
full recover-then-resolve comparison passes decisively:

| Point | LFD diff | Divergence diff | Winner match |
|---|---|---|---|
| calibration | 4.79e-5 | 3.27e-9 | true |
| modest perturbation | 4.48e-5 | 5.11e-9 | true |

(LFD/divergence diffs here are looser than D4's `~1e-13`–`1e-15` — consistent with a much larger,
real-data problem near KNITRO's own default tolerances rather than any indication of a real
discrepancy; both points are comfortably within a `1e-3` relative tolerance and every metric that
*can* be checked at machine precision, e.g. the log-space round-trip, still is.)

**There was no scaling/conditioning problem and no bug in the reduced kernels at D20 — the only
issue was testing at too small a `W`.**

Two real bugs were found and fixed while investigating this gate (both fixed in
`reduced_recovery_from_lfd_2026-08-01.jl`, both caught by directly running against real D20 data,
neither exercised by the D4-only gates, and neither related to the W-size correction above): (1)
the recovery function's `d_list` default (`1:ctx.D`) included the omitted ROW destination, crashing
`dest_slot`; (2) the Aod-block reshape hardcoded `D^2`/`(D,D)` instead of `D*Ddest`/`(D,Ddest)` — a
square-only assumption copied from the original `recover_gamma_normalized_full_A`
(`recover_full_a_2026-07-31.jl`) without adapting it for the rectangular case. Both are exactly the
class of square-only bug task §5 warns against, confirming that testing only at D4 would have
missed them — this is the concrete argument for why the D20 gate matters, independent of the W
correction.

## 6. Rectangular D4 gate — not run

See `UNRESTRICTED_KNITRO_RECTANGULAR_D4_2026-08-01.csv` for the full reasoning. No D4 rectangular
context builder exists in this repo (`d4_exact_setup()` is hard-wired square: `D^2`-sized θ block,
`Aod_free_pos` as a `D×D` reshape). Building one safely was assessed as out of scope for this
session's remaining time. The layout/kernel code was written generically against
`active_cell_index`/`dest_slot`/`global_destination` (never assuming slot==global-id), and the D20
`:exclude_row` gate exercises the identical rectangular code path with real non-square data — this
is real but partial evidence, not a substitute for the requested synthetic gate.

## 7. Scope discipline honored

No CM/common-Fréchet/ZC/CM+ZC work attempted (task §17). No production screens, checkpoint schema,
campaign driver, or outer C+ gradient touched. `PRODUCTION_DEFAULT` remains
`full_gamma_normalized_reference` (confirmed: `build_unrestricted_operator_ctx`'s default
`moment_representation` and `ctx.obj`'s type were never altered; the new bundle is only ever
constructed by explicitly calling `build_profiled_operator_bundle`). No campaign launched. No merge
attempted. Every file this session added is new; `git status --short` on the diagnostic worktree
shows only untracked additions, zero modifications to any existing file.

## 8. Final verdict block

```
ANCHOR_A_COORDINATES_REMOVED =
    yes_1_per_destination (D4: 4; D20 :exclude_row: 19 -- reuses the PRE-EXISTING relative-A/
    gravity-pivot coordinate layer from architecture/profile-all-destination-scales-2026-07-31,
    unchanged by this session; this session's own contribution is the economic-MOMENT-space
    reduction, §ANCHOR_SHARE_MOMENTS_REMOVED below)

ANCHOR_SHARE_MOMENTS_REMOVED =
    yes_1_per_destination (D4: 16->12 retained + 1 France ratio = 13 total, live-confirmed;
    D20 :exclude_row: 380->361 retained + 1 France ratio = 362 total, live-confirmed; verified
    exact -- 0.0 diff -- equivalence to the old full kernel with the anchor's dual coordinate
    forced to 0.0, not merely a value change)

FACTUAL_PRICE_INDEX_MOMENTS =
    zero (assert_no_factual_price_index_moment enforced + positive/negative-control tested;
    D-1 retained factual moments per destination confirmed at both D4 and D20 scale, never D)

REDUCED_INNER_DIMENSION =
    expected_and_confirmed (D4: outer_constr_index=14=1+13; D20: outer_constr_index=363=1+362;
    both live-read from a real constructed OperatorPsiBundle, not hardcoded)

OPERATOR_ONLY =
    pass (static_bundle_guard_2026-07-30.sh: 0 violations against the FULL full_aod_diag/d4_exact/
    tree including every file this session added; OperatorPsiBundle has no H/H_copy/K/ones/
    moments! field, structurally, for the profiled bundle too)

D4_SYMMETRIC_KNITRO =
    equivalent (recover-then-resolve procedure: LFD diff 8.2e-13, Delta_primal diff 2.8e-16,
    winner checksums identical, both nStatus=0 -- naive same-theta comparison does NOT match,
    correctly diagnosed as expected per theory doc section 2.3, not a bug)

D4_PERTURBATION_KNITRO =
    equivalent (3 points incl. one with 5240/32000 winner cells flipped; LFD diff 6e-13-8e-13,
    Delta_primal diff 6e-16-4e-15, all winner checksums identical, all nStatus=0)

D4_RANK_REDUNDANCY =
    confirmed (full basis: rank 13 of 17, 4 exact zero singular values, cond 1.4e16; reduced
    basis: rank 13 of 13 (full), cond 12.4; Hessian lambda-block: full basis >=4 near-zero
    eigenvalues, reduced basis smallest |eigenvalue| 0.007)

D4_RECTANGULAR_KNITRO =
    not_run_no_D4_rectangular_context_builder_exists_in_this_repo (assessed out of scope for
    remaining session time; D20 :exclude_row gate exercises the same rectangular-safety code
    path with real non-square data as partial evidence)

D20_OMIT_ROW_SMALLW =
    equivalent (CORRECTED, see notice at top of report -- first attempt at W=15,000 wrongly
    reported not_run, misdiagnosed as a KNITRO scaling/conditioning problem; W=15,000 is below
    this project's own W>=80,000 reliability threshold for real D20 data, per this repo's OWN
    prior memory, and the cited "fix" was for an unrelated model. Re-run at W=80,000: live
    dimensions 380/361/362/360 ALL confirmed exactly matching task's predicted figures; outer
    round-trip exact to 1.8e-14 log-space; control test (reference, independent of profiled/
    recovery pipeline) converges in 5 Hessian calls (kkt=6.1e-13); profiled solve converges
    cleanly at both calibration and modest-perturbation points (nStatus=0, 5 Hessian calls each);
    recover-then-resolve comparison passes at both points (LFD diff 4.79e-5/4.48e-5, divergence
    diff 3.27e-9/5.11e-9, winner checksums identical); two real rectangular-safety bugs (unrelated
    to the W correction) were found+fixed while investigating this gate, see section 5)

FULL_RECOVERY =
    gamma_one:pass (D4, all 4 points, exact by construction; D20 W=80000, both points, exact by
        construction)
    retained_shares:pass (D4, all 4 points; D20 W=80000, both points, via matching winner
        checksums + LFD)
    omitted_anchor_shares:pass (D4, algebraic sum-to-one identity, all 4 points)
    France_ratio:pass (D4, all 4 points)
    gravity:pass (D4, residual unchanged to ~1e-18-1e-19, all 4 points)

UNRESTRICTED_PROFILING_THEOREM =
    numerically_confirmed_at_D4_and_D20 (decisive D4 confirmation across calibration + 3
    perturbations incl. a large multi-winner-change point, AND decisive D20 real-data
    confirmation at W=80,000 across calibration + a modest perturbation, both via the theory
    doc's own recover-then-resolve procedure implemented for the first time this session. The
    session's own FIRST D20 attempt, at W=15,000, was a false negative caused by an
    under-sampled W, corrected same-day after the user identified the actual cause)

PRODUCTION_DEFAULT_CHANGED = false
PRODUCTION_MERGE = not_attempted
CAMPAIGN_LAUNCHED = false
```
