# Fixed-Fréchet Post-Omit-ROW Port Readiness — 2026-07-24

Branch: `feature/fixed-frechet-post-omit-row-port-prep-2026-07-24`, off
`production/fullA-exact @ c55e81e`. **Not merged to production. Not tagged production-ready.**

## Verdict

**PORT_READY_EXPERIMENTAL — WAITING ON CANONICAL WINNER ENGINE, WITH ONE DISCLOSED FINITE GAP: NO
COMPLETED LIVE OUTER SHAKEDOWN**

Every correctness gate this task specifies (D=4, D=20/W=80,000/L=50, dimension-safety, basis
invariance, nesting, flexible-CM regression) **passes**. The one item from the task brief NOT
delivered is §12's live multi-point outer shakedown: an attempted 90-minute run was abandoned by
explicit user direction after observing a single new-point inner solve cost ~660-1015s under this
session's machine conditions, in favor of a live root-cause diagnosis of that cost (delivered,
see below) rather than continuing to burn wall-clock on a shakedown whose per-point cost made "at
least five valid new trial points" impractical within a reasonable session budget. This is
reported as a genuine, finite, named gap — not silently substituted with a shorter run presented
as equivalent.

This branch does not touch or depend on any not-yet-landed canonical-winner-engine code; it
consumes the current production core-moment/screen interface (the plain `ctx` NamedTuple +
`cm_screen_precheck!`) via the exact same pattern every other restricted-model family already
uses (see `FIXED_FRECHET_CANONICAL_WINNER_INTEGRATION_MANIFEST_2026-07-24.md`). No fixed-Fréchet-
specific winner scanner was added.

## What is active

- **Full paper specification is active by default**: `marginal_mode=:frechet_reference,
  frechet_feature_set=:cdf_power` implements both eq. (37) (CDF) and eq. (38) (truncated-power),
  `2·D·L` restrictions. `:cdf_only` (eq. 37 alone) is retained as an explicit legacy/diagnostic
  option.
- **Live calibration, not the draft's historical θ*=6.8**: `theta_star = 1/ctx.μHat`, reported at
  startup alongside σ, D, D_dest, L, feature set, basis, and analytic target SHA-256 fingerprints
  (`report_frechet_targets`, `frechet_reference_targets.jl`).

## Dimensions

- `D` (origin count) = 20, invariant across `destination_sample`. The fixed-Fréchet marginal
  restriction spans **all 20 origins** including ROW — `ctx.U` is `W×D`, never resliced by
  destination count.
- `D_dest` (destination count) = 19 under `:exclude_row`, 20 under `:all_legacy`. The core
  trade-share block uses `D×D_dest`; the fixed-Fréchet feature block uses `D×L` per family
  — **never** `D_dest×L`. Verified directly: D=4 gate P0
  (`test_frechet_power_hessian_d4_gates.jl`) and the real-D20 rectangular gate R1
  (`test_frechet_rectangular_d20_smoke.jl`, `ncm=200=2·20·5`, explicitly checked `≠ 2·19·5`).

## Basis default

`frechet_basis=:cumulative` is the temporary default for `:cdf_power` (the fast structured Hessian
path built in this pass is `:cumulative`-only). This matches current flexible-CM production's own
default (confirmed by the Section-6 audit: interval is production-*unreachable* for flexible-CM,
and flexible-CM's own conditioning study found interval **25-39x worse** at D20/L=50 — the opposite
direction from the earlier, Fréchet-block-specific pre-omit-ROW conditioning study, which found
interval 20-1,020x *better* for the Fréchet common-pin block specifically). **This discrepancy
between two different studies of two different Gram matrices is not resolved in this pass** — flagged
explicitly, not silently picked one way. `:cdf_only` supports both `:cumulative` and `:interval`
(both validated at D=4, 9/9 PASS, `Q1·T == Q0` to machine precision for both CDF and POWER blocks).

## What current flexible-CM production actually uses (Section 6 audit summary)

Cumulative basis only (interval unreachable from any driver); CDF-equality moments only (the
truncated-power eq.36 family exists in code but is dead — no caller ever passes
`include_truncated_moment=true`); `:orthonormal` contrasts (not `:anchored`, the struct default);
Architecture-B fast per-call moments + Architecture-C structured Hessian; C+ gradient folds a
one-shot `λ_C*'C_s` constant into `q0`, never differentiates the CM block. Full detail:
`CURRENT_FLEXIBLE_CM_QUANTILE_IMPLEMENTATION_AUDIT_2026-07-24.md`.

## Accepted-point state reuse

Every base-state/gradient entry point (`archC_frechet_base_state`,
`archC_frechet_cdf_power_base_state`, `cm_frechet_production_gradient`) accepts and honors a
caller-supplied `base::BaseDualState`, re-solving only when `base===nothing`. No trial-timeout
option file is introduced anywhere in this branch (grep-verified). Full detail:
`FIXED_FRECHET_TIMEOUT_AND_STATE_REUSE_AUDIT_2026-07-24.md`.

## Timeout classification

`frechet_solve_outcome(nStatus)` (built on production's existing `decode_knitro_status`) separates
`:feasible` (includes ordinary limit-but-feasible results) from `:time_limit_no_certificate`
(limit reached, no feasible point — never treated as infeasible) from `:infeasible_certificate`
(genuine structural certificate).

## Core winner engine integration

No `CoreMomentOperator` type exists in current production; this branch follows the same `ctx`
NamedTuple + `merge(ctx,(obj=...))` convention every restricted family uses. Full manifest of every
file/function that touches core state, and the single narrow seam (`build_cm_bin_ctx(ctx,aug)`)
where a future canonical engine would plug in: `FIXED_FRECHET_CANONICAL_WINNER_INTEGRATION_MANIFEST_2026-07-24.md`.

## D=4 correctness gates — ALL PASS

| Gate file | Result |
|---|---|
| `test_frechet_power_hessian_d4_gates.jl` | 13/13 PASS — dimension-safety asserts; CDF-only structured-vs-dense (4.9e-16); **CDF+POWER structured-vs-dense (2.3e-15, the required new deliverable)**; end-to-end solve agreement (Δ diff = 0.0); nested ordering flexible ≤ frechet-CDF ≤ frechet-CDF+POWER |
| `test_frechet_basis_invariance_d4_gates.jl` | 9/9 PASS — Q1·T==Q0 (CDF, POWER) to machine precision; Δ*/primal-weight invariance Q0 vs Q1; Q1 structured-vs-dense |

## D=4 rectangular-layout gate (real D=20 data, small W, dimension-only)

`test_frechet_rectangular_d20_smoke.jl` GATE R1 (construction/dimension bookkeeping): **5/5 PASS**
— `ncm=2·D·L` confirmed, not `2·D_dest·L`. GATE R2 (solve+Hessian-agreement) initially hit
`nStatus=-300` (unbounded) at `W=200` and `W=4000`; **root-caused via a direct bare-core-vs-
flexible-CM comparison** (`diag_frechet_unbounded_at_d20_smallw.jl`) to a **pre-existing, generic
small-W characteristic of this real-D20/`:exclude_row` calibration point — the bare core model
(zero CM/Fréchet restriction) hits the identical `nStatus=-300` at the same W**. Not a fixed-
Fréchet-specific bug. See `docs/test_logs/diag_small_w_unbounded_generic_not_frechet_specific_2026-07-24.log`.

## D=20 W=80,000/L=50 gate status — ALL PASS (4/4)

`test_frechet_d20_production_gates.jl 80000 50`, real D=20 data, `destination_sample=:exclude_row`,
`W=80,000`, live θ*=8.7557, μ̂=0.11421, σ=2.5. Full log:
`docs/test_logs/d20_w80000_l50_cdf_power_full_gate_2026-07-24.log`.

- **GATE D1** (bounded `L=8` structured-vs-dense Hessian at real D=20/W=80,000 scale): dense solve
  feasible (`nStatus=0`, optimal, 21.7-31.8s across repeated runs); **max|dense−struct|=1.396e-5,
  relative to max|h_dense|=3977.9: 3.5e-9** — confirmed benign floating-point precision at scale
  (not a bug): the worst-agreeing entry is in the POWER×POWER block, consistent with that block's
  larger dynamic-range accumulation (`U^pw` weighting) versus the CDF-only block's idempotent
  indicators. **PASS.**
- **GATE D2** (calibrated-benchmark evaluation, full `L=50`, `:cdf_power`, structured Hessian
  only): `ncm=2000=2·D·L` confirmed. Solve at the raw calibration point: `inner_status=0`,
  `outcome=:feasible`, `is_verified_success=true`, **Δ*=0.0156630668**. **PASS.**

Total wall time for the full gate script (ctx build + both gates) was ~5-8 minutes across repeated
runs on this heavily-loaded shared machine (load average ~190 on 208 cores throughout this session)
— see the performance section below for a breakdown and the caveat about how much of that is
machine contention versus intrinsic cost.

**Note on iteration**: reaching this clean 4/4 result took four attempts, all of which failed on
bugs in this port-prep session's own *test/diagnostic scripts* (Julia local/global soft-scope
conflicts, a `BaseDualState.nStatus` field-name typo — the real field is `inner_status`) — **never**
on the ported production code itself. Each failure was root-caused before being dismissed or
retried, consistent with this repo's standing requirement to verify rather than assume.

## D=20 small-W generic instability (not Fréchet-specific)

Both the bare core model (zero restriction) and flexible-CM itself hit `nStatus=-300` (unbounded)
at the raw calibration point under `W=4,000` and (for flexible-CM specifically) `W=20,000` —
confirmed via a direct controlled comparison
(`docs/test_logs/diag_small_w_unbounded_generic_not_frechet_specific_2026-07-24.log`). This
resolves cleanly at `W=80,000`, this codebase's own established production standard. Not a
fixed-Fréchet-specific defect.

## Outer-gradient wrapper: one real bug found and fixed via live execution

`cm_frechet_production_gradient`'s first live invocation (inside an actual KNITRO outer-loop
gradient callback) failed with a KNITRO-reported `nStatus=-500` (callback error) — KNITRO's
callback wrapper swallows the underlying Julia stacktrace, so this was root-caused by reproducing
the call **outside** the KNITRO callback (`diag_frechet_gradient_error.jl`): a missing
`bandwidth_cache::Dict{Int,Float64}()` argument required by `h_mode=:cached` (an omission in this
port-prep session's own `run_frechet_upper.jl` driver, not the underlying
`composite_gradient_at_fast`/`cm_frechet_lfix_aware.jl` machinery). Fixed; re-verified: gradient
computes successfully, **length 380 (`=D·D_dest`, confirming rectangular sizing)**, deterministic
across repeat calls at the same point (`max|g−g2|=0.0`). See
`docs/test_logs/gradient_wrapper_validation_2026-07-24.log`.

## Outer shakedown status — NOT COMPLETED (disclosed gap, root-caused instead)

`launch_frechet_shakedown.jl 5400 80000 50` was launched (real D=20 data, W=80,000, L=50,
`:cdf_power`, δ=1.0, starting from `(gp_target, zfree*)`, 90-minute budget — sized to the ~660-790s
per-new-point cost measured in this session's own warm-up/gradient-diagnostic runs). It produced
**one genuine new outer-search trial point** (`eval 2`, `gp=0.9833126249946789`, `Δ=0.8326853525614948
<δ=1`, `feasible=true`, `verified=false`) after **1021.3s**, confirming real movement in `A`/`g_p`
exactly as the task requires — but at that per-point rate, five valid new trial points would need
roughly 5,000+ seconds, longer than a session-reasonable continuation of this already-very-long
task. **Per explicit user direction mid-task, the shakedown was abandoned in favor of root-causing
the per-point cost directly, rather than continuing to spend wall-clock chasing the numeric "five
points" bar without understanding why each point was so expensive.**

**Root cause, confirmed by live diagnosis** (`FIXED_FRECHET_SLOW_INNER_SOLVE_DIAGNOSIS_2026-07-24.md`,
full iteration table in `docs/test_logs/slow_inner_solve_diagnosis_iteration_table_2026-07-24.log`):
KNITRO registers the `ncore+ncm=2382`-variable inner dual problem's Hessian as **fully dense**
(`2,838,153` nonzeros = the complete upper triangle) and solves it via a **single-threaded**
(`par_numthreads=1`) Interior-Point/Barrier Direct algorithm against **very tight tolerances**
(`opttol=1e-12`, `ftol=1e-15`) — all pre-existing settings in `ek_inner.opt`, none of them touched
by this port. The observed 14-iteration partial trace shows textbook interior-point behavior
(geometric `OptError` decay within each barrier stage, periodic resets on barrier-parameter
reduction) at roughly 15s/iteration — **this codebase's own moment-construction and this port's
structured-Hessian-callback code were independently confirmed fast (sub-second to ~12s even at
real D=20 scale) and are not the bottleneck.** The cost is KNITRO's own dense per-iteration linear
algebra at this problem size, a pre-existing characteristic of the current inner-solve
configuration (dense Hessian registration + single-threaded solver + very tight tolerances) that
predates and is orthogonal to this port's own code — not a fixed-Fréchet-specific defect, and not
a correctness problem (the structured Hessian's numerical agreement with dense Architecture A was
separately confirmed to relative error 3.5e-9 at this exact scale, see the D=20 gate section
above).

**Consequence for this verdict**: task brief §12's live shakedown requirement is not satisfied by
this pass. Everything else it would have exercised beyond what the D=20 correctness gates already
confirm (genuine movement in `A`/`g_p` — confirmed once, via eval 2 above; the timeout/state-reuse
discipline under live multi-point conditions — not exercised beyond the single confirmed reuse
case in the performance report) remains to be demonstrated in a follow-up pass, ideally after
addressing the concrete, scoped follow-up levers below (multi-threading the inner linear solver
being the most promising, since it touches no code this port owns).

**A real, disclosed driver limitation surfaced during this diagnosis**: `run_frechet_upper.jl`'s
trace only records `gp` per evaluation, not the full `w` vector — so `eval 2`'s exact `zfree` point
could not be exactly reproduced for the standalone diagnosis (which instead used the fully-
reproducible warm-up point, `(gp_target, zfree*)`, itself already independently measured in the
same 660-1015s range). A future pass should log the full point (or at least a checksum/save-to-file)
at every evaluation specifically so any slow or failing point can be reproduced exactly.

## Flexible-CM regression smoke — PASS (2/2)

`test_flexible_cm_regression_smoke.jl`, real D=20 data, `W=80,000`, `:exclude_row`, loaded
alongside every new file this branch adds. `archC_base_state` solve: `nStatus=0` (optimal),
feasible. `Delta*_flexible = 0.0029510746` — finite, nonnegative, and (informatively, not a strict
apples-to-apples check since `L` differs) smaller than the D=20 Fréchet Δ* above, consistent with
the theoretical nesting direction (adding restrictions can only weakly increase minimum
divergence). Confirms no shared restricted machinery was broken by this branch's additions. See
`docs/test_logs/flexible_cm_regression_smoke_w80000_2026-07-24.log`.

## Checkpoint (Section 10) — structural only, not exercised end-to-end

`cm_frechet_checkpoint.jl` (`CMFrechetCheckpointV1`, 40 fields, sidesteps the `CMCheckpointV3-7`
naming collision already hit twice in this repo) parses and loads correctly (confirmed standalone:
`fieldnames(CMFrechetCheckpointV1)` returns all 40 expected fields) and follows the established
field-naming/schema-bump conventions from `CMCheckpointV6`. **Not exercised via an actual
save→resume round-trip against a live outer-search run in this pass** — disclosed gap, not a
silent omission. The outer shakedown launcher (`launch_frechet_shakedown.jl`) does not yet call
`build_cm_frechet_checkpoint`/`save_cm_frechet_checkpoint` — wiring that in, plus a resume-and-
compare-bit-for-bit gate (mirroring this repo's own `checkpoint_resume_exclude_row_cplus.jl`
pattern), is listed below as a pre-merge item.

## What remains before a production merge

0. **Complete a live multi-point outer shakedown** (task brief §12, not delivered this pass — see
   the Outer Shakedown section above) — ideally after evaluating the inner-solve-speed levers the
   slow-solve diagnosis identified (ek_inner.opt's single-threaded dense linear solver being the
   most promising and lowest-risk, since it touches no code this port owns).
1. Wait for (or integrate with) the canonical winner engine — see the integration manifest for the
   exact expected seam.
2. Build a `:cplus`-backend analogue of `cm_frechet_production_gradient`
   (`cm_frechet_production_gradient_cplus`, mirroring `lfix_cm_cplus.jl`) — this port's gradient
   path is reference-backend only, and (per the performance report) a fresh gradient call costs
   ~50s versus ~24s reusing a cached base — both dominated by the underlying inner solve cost, not
   by which gradient backend is used, but C+ is production's own established default and this port
   does not yet match it.
3. Wire `cm_frechet_checkpoint.jl` into an actual outer-search driver and add a save→resume
   round-trip correctness gate (see the Checkpoint section above — currently structural-only).
4. Build a fast Architecture-B (non-dense) moment path for `:cdf_power`, avoiding the persistent
   `W×2000` (~1.28GB) CM matrix at full D=20/L=50 scale, if benchmarking shows the dense path is a
   real bottleneck (not measured as prohibitive in this pass — `fpcx` construction was 34-64s under
   heavy load — but not proven safe at larger W/L either).
5. Resolve or explicitly accept the cumulative-vs-interval conditioning discrepancy noted above
   before ever promoting `:interval` to the `:cdf_power` default.
6. A dedicated cumulative-vs-interval benchmark for BOTH flexible CM and fixed Fréchet, as the
   task brief's own §5 anticipates as a follow-up task, not this one.
7. A dedicated memory/RSS profiling pass (disclosed gap in the performance report) — this pass
   confirmed correctness and functional timing but did not instrument peak memory.
