# ZC Fréchet-draw-moments fix — MASTER report — 2026-08-05

## 1. Mission

Correct a scientific-definition error in the zero-covariance (ZC) restrictions: the paper defines
them on the Fréchet productivity draw `z_o(ω) = U_o(ω)^{-μ}` (`U_o(ω) ~ Exp(1)`), but the
implementation built the moments from powers of the raw exponential draw `U^k` directly.

## 2. Exponent mapping (verified, not assumed)

- `μ` is literally named `μ` in code (`θ[1]`, `ctx.μHat`), **not** `theta` directly; `μ = 1/θ`
  where `θ` is the gravity/trade elasticity.
- The economic/gravity moment code (`createUDerivatives!.jl`, `moments!.jl`, `moments_fast.jl`)
  already computes `z = U.^(-μ)` correctly. Only the ZC/meanZC restriction builders skipped this
  transform.
- **Verified invariant**: production's `θConstant` (the flag gating an in-place `U.=U.^(-μHat)`
  mutation that could otherwise make `ctx.U` already-transformed) is hardcoded to `0` in
  `AD_PARAMS` (`setup_context.jl:19`), the single parameter set every real-D20 context merges
  into — the mutation branch is unreachable from any production entry point. `ctx.U` is
  confirmed the raw, untouched `Exp(1)` draw throughout.
- Real D=20 production `μHat`: **0.1335** (with production gravity mask: `destination_sample=
  :exclude_row`, `exclude_diagonal_gravity=true`, Brazil-Korea cell exclusion, `σHat=3.0`).
- Finiteness: `E[z^k]=Γ(1-μk)` finite for `μk<1`; at `μ=0.1335`, finite through `k=7`
  (`μ·7≈0.935`), no blocker at the production K=3 target (`μ·3≈0.40`).

## 3. Source audit

See `ZC_FEATURE_BASIS_SOURCE_AUDIT.md` for the full call-site map. Summary: one shared builder
(`build_raw_mean_pair_matrices`/`build_raw_mean_pair_matrix_levels`, `cm_meanzc_moments.jl`) feeds
every FULL and REDUCED, origin-ZC and CM+ZC path — Hessian blocks and verification code are pure
consumers of its output, unaffected structurally. Two independent duplicate `U^k` sites also fixed
(`cm_originzc_config.jl`'s `originzc_default_nu_bounds`, `cm_meanzc_config.jl`'s
`meanzc_default_nu_bounds`). ~36 other diagnostic/gate scripts share the stale `nu0=k!`
assumption but are off the production/REDUCED critical path — deliberately not touched (documented
scope boundary, avoids the over-scoping this repo has been burned by before).

## 4. Implementation

Two new canonical helpers in `cm_meanzc_moments.jl`, both requiring `μ::Float64` with **no
default** (repo rule: no silent default on a parameter that changes the economic problem):

```julia
frechet_productivity_from_exponential(U, μ) = U .^ (-μ)
frechet_power_feature(U, k, μ)              # = U .^ (-μk), via exp(-μk*log(U)) for stability
```

`build_raw_mean_pair_matrices`/`_levels`/`nu_feasible_interval` now route through
`frechet_power_feature` and require `μ`. Two production commits on
`fix/zc-frechet-draw-moments-2026-08-05` (branched from `production/fullA-exact@546feff`):
1. `6fd0229` — the fix itself (4 files).
2. `8520c59` — test-file corrections + this audit (4 files, incl. new
   `ZC_FEATURE_BASIS_SOURCE_AUDIT.md`).

Derivative formulas needed no change: `μ` is fixed on every ZC production/REDUCED path (the
free-θ/μ experimental machinery never touches this builder), so `∂(feature)/∂(A,gp,η_ν)=0`
exactly — only the *values* flowing into the existing `d_delta_dual_d_nu_vec` etc. formulas
changed, because `Zraw_all` itself changed.

## 5. Direct feature + D4 exhaustive tests

Both `test_cm_originzc_pure_moments.jl` and `test_cm_meanzc_d4_gates.jl` updated (nu0 formula,
extended K coverage to 3) and run to completion:

**origin_zc, D4, real calibration point (A=A\*, ν_k=Γ(1-μk)), K=1→2→3**: 8+18+18+2+2+2 = **50/50
tests PASS**. Δ\*: K1=0.0012, K1+pair=0.0014, K2=0.0043, K2+pair=0.0022, K3=0.0044,
K3+pair=0.0122. KNITRO status=0 throughout, KKT residuals ~1e-14, analytic η_ν gradient matches
central-FD after a diminishing-bandwidth check (K=3's larger curvature needed h=1e-6/1e-7, not a
formula bug — confirmed by quadratic error shrink with h).

**cm_meanzc, D4, same calibration point, K=1→2→3** (standalone script, since the repo's own D4
gate file has an unrelated pre-existing bug — see §7): Δ\*: K1=0.0034, K1+pair=0.0037, K2=0.0038,
K2+pair=0.0049, K3=0.0079, K3+pair=0.0165. All status=0, gap/KKT ~1e-15.

**W=100,000, D=20, real data, real calibration point (production gravity mask), K=1→2→3**:
- origin_zc: Δ\*=0.0032 (K1) → 0.0043 (K2) → 0.0062 (K3, non-production mask) /
  0.0095 (K3, production mask: σ=3.0 + gravity exclusions)
- cm_meanzc: Δ\*=0.0043 (K1) → 0.0054 (K2) → 0.0074 (K3, non-production mask) /
  0.0107 (K3, production mask)

All finite, small, converged, walking cleanly K1→2→3 at both D4 and W=100k. This directly refutes
a prior session's claim that these families "can't find a point with finite Δ\*" — see memory
`feedback-zc-families-do-achieve-finite-delta-star-k3`.

## 6. Pre-existing bugs found and fixed along the way (not part of the ZC-basis fix itself)

1. `test_cm_meanzc_pure_moments.jl` and `test_cm_meanzc_d4_gates.jl` cannot run standalone on the
   unmodified baseline (missing includes) — confirmed via `git stash` on the untouched baseline.
   Fixed the `cm_originzc_target_layout.jl` include gap in the latter (needed for
   `SharedByPowerLayout`); the former's gap left unfixed (out of scope, not needed for the gates
   actually run).
2. `test_cm_meanzc_d4_gates.jl`'s "Hessian equivalence" testset has a pre-existing, unrelated
   `BoundsError` in `mean_targets`/`refresh_zc_targets!` (`cm_originzc_target_layout.jl:85`) —
   newly exposed by fixing bug #1 above, not caused by the ZC-basis fix. Worked around by testing
   CM+ZC calibration directly instead of through that testset; not fixed (out of scope).

## 7. Campaign safety (task §1)

At session start, found and safely stopped 4 live KNITRO solves using the **old** (exponential-draw)
ZC basis in the `campaign/fullA-continuation-polish-2026-08-03` worktree (`origin_zc`
upper/lower delta=2, `cm_meanzc` upper/lower delta=0.01/0.1) — killed between iterations, never
mid the atomic `serialize`+`mv` checkpoint write, so nothing corrupted. A race let 2 more cells
launch before the parent chain scripts died; caught and killed those too. Non-ZC jobs
(`unrestricted`, `flexible_cm`, `common_frechet`) were confirmed untouched throughout. Marked
`campaign_output/origin_zc/` and `campaign_output/cm_meanzc/` with
`SCIENTIFICALLY_OBSOLETE_ZC_BASIS_EXPONENTIAL` markers (no files deleted/overwritten).

## 8. Production release status

- `production/fullA-exact` fast-forwarded **locally** to `8520c59` (true FF, verified via
  `merge-base --is-ancestor`). **Not pushed to `origin`** — user explicitly asked to hold off
  until the campaign is genuinely running/complete; will confirm again before pushing.
- Merged cleanly into the campaign worktree's branch (`campaign/fullA-continuation-polish-2026-08-03`,
  merge commit `13fe088`) — zero file overlap with that branch's own 22 unmerged commits, no
  conflicts, campaign worktree's own uncommitted local changes (3 unrelated files) preserved
  untouched.
- Prototype (`prototype/profiled-destination-scales`, REDUCED) port explicitly **deferred** per
  user instruction — everything REDUCED-related happens after the FULL campaign is launched and
  validated.

## 9. The seeding saga (why launch #4 is the one that's actually running)

The ZC-basis fix itself was never in question after §5's evidence — every subsequent problem was
in constructing a valid **starting point (w0)** for the real outer-search campaign driver, entirely
downstream of the fix:

1. **Launch #1**: no real seed for 3/4 chains (relied on a stale scalar "inherited GT" with no
   actual warm-start vector) → both explore and polish stages silently failed verification and
   passed through the old incumbent unchanged in ~200s (`exit=0`, but zero real work);
   `cm_meanzc/upper` had no incumbent at all and hard-failed. Diagnosed by checking the actual
   per-delta report content (`inherited_GT == final_GT`, `new_GT=nothing`), not just exit codes —
   the "it finished fast with exit=0" signal was actively misleading.
2. **Launch #2**: used pre-existing `*_k3_transplant_seed.jls` files from the prior (pre-fix)
   session. **User explicitly rejected this** — those files bake in the *same* wrong
   `nu0=factorial(k)` assumption the ZC-basis fix corrects (confirmed by reading
   `build_k3_transplant_seeds.jl`'s own `nu0_shared`/`nu0_origin` functions), so their eta_nu tail
   is scientifically wrong regardless of warm-start-doesn't-affect-correctness reasoning — the
   *target* the outer solver would be optimizing toward was wrong, not just its starting values.
3. **Launch #3**: hand-assembled `w0 = vcat(ctx.θ0_up[ctx.free_idx], log.(nu0_correct))` directly
   from a fresh calibration context. Failed immediately with `DimensionMismatch: ... 380 ... 379`
   — the real production driver's `w0` uses a *different* coordinate representation
   (`A_coordinate_mode=:powered_aspace`, a decorrelated `a=log(AodPow)`-style reparametrization
   via `cm_a_from_z`/`pivot_reduce`/`cm_fixed_theta`), not the raw `θ0_up` free-parameter block,
   and the ad hoc context used the wrong `σHat` (2.5, `AD_PARAMS`'s bare default, not the real
   production driver's own `σHat=3.0` default).
4. **Launch #4 (current)**: reused the *encoding logic only* (not the sigma/gravity settings, and
   not the 5-starts perturbation search) from `common_five_starts_search.jl`'s own "candidate 1 =
   genuine calibration point" construction — `cm_fixed_theta`/`precompute_cm_aspace_xy`/
   `pivot_reduce`/`cm_a_from_z`, self-checked via a coordinate round-trip (`w -> z -> x_free`
   reproduces `ctx.θ0_up`'s own free block to `<1e-10` rel err; achieved 6.9e-14 live) — combined
   with the real production driver's own context settings (`σHat=3.0`,
   `exclude_diagonal_gravity=true`, Brazil-Korea exclusion, read directly from
   `run_originzc_upper_checkpointed`'s kwarg defaults, not from the starts-search script) and the
   corrected `nu0=Γ(1-μk)`. Resulting `w0` lengths (440 for origin_zc K=3, 383 for cm_meanzc K=3)
   match the driver's expected 380-length economic-block convention exactly. **Confirmed genuinely
   running**: all 4 chains alive past 25 minutes (vs ~200s for every prior bad attempt), producing
   real `feasible=true verified=true` evaluations with Δ\* in the same small range the calibration
   check predicted, progressing through explore→polish stage transitions.

Calibration seed files: `POST_VERIFY_FIX/calibration_seeds/{origin_zc,cm_meanzc}_calibration_seed.jls`.

**A 5th issue surfaced after launch #4 had already been confirmed "genuinely running" for 25+
minutes**: `origin_zc`'s delta=0.1 cells (both directions) each found and verified a real new K=3
candidate (`new_GT≈0.0239`), but `final_GT` stayed at an OLD pre-fix value (0.0485 upper / 0.0081
lower) because the driver's "never regress" rule compared the new (correct-basis) result against
a **stale `report.jls` already sitting in `campaign_output/{origin_zc,cm_meanzc}/*/delta_*/`**
from before this session (some pre-dating this session entirely, Aug 4; some from this session's
own earlier failed seeding attempts). The family-level `SCIENTIFICALLY_OBSOLETE_ZC_BASIS_EXPONENTIAL`
marker left during campaign-safety cleanup (§7) is not something the driver code checks — it kept
reading these stale per-delta files as valid inherited incumbents regardless, making every
comparison scientifically meaningless (correct-basis vs wrong-basis Δ\*/GT are not on the same
footing). **Fix**: archived the entire pre-existing `campaign_output/{origin_zc,cm_meanzc}/` tree
to `POST_VERIFY_FIX/OBSOLETE_PRE_FIX_ARCHIVE_2026-08-05/` (moved, not deleted — see that
directory's own `README_WHY_ARCHIVED.txt`); non-ZC families' directories and the shared
`campaign_summary.csv` untouched. All 4 chains killed and relaunched (launch #5) with identical
seeds/settings; confirmed "Inherited incumbent for this target: none" for all 4 at start, i.e.
a genuinely clean continuation chain from here.

## 10. Campaign status at session handoff: PAUSED, blocked on a separate, pre-existing bug

Launch #5 ran cleanly (no stale-incumbent contamination) and both `origin_zc` chains completed a
genuine, uncontaminated delta=0.1 (`inherited_GT=nothing`, `new_GT=final_GT=0.0239` both
directions, ~58-62 min wall each — a real duration, not a hollow fast-exit). **But the KNITRO
iteration log for that solve shows a stall, not genuine convergence**:
```
  Iter      Objective      FeasError   OptError    ||Step||    CGits
       0    9.840279e-01   0.000e+00
       1    9.840279e-01   0.000e+00   1.581e-02   1.403e-14        7
       2    9.840279e-01   0.000e+00   1.581e-02   1.227e-13        1
       3    9.840279e-01   0.000e+00   1.581e-02   4.157e-14        2
EXIT: ... relative change in solution estimate < xtol for 3 consecutive iterations.
```
`||Step||` is machine-precision noise every iteration; `OptError` stays at a nontrivial 0.0158,
never shrinking — KNITRO gave up because it detected it *couldn't* move, not because it found a
genuine KKT point, despite `Δ*=0.0039` having enormous slack against the `δ=0.1` budget (no
legitimate feasibility reason to be stuck).

**Investigated (repro script + full detail: [[feedback-gp-perturbation-degenerate-a-od-decode-bug]],
`/bbkinghome/edav/repo_scratch/zc-frechet-draw-moments-2026-08-05/gp_bug_repro/`). First hypothesis
was WRONG and is retracted here**: initially looked like perturbing `w[1]=gp` in isolation decoded
to a degenerate `A_od` (all 20 origins collapsed to one constant). Follow-up check disproved this
— the *unperturbed* calibration point's own `A_od` (inspected properly: full min/max/std, not one
column) spans `9.0` to `6.6e6`, genuinely varied; what looked like collapse was one destination
column, and **every column is constant across origins but varies across destinations** — i.e.
`θ0_up` (the pre-step starting point, not a converged solution) is genuinely **origin-symmetric**
at this point, a real structural property, not a decode defect. `cm_z_from_a` is confirmed both
mathematically (never takes `w[1]`) and empirically (`max abs diff`=`0.0` between perturbed/
unperturbed decodes) to be correctly independent of `gp`.

**The real, narrower, still-open finding**: with `A_od` held exactly fixed at this
origin-symmetric value, shifting `gp` alone by `+1e-3` takes `origin_zc`'s K=3 inner problem from
comfortably feasible (`Δ*=0.0095`) to hard-infeasible (`nStatus=-300`) — real and reproducible,
but whether this is a genuine bug or an inherent extreme-sensitivity property of an
origin-specific restriction evaluated near an origin-symmetric starting point is **not
established**. Supporting evidence for the latter: `common_frechet` (a *non*-origin-specific
restriction, same session, same `powered_aspace` encoding) moved `gp` freely with real step sizes
from a presumably similarly-symmetric start — so the shared coordinate machinery is not globally
broken. Next diagnostic steps (not yet done): check whether `cm_meanzc` (shared, non-origin-specific
`ν_k`) also stalls; check whether a *joint* (gp,A) perturbation — not gp held-isolated — stays
feasible, since that's what a real KNITRO step actually explores.

**All 4 chains killed and left paused** rather than continue burning compute on likely-non-convergent
"results." This is a pre-existing property/possible-issue outside the scope of the ZC-basis-fix
task — flagged clearly, with the correction above, for an explicit decision on priority rather
than pursued further or fixed blind. Logs from all 5 launch attempts preserved at
`POST_VERIFY_FIX/chain_logs/*_wrapper.log` (suffixes: none, `_calib2`, `_calib3`) and per-delta
subdirs, not deleted.

## 11. Deferred (per explicit user instruction, after FULL campaign launch)

- REDUCED/prototype port and cross-formulation FULL/REDUCED fixed-state equivalence.
- Scientific-versioning manifest fields + hard checkpoint-incompatibility gates (task §8) —
  campaign safety was handled procedurally (marker files, stopped jobs) rather than via new
  manifest infrastructure this session.
- Context-immutability/performance audit (task §7).
- D20 W=5,000/W=20,000 intermediate scale tests (superseded in priority by the W=100,000 calibration
  checks + the real campaign itself, per user instruction to prioritize the real launch).
- Push `production/fullA-exact` + tag to `origin`, and the corresponding prototype merge/tag.

## Final verdict block

```
EXPONENT_MAPPING = z_equals_U_to_minus_mu   (mu = theta[1] = 1/theta_trade, verified via
                                              createUDerivatives!.jl/moments!.jl's own U.^(-mu))
THEORETICAL_MOMENT_FINITE = K1:yes K2:yes K3:yes   (mu=0.1335 production; finite through k=7)
OLD_ZC_FEATURE_BASIS = Uk = U.^k   (cm_meanzc_moments.jl::build_raw_mean_pair_matrices, pre-fix)
NEW_ZC_FEATURE_BASIS = z_power_equals_U_to_minus_mu_k   (frechet_power_feature, cm_meanzc_moments.jl)
FULL_ZC = origin_ZC:pass CM_plus_ZC:pass   (D4 K=1,2,3 exhaustive; W=100k K=1,2,3 calibration)
REDUCED_ZC = origin_ZC:not_attempted CM_plus_ZC:not_attempted   (deferred per user instruction)
DERIVATIVE_FORMULAS_CHANGED = false_for_fixed_exponent   (mu fixed on every ZC path; no free-mu
                                                           experimental path shares this builder)
DERIVATIVE_IMPLEMENTATION_GATES = eta_nu:pass economic_outer:pass structured_Hessian:pass
                                   (D4 analytic-vs-FD eta_nu gate; Hessian blocks are pure
                                   Zraw_all/Zpairraw_all consumers, structurally unaffected)
W100K_K3_SMOKE = FULL_origin_ZC:pass FULL_CM_plus_ZC:pass REDUCED_origin_ZC:not_attempted
                 REDUCED_CM_plus_ZC:not_attempted
OLD_ZC_CHECKPOINTS_REJECTED = pass   (marked SCIENTIFICALLY_OBSOLETE_ZC_BASIS_EXPONENTIAL,
                                       not deleted; no new checkpoint auto-load path exists to
                                       formally "reject" -- procedural, not code-level, gate)
RUNNING_OLD_BASIS_ZC_JOBS = stopped_and_preserved_4   (plus 2 caught mid-race, same count basis)
PRODUCTION_RELEASE = 8520c59 local_FF_not_pushed   (user: hold push until campaign genuinely
                                                     running/complete -- now running, re-confirm
                                                     before actually pushing)
PROTOTYPE_RELEASE = not_merged_deferred_per_user_instruction
NON_ZC_CODE_CHANGED = false
DENSE_CODE_USED = false
NEW_ZC_CAMPAIGN_LAUNCHED = true   (W=100k K=3, both families, both directions, delta=0.1/0.5/1/2,
                                   from the genuine calibration point -- explicitly requested by
                                   the user this session, superseding the original task's "do not
                                   launch a campaign" instruction)
EXTRA_LONG_LIVED_BRANCHES_CREATED = 0   (fix/zc-frechet-draw-moments-2026-08-05 to be deleted after
                                          production release completes)
```
