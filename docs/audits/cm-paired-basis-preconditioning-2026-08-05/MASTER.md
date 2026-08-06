# Paired-basis preconditioning pilot for the two-family common-marginals moment system — MASTER

Branch: `diagnostic/cm-paired-basis-preconditioning-2026-08-05`, base commit `744ea1d`
(`fix/cm-add-truncated-power-moments-2026-08-05`) plus a carry-forward commit (`5c3eb8e`, see
section 0) reproducing that branch's own uncommitted continuation-session state (the indicator-
direction bugfix and Architecture C extension — MASTER.md sections 17-25 of the source branch).
Worktree: `/bbkinghome/edav/cdw_worktrees/cm-paired-basis-preconditioning-2026-08-05`.

**Bottom line up front**: the pairwise residualization+RMS-scaling transform is real, correct, and
gives a large, reproducible conditioning improvement (~15-22x on the full moment-system condition
number, confirmed identically at two independent `W` scales, with machine-precision scientific
equivalence to the raw formulation). **But it does not reproduce, and therefore cannot be
confirmed to fix, the actual documented production failure** (`nStatus=-400` under Architecture
C's no-dense-H `:operator`/`:cm_lookup` path) — every arm in this pilot, **including the
unmodified raw basis**, converges cleanly via the dense Architecture A path this task's own scope
restricts diagnostics to. Verdict: **`inconclusive_architecture_mismatch`**, not productionized.
See section 5 for the full finding and section 8 for the exact reasoning.

## 0. Provenance of the frozen base state

The task brief's own cited diagnostics (CDF block cond≈30-32, POW block cond≈30, stacked CM
cond≈8,700+, max paired-column correlation≈0.9997, full system cond≈85,000) are the exact numbers
reported in the source branch's `MASTER.md` section 22 (`test_cm_moment_rank_2026-08-05.jl`,
real D20 data, W=100,000, L=10) — computed against a state that included two same-day fixes **not
yet committed** on `fix/cm-add-truncated-power-moments-2026-08-05` at the time this task began:
the eq.36 indicator-direction fix (`1{U>c}`, not `1{U<=c}`) and the Architecture C extension to the
two-family spec. That state existed only as an uncommitted working-tree diff in the sibling
worktree `/bbkinghome/edav/cdw_worktrees/cm-add-truncated-power-moments-2026-08-05`. Commit
`5c3eb8e` on this branch reproduces that diff byte-for-byte (no new lines authored) so this pilot
runs against the exact state the task brief's numbers came from; the original worktree/branch was
left untouched (nothing committed there, no files removed).

## 1. Coordinate pairing (task §3)

Column layout confirmed directly from `precalc_common_marginals_cdf` (threshold-major, CDF block
then POW block, each block indexed `(l-1)*nO + oi`): CDF column `j` and POW column `ncm_cdf+j`
share the identical `(quantile l, origin o)` pair by construction — confirmed by reading the
feature-construction code (`common_marginals_moments.jl`), not assumed from matching offsets.
Full map: `CM_PAIRED_COORDINATE_MAP_{scale}_{W}_{L}.csv` (one per scale below; D4: 30 pairs, D20:
190 pairs, columns: `cdf_col, pow_col, quantile_index_l, quantile_prob, z_cutoff, origin_position,
origin_index, ref_index1`).

## 2. Transform diagnostics (task §4)

`β_j = E_{F*}[c_j p_j]/E_{F*}[c_j^2]`, `r_j = p_j - β_j c_j`, `s_{c,j}/s_{p,j}/s_{r,j}` = RMS
(uncentered, per task spec) under the fixed raw draws `ctx.U` (the reference measure `F*`).
Full per-pair CSVs: `CM_TRANSFORM_DIAGNOSTICS_{scale}_{W}_{L}.csv`.

| scale | n pairs | β range | mean \|corr\| | max \|corr\| | s_r/s_p ratio (mean) | degenerate (s_r≈0) |
|---|---|---|---|---|---|---|
| D4, W=8,000 | 30 | [-1.10, -0.75] | 0.960 | 0.9978 | 0.225 | 0 / 30 |
| D20, W=20,000 | 190 | [-1.03, -0.89] | 0.993 | 0.99966 | 0.094 | 0 / 190 |
| D20, W=100,000 | 190 | [-1.03, -0.89] | 0.993 | 0.99966 | 0.094 | 0 / 190 |

No degenerate residual coordinate at any scale (`s_r < 1e-8 * max(s_c,s_p,1)` tolerance) — every
pair is an ordinary, non-redundant near-collinearity, consistent with MASTER.md section 22-23's
own finding of full formal rank (no hard singularity, unlike the separate CM+ZC K=2(σ=3) case).
`β_j ≈ -1` almost everywhere is itself informative: it says `p_j ≈ -c_j + r_j` with a *small*
residual (`s_r/s_p ≈ 0.09-0.22`), i.e. the POW feature is close to the *negative* of the CDF
feature plus a small correction — expected, since both features share the same `1{...}` event
structure and differ mainly in whether that event is weighted by 1 or by `z^(σ-1)`.

## 3. Four-arm results

Dense Architecture A (`moment_representation` equivalent — `evaluate_fullA`/`wrap_moments_with_cm`,
the same generic dense splice MASTER.md sections 19-20/24b already used for two-family correctness
gates and the POW-only isolation experiment), per task's own precedent for diagnostic-only dense
use (never a campaign default). Same context/draws/calibration point/inner solver/tolerances/maxit
across all 4 arms at each scale — only the CM matrix content differs.

### D4, W=8,000, L=10 (`d4_exact_setup`, native W)

| Arm | status | Δ_dual | cond(CM) | cond(fullG) | t (s) | iters |
|---|---|---|---|---|---|---|
| A raw | 0 | 0.00452213382691255 | 1,431.1 | 1,510.3 | 9.26† | 4 |
| B diag-RMS | 0 | 0.00452213382691250 | 1,457.6 | 1,488.1 | 0.067 | 4 |
| C residual-only | 0 | 0.00452213382691254 | 824.2 | 894.2 | 0.063 | 4 |
| D residual+RMS | 0 | 0.00452213382691255 | 234.5 | 345.3 | 0.061 | 4 |

† Arm A pays first-call JIT/compile cost within the process; not a clean per-arm timing signal
(see section 5).

### D20, W=20,000, L=10 (real data, raw calibration point `ctx.θ0_up[ctx.free_idx]`)

| Arm | status | Δ_dual | cond(CM) | cond(fullG) | t (s) | iters |
|---|---|---|---|---|---|---|
| A raw | 0 | 0.0216122600148720 | 8,938.3 | 86,821.1 | 17.52† | 7 |
| B diag-RMS | 0 | 0.0216122600148724 | 9,337.4 | 57,413.5 | 6.20 | 7 |
| C residual-only | 0 | 0.0216122600148722 | 4,687.6 | 61,635.1 | 6.50 | 7 |
| D residual+RMS | 0 | 0.0216122600148723 | 605.8 | 4,047.8 | 6.76 | 7 |

### D20, W=100,000, L=10 (real data, raw calibration point — directly comparable to source
### MASTER.md section 22, which used this exact W/L/point)

| Arm | status | Δ_dual | cond(CM) | cond(fullG) | t (s) | iters |
|---|---|---|---|---|---|---|
| A raw | 0 | 0.00422809124930465 | 8,729.9 | 85,342.9 | 37.43† | 5 |
| B diag-RMS | 0 | 0.00422809124930467 | 9,115.8 | 56,582.2 | 26.13 | 5 |
| C residual-only | 0 | 0.00422809124930477 | 4,574.5 | 60,538.5 | 24.46 | 5 |
| D residual+RMS | 0 | 0.00422809124930476 | 585.5 | 3,953.7 | 24.54 | 5 |

`cond(CM)=8,729.9` and `cond(fullG)=85,342.9` here match the source MASTER.md section 22 values
(8,730 / 85,343) to 5 significant figures — confirms this pilot is running against the identical
numerical state the task brief's citations came from.

**Every arm at every scale, including the RAW two-family basis, converges to `nStatus=0`** with a
healthy, non-degenerate solution (`m_min` at W=100k: 0.336, far from 0; `max_abs_moment_kkt_resid`
~1e-14 to 1e-15 for every arm). Δ_dual agrees across all 4 arms to 11-12 significant digits at
every scale (expected: same feasible set, same optimum, different — but equivalent — coordinates).

## 4. Exact mathematical equivalence (task §6)

`T_j` per pair, `[c̃_j; g̃_j] = T_j [c_j; p_j]`: Arm A `T_j=I`; Arm B `T_j=diag(1/s_c,1/s_p)`;
Arm C `T_j=[[1,0],[-β_j,1]]`; Arm D `T_j=[[1/s_c,0],[-β_j/s_r,1/s_r]]`. `T_j^{-1}` correspondingly
`I`, `diag(s_c,s_p)`, `[[1,0],[β_j,1]]`, `[[s_c,0],[β_j s_c,s_r]]` (all invertible since
`s_c,j, s_r,j > 0` at every tested pair, confirmed section 2). Raw dual recovered as
`λ_raw,pair = T_j^T λ̃_pair`.

All three checks below hold at **machine precision, at every scale**, confirming the transform,
its inverse, and the dual map are implemented correctly (not merely "close"):

| scale | arm | raw CDF-moment resid | raw POW-moment resid | Δ_dual (recompute vs. original) | dual-map identity max\|Δ\| |
|---|---|---|---|---|---|
| D4 | B | 3.80e-16 | 3.41e-16 | exact match | 3.44e-15 |
| D4 | C | 2.58e-17 | 3.46e-17 | exact match | 2.66e-15 |
| D4 | D | 1.95e-17 | 2.62e-17 | exact match | 3.77e-15 |
| D20/20k | B | 4.80e-16 | 4.57e-16 | exact match | 3.29e-14 |
| D20/20k | C | 1.30e-16 | 1.32e-16 | exact match | 4.81e-14 |
| D20/20k | D | 5.97e-17 | 5.44e-17 | exact match | 4.09e-14 |
| D20/100k | B | 1.25e-15 | 1.17e-15 | exact match | 1.55e-14 |
| D20/100k | C | 2.19e-17 | 2.15e-17 | exact match | 1.57e-14 |
| D20/100k | D | 3.51e-17 | 3.72e-17 | exact match | 1.69e-14 |

"Raw CDF/POW-moment resid" = `|Σ_s m(s)·g_raw[s,j]|/W` at the transformed-arm's converged
`(ζ*,λ̃*)`, i.e. the raw eq.35/eq.36 restrictions ARE satisfied (≈0) under the reweighting the
transformed-basis solve actually recovers — this is the literal requirement of task §6 ("verify
the original raw moments, not just transformed residuals"). "Dual-map identity" = a pure linear-
algebra check (`λ_raw^T g_raw[s,:] == λ̃^T g̃[s,:]` at 500 random draws `s`), independent of any
solver — confirms `T`/`T^{-1}`/the dual map themselves, not just that the solves happened to agree.

(An earlier version of this harness had two bugs — a dual-vector slicing offset error and an
object-reuse error that rebuilt a never-solved fresh object for the recompute — both caught by
this exact section-6 check failing loudly with a `DimensionMismatch` and then a wildly wrong
`Δ_dual` recompute, and both fixed before any result above was accepted. Left as evidence the
equivalence check is a real, sensitive test, not a rubber stamp.)

## 5. The decisive, unexpected finding: raw already converges under Architecture A

MASTER.md section 21-25 (source branch) documents `nStatus=-400` for the raw two-family CM at the
identical `(θ, W, L)` combination tested here — `W ∈ {20k,100k,300k} × L=10`, and `L ∈ {3..10} ×
W=100k` — via `build_cm_production_context(...; moment_representation=:operator,
inner_fg_backend=:cm_lookup, use_archB_moments=false)` + `archC_base_state(...)`, i.e. the real,
no-dense-H, structured-Hessian Architecture C production path.

**This pilot's Arm A — the unmodified raw two-family CM, same context, same W, same L, same
calibration point, same `.opt` file, conditioning numbers matching to 5 significant figures —
converges cleanly (`nStatus=0`) via `evaluate_fullA` (dense Architecture A).** This is not a
different, easier problem: the moment matrix is numerically the same near-collinear system
(`cond(fullG)=85,343` here, `85,343` in the source document, at W=100k/L=10). The two KNITRO
backends behave completely differently on the identical mathematical problem.

Section 20 of the source MASTER.md established that Architecture C's Hessian/gradient *formulas*
match Architecture A/ForwardDiff at machine precision at 4 random dual points — but formula
equivalence at a handful of random points does not guarantee identical convergence *behavior* along
the specific iterate sequence a real solve traverses, especially near a near-singular Hessian
direction where floating-point roundoff order (dense BLAS accumulation vs. Architecture C's
bin-table summation) or a KNITRO Hessian-provision-mode difference could plausibly tip a borderline
Newton step one way or the other. **This pilot did not, and per its own scope could not, determine
which of these (or something else specific to Architecture C) is the actual cause** — diagnosing it
would mean instrumenting or modifying Architecture C's own Hessian/FG code, which this task
explicitly prohibits ("do not create new Hessian, gradient, verifier, or runner implementations").

This is reported as the single most important finding of this pilot, not buried in a footnote,
because it directly determines whether "productionizing the transform" would fix anything real (see
section 8).

## 6. Conditioning improvement is real, large, and reproducible — but doesn't touch section 5's gap

Arm D vs. Arm A, `cond(fullG)`: **21.4x at W=20,000, 21.6x at W=100,000** (near-identical ratio at
two independent scales — not a fluke of one draw realization). `cond(CM)`: 14.8x / 14.9x. Arm C
(residualization alone, no scaling) already gets roughly half this benefit (cond(fullG) 1.4x, 1.41x
at 20k/100k respectively relative to raw — computed as 86821/61635=1.409, 85343/60538=1.410); the
RMS scaling in Arm D contributes the rest. Diagonal RMS scaling alone (Arm B) is not sufficient by
itself and sometimes makes `cond(CM)` marginally *worse* (9,337 vs. 8,938 raw at W=20k) — confirms
the leading hypothesis was correctly specific to the *pairwise* near-proportionality, not a generic
scale mismatch between the two families.

Iteration counts are **identical** across all 4 arms at each scale (4 at D4, 7 at W=20k, 5 at
W=100k) — the better-conditioned bases do not reach the solution in fewer KNITRO iterations here
(all arms were already comfortably within budget). Wall-clock is faster for B/C/D than A at D20
(~3.6x at W=100k), but Arm A is the first arm solved in-process and pays first-call
`evaluate_fullA`/KNITRO-callback JIT/compilation cost that B/C/D inherit already-compiled — this
is a **confound, not a clean conditioning-driven speed win**, and is reported as such rather than
claimed.

## 7. Old-implementation history check (task §12, time-boxed ~10 min, well under the 20-min budget)

Searched `cdw` git history (`git log --all --grep`, targeted at `common_marginals*`/`cm_*.jl`) for
column scaling, residualization, QR/Cholesky, orthonormalization, interval-basis, or
moment-normalization work that might already have solved this. Found: `contrasts=:orthonormal`
(the pre-existing origin-contrast basis `R`, applied identically to *each* family's own block
independently — unrelated to CDF/POW cross-family conditioning) and a 2026-07-20 CDF-only
interval-vs-cumulative basis reformulation (`common_marginals_interval.jl`, single-family, predates
eq.36 entirely). Checked KNITRO `.opt` files for a scaling option: `ek_inner.opt` sets
`scale user_internal`, ambient KNITRO auto-scaling infrastructure that predates and is unrelated to
this specific two-family interaction. **No prior implementation targeted the eq.35/eq.36
collinearity** — unsurprising, since eq.36 (and therefore this exact collinearity) did not exist in
this codebase before the same day this whole investigation started (2026-08-05).

## 8. Go/no-go decision

Per task §8, strong criteria (any one, plus no equivalence-gate failure):

- CM condition estimate falls ≥100x: **no** (14.8-14.9x).
- **Full condition estimate falls ≥10x: yes — 21.4x (W=20k), 21.6x (W=100k), reproduced at two
  independent scales.**
- Raw combined system fails/stalls while Arm D converges: **no — the raw system (Arm A) already
  converges cleanly in every instrumentation this task's scope permits** (section 5).
- OptError after fixed iteration budget improves ≥100x: not applicable — every arm reaches full
  KKT convergence within budget (no arm was iteration-starved to compare a partial-convergence
  OptError against).
- Iterations/CPU fall materially and reproducibly: **no** — iteration counts are identical across
  arms; the wall-clock gap is a JIT-order confound, not a clean effect (section 6).

Equivalence gates (raw-moment residuals, Δ_dual agreement, dual-map identity): **all pass at
machine precision, at every scale** (section 4).

**One strong criterion is unambiguously met** (full-condition ≥10x, cleanly reproduced) **and no
equivalence gate fails** — a literal reading of the rule says "go." But the criterion this whole
pilot exists to satisfy — fixing the *actual documented failure* (`nStatus=-400` under the real
production Architecture-C path) — is not merely "not clearly improved," it is **not reproduced at
all** within this task's permitted instrumentation (dense Architecture A only; extending
Architecture C to accept the transformed basis is exactly the "new Hessian/gradient" work this task
prohibits). Productionizing per task §9 can only route the transform through "the existing generic
CM operator," which *is* the dense splice — the same path that already solves the raw system fine.
Doing so would give a real, measurable conditioning benefit to whoever already uses
`moment_representation=:dense_reference` (correctness gates, diagnostics, `archC_verified_state`'s
own dense-reference comparison arm), but **would not be validated to fix, and cannot currently be
validated to fix, the real production blocker**, because that blocker isn't reachable by any
solve this pilot is allowed to run.

**Verdict: `inconclusive_architecture_mismatch`.** Not productionized. The transform itself is
correct and the conditioning win is real (worth keeping as a documented, tested candidate for later
use if Architecture C's own root cause turns out to be conditioning-sensitive after independent
investigation) — but merging it now, based on evidence gathered on a path that never exhibited the
failure being fixed, would be exactly the kind of overclaimed fix this project's own standing
practice (CLAUDE.md, memory `feedback-solver-option-cannot-fix-genuine-rank-deficiency`,
`feedback-verify-mechanism-across-every-case-not-just-first`) warns against.

**Recommended follow-up (out of scope here, not attempted):** instrument Architecture C's own
KNITRO Hessian-provision call (compare its exact options/mode against `evaluate_fullA`'s) and/or
compare the two backends' Hessian values numerically along the *actual failing iterate sequence*
(not just at random points as section 20 did) to determine whether the discrepancy is a KNITRO
options difference, a roundoff-order sensitivity, or a residual Architecture C bug. If it turns out
to be conditioning-sensitive after all (e.g. Architecture C's own factorization is less robust to
`cond(fullG)~85,000` than KNITRO's dense-Hessian path), this transform — already built, tested, and
proven correct here — would need Architecture C's own kernels extended to consume `(c̃, r̃)`
instead of `(c, p)` before it could actually help; that extension was explicitly out of this task's
scope.

## 9. Deliverables checklist

- `CM_PAIRED_COORDINATE_MAP_{scale}_{W}_{L}.csv` — three files (D4/W8000, D20/W20000, D20/W100000),
  `/bbkinghome/edav/repo_scratch/cm-paired-basis-preconditioning-2026-08-05/`.
- `CM_TRANSFORM_DIAGNOSTICS_{scale}_{W}_{L}.csv` — beta/scale/residual/correlation per pair, same
  directory, three files.
- Four-arm tables: section 3 (this document).
- Raw-vs-transformed equivalence table: section 4.
- Condition estimates: sections 2-3, 6.
- Old-implementation history note: section 7.
- Go/no-go decision: section 8.
- Full run logs: `run_d4_L10_v3.log`, `run_d20_w20k_L10.log`, `run_d20_w100k_L10.log` (same
  directory).
- Harness: `full_aod_diag/d4_exact/test_cm_paired_basis_precond_2026-08-05.jl` (this branch).

## 10. Final verdict block

```
RAW_BASIS =
    status:0 (all 3 scales)
    condition:1510.3(D4) / 86821.1(D20,W20k) / 85342.9(D20,W100k)   [cond(fullG)]
    OptError:converged (opttol met, nStatus=0 at every scale)

RMS_ONLY =
    status:0 (all 3 scales)
    condition:1488.1(D4) / 57413.5(D20,W20k) / 56582.2(D20,W100k)
    OptError:converged

RESIDUAL_ONLY =
    status:0 (all 3 scales)
    condition:894.2(D4) / 61635.1(D20,W20k) / 60538.5(D20,W100k)
    OptError:converged

RESIDUAL_PLUS_RMS =
    status:0 (all 3 scales)
    condition:345.3(D4) / 4047.8(D20,W20k) / 3953.7(D20,W100k)
    OptError:converged

RAW_MOMENT_EQUIVALENCE =
    pass  (raw CDF/POW moment residuals under each transformed arm's recovered reweighting:
           1e-15 to 1e-17 at every scale/arm, section 4)

DELTA_STAR_EQUIVALENCE =
    pass  (Delta_dual agrees to 11-12 significant digits across all 4 arms at every scale;
           exact recompute match for every arm; dual-map identity check ~1e-14 to 1e-15)

CONDITION_IMPROVEMENT =
    CM_ratio:14.8x (D20,W20k) / 14.9x (D20,W100k) / 6.1x (D4)
    full_ratio:21.4x (D20,W20k) / 21.6x (D20,W100k) / 4.4x (D4)

PILOT_DECISION =
    inconclusive_<architecture_mismatch: the raw basis already converges (nStatus=0) via the only
    inner-solve architecture this task's scope permits (dense Architecture A), so this pilot never
    reproduced the actual documented production failure (nStatus=-400 under Architecture C's
    no-dense-H :operator path) to test whether the transform fixes it. The condition-number strong
    criterion IS met (>=10x full-condition improvement, reproduced at 2 independent W scales) and
    no scientific-equivalence gate fails, but productionizing on that basis alone would be an
    unvalidated claim against the real failure mode -- not attempted, per this project's standing
    practice against overclaiming fixes that were only verified on a path that never exhibited the
    original problem.>

GENERAL_WHITENING_BUILT = false
NEW_HESSIAN_FORMULAS = 0
NEW_GRADIENT_FORMULAS = 0
NEW_FAMILY_RUNNERS = 0
SCIENTIFIC_RESTRICTION_CHANGED = false
DENSE_PRODUCTION_GH_USED = false (dense Architecture A used only as this pilot's explicit, disclosed,
    non-default diagnostic instrument -- matching this exact repo's own established precedent for
    two-family CM correctness gates (source MASTER.md sections 19-20, 24b) -- never as an ambient
    default, never in a real campaign)
CAMPAIGN_LAUNCHED = false

PRODUCTION_RELEASE =
    not_merged_inconclusive_pilot
```
