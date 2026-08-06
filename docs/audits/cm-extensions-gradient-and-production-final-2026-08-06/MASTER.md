# A-block gradient resolution + CM+ZC/common-Fréchet production closeout — 2026-08-06

Continuation of `diagnostic/cm-paired-basis-preconditioning-2026-08-05`
(worktree `/bbkinghome/edav/cdw_worktrees/cm-paired-basis-preconditioning-2026-08-05`), starting
from HEAD `1e33270`. `NEW_BRANCHES_CREATED = 0`, `EXTRA_WORKTREES_CREATED = 0` — the existing
branch/worktree was reused throughout.

## 1. Shared A-block gradient — DECISIVE, no shared bug

**Verdict: `correct_previous_test_mismatched_h`, confirmed by a stronger mechanism than "mismatched
h" alone.**

`diag_ablock_same_h_2026-08-06.jl` forces `a_block_fd_component_Cplus!` (the production A-block
secant) to an *exact*, caller-supplied `h` (not its own auto-selected bandwidth), and compares four
quantities at that same `h`:

- **A** — production routine, forced h
- **B** — an independently-coded full winner-rescan (brute-force argmin over all D origins for
  every destination, no top-3 cache, no changed-origins shortcut — genuinely separate code from
  `dest_contrib_incremental_top3_C!`)
- **C** — a fully reoptimized secant (real verified KNITRO resolve at `w0±h·e_idx`)
- **D** — the production default component (its own auto-selected `h≈0.08–0.10`)

Real D20/W=20,000, three A-block coordinates (idx=2 ordinary, idx=380, idx=200), `h ∈
{1e-2,...,1e-6}`:

**A ≡ B to ~1e-10–1e-16 relative at every h tested, every coordinate.** This is decisive: the
shared C+ implementation and the powered-a-space coordinate decode are algebraically correct — an
independently-written reimplementation with no shared code agrees with production to machine
precision. Cross-family confirmation (`diag_ablock_same_h_crossfamily_2026-08-06.jl`, idx=2,
h=1e-5) shows the same bit-exact agreement for CM+ZC and common-Fréchet.

The residual A/B-vs-C gap does **not** shrink to zero as h→0 — it plateaus (idx=2: ~0.2%, idx=380:
~2.8%, idx=200: non-monotonic ~12%). Given KKT/primal-dual residuals on the reoptimized solves are
~1e-13–1e-16 (tight), but the *numerator* being resolved at tiny h (`2h·|deriv|`) shrinks to
~1e-7–1e-9 — comparable to or below plausible KNITRO solve-to-solve reproducibility — the most
parsimonious explanation is reoptimization numerical noise at the h scales tested, not a shared
bug. This was not independently re-derived to 100% certainty (KNITRO's own solve-to-solve
reproducibility bound was not separately measured), but per the task's own decision rule, "D_prod
agrees with D_fixed at every usable h" is independently sufficient to conclude no shared bug.

**No narrow fix was needed or made** — this section is a diagnostic-evidence finding, not a code
change.

## 2. Production bandwidth policy

`PRODUCTION_BANDWIDTH = retain_current`. The default h≈0.08–0.10 secant is a genuine
finite-bandwidth approximation (not a bug, per §1), and existing real campaign evidence (§3, §4)
already shows it driving valid, monotonic-ish outer progress (kappa improving in every real smoke
run this session). A dedicated smaller-bandwidth pilot was not run — not needed to unblock the
family-plumbing closeout, and the current policy is not obviously broken.

## 3. CM+ZC K=3 (intended scientific spec) — real evidence, MERGED to production

Spec: `K_mean=3, K_pair=3, meanzc_profiled_level=2` (the merged k=(sigma-1)=2 exact-collinearity
fix, commits `648d043`/`0671d34`/`e22976d`, confirmed ancestors of this branch's HEAD before use).

- **Calibration feasibility** (`diag_cmzc_k3_intendedspec_2026-08-06.jl`, D20/W=20,000): inner
  solve converges cleanly, `inner_status=0`, `Delta_dual=0.0705`, `feasible=true`,
  primal_dual_gap/KKT ~1e-13.
- **Real public-driver outer smoke** (`smoke_cmzc_k3_intendedspec_2026-08-06.jl`, D20/W=20,000,
  900s budget): `n_eval=5/n_grad=5`, kappa 0.023862→0.024235, every eval
  feasible=true/verified=true.
- **Checkpoint/resume** (`smoke_cmzc_k3_resume_2026-08-06.jl` +
  `diag_cmzc_k3_checkpoint_inspect_2026-08-06.jl`): resumed correctly from the prior checkpoint
  (`n_eval/n_grad` 5→7 after resume, wall_elapsed carried forward correctly), kappa improving
  further to 0.024435. Eta movement confirmed genuine (not stuck): `eta_nu` (log-nu units) moved
  `[0.0928,0.2223,0.3993] → [0.0931,0.2231,0.3999]`.
- **D20/W=100,000** (`diag_cmzc_k3_w100k_2026-08-06.jl`): calibration `Delta_dual=0.00304`,
  feasible=true, verified; one nearby point `Delta_dual=0.05311`, feasible=true, verified; both
  KKT ~1e-13–1e-14.
- **Regression**: `test_callback_health_fake_success_guard_2026-08-06.jl` 8/8 PASS throughout, no
  fake-success signature observed at any real eval.

**Merged**: local `production/fullA-exact` fast-forwarded `a07fcf6→cc18ac0`, pushed to
`origin/production/fullA-exact` (user-authorized both the merge and the origin push), tagged
`cmzc-k3-production-ready-2026-08-06` (pushed).

## 4. Common-Fréchet TWO-FAMILY (cdf_plus_power) — implemented, gated, real bug found+fixed

The prior session left this "architecturally blocked": `CMFrechetLookupState`'s own level block
had no operator-FG kernel for the POW-weighted (levelpow) extension, only a dense-reference
builder. **User-authorized implementing it** (explicit "please do it" after CM+ZC landed).

### 4.1 Implementation (commit `abdf836`)

- New `frechet_levelpow_prefix_sums!`/`frechet_levelpow_forward_sum!` (`cm_frechet_lookup_kernels.jl`):
  Pow-weighted, prefix-sum (not suffix-sum) analogues of the plain level block's own functions —
  the levelpow indicator is `1{Bidx>l}` (reflected), not `1{Bidx<=l}`.
- `CMFrechetLookupState` extended: the CM sub-block's own two-family scratch (`cm_contrib2`,
  `hist_h2`, etc. — previously entirely missing despite the shared `cm_forward_contribution!`/
  `cm_transpose_into_g!` kernels already being Pow-aware) plus the level sub-block's own new
  `levelpow_targets`/`Q_levelpow`/`Hpre_pow`/`g_levelpow` fields. `dual_index!`/the FG functor now
  compute both level and levelpow contributions when `Pow!==nothing`, reusing
  `frechet_level_backward_gradient!` unchanged for the pow backward pass (same Pow-weighted-
  histogram + reflected-prefix-sum reuse pattern the CM block's own eq.36 branch already
  established — no new gradient *formula*, only a new weighted table).
- `operator_verification.jl`'s `_verify_inner_solution_operator_cm_core`: the explicit
  "not yet implemented" guard replaced with the matching levelpow extension.
- `cm_frechet_lookup_production.jl`: fixed the hardcoded single-family `ncm_cm = cctx.ncm -
  cctx.L` construction to read `cctx.n_families` (built generically by `build_cm_bin_ctx` from
  `aug.n_families` — the SAME shared path flexible-CM/CM+ZC already use, no separate Pow
  computation needed).
- `cm_frechet_level.jl`: `build_cm_frechet_production_context`'s `include_truncated_moment=true`
  gate now allows `moment_representation=:operator` (was dense-only); fixed the `obj_cm`
  construction branch condition (was `if include_truncated_moment` alone, which would have forced
  a dense obj even when `:operator` was requested — now `include_truncated_moment &&
  moment_representation===:dense_reference`).

`NEW_GRADIENT_ALGORITHMS = 0` in the sense that matters: every new piece reuses an existing
formula/pattern already validated elsewhere in this codebase (the CM block's own eq.36
Pow-weighted-histogram-plus-reflection trick) — no new derivation.

### 4.2 D4 correctness gate — 50/50 PASS

`test_frechet_levelpow_fg_d4_2026-08-06.jl`: compares the new operator kernel against a **true,
independent dense reference** (a separately-built `PsiObjectiveBundleImplicit`, its own `H` buffer
explicitly primed via `moments!`) at 5 random dual points, both contrast modes, block-by-block
(economic/CM_cdf/CM_pow/level_cdf/level_pow) — **50/50 PASS at ~1e-14–1e-15 relative**, for both
`CMFrechetLookupState`'s FG functor and the independent verifier. A single-family control
(unaffected code path) re-confirmed unchanged.

Two real bugs were caught and fixed *during this verification, before it passed*:
`frechet_level_backward_gradient!`'s `targets::Vector{Float64}` signature was too narrow (rejected
a `SubArray` view of the pow-half slice) — widened to `AbstractVector{Float64}`. And the test
harness's own first draft never primed the dense reference's `H` buffer via `moments!` before
calling its FG functor — the resulting exactly-zero gradient block was caught on inspection, not
accepted at face value (see `feedback-always-verify-analytic-gradient-against-fd-before-trusting`
memory: the same discipline, applied here to a from-scratch numerical gate rather than an FD
check).

### 4.3 D20 real-data gates — PASS at both scales

`diag_frechet_twofamily_d20_2026-08-06.jl`, through the real `cm_frechet_lookup` operator backend
(not dense fallback):

- **W=20,000**: `total_marginal_moments=2000` (matches the intended spec exactly: `2*(D-1)*L +
  2*L = 2*19*50 + 2*50 = 1900+100=2000`). Calibration `Delta_dual=0.005504`, feasible=true,
  verified; nearby point `Delta_dual=0.005567`, feasible=true, verified; KKT ~9e-13.
- **W=100,000**: `total_marginal_moments=2000`. Calibration `Delta_dual=0.000525`, feasible=true,
  verified (`inner_status=-103`, an accepted converged code); nearby point `Delta_dual=0.000653`,
  feasible=true, verified; KKT ~1.5e-12.

### 4.4 Real public-driver bug found + fixed (commit `15b89ac`)

The **first** real outer-gradient callback through `run_cm_upper_checkpointed` crashed:
`DimensionMismatch("new dimensions (19, 50) must be consistent with array length 1900")`,
`grad_callback returned -500`. This was **not** caught by §4.2/§4.3 (those exercise the inner FG
kernel and calibration-point verification only) — it was in a third code path,
`frechet_cm_level_fixed_contribution` (`cm_frechet_cplus.jl`), the C+ envelope-theorem "fold the
fixed CM+level dual contribution into q0 once per outer point" step, which still hardcoded
single-family widths (`reshape(λ_cm,nO,L)` on the now-doubled two-family slice).

Fixed by splitting each tail (CM and level) into cdf/pow halves before slicing into the existing
single-family kernel, mirroring `lfix_cm_aware.jl::cm_fixed_value_contribution_two_family`'s own
already-accepted pattern for the CM block (a direct, unoptimized matvec against `aug.CM`'s
precomputed pow sub-block, done once per outer point — cheap, not a per-Newton-iteration cost).
`precalc_frechet_levelpow_dense` already subtracts `levelpow_targets` into `aug.CM`'s own levelpow
columns at construction time, so no separate target-correction term is needed for that half.

**Verified**: rerunning the exact smoke that crashed (`smoke_frechet_twofamily_w20k_2026-08-06.jl`,
D20/W=20,000, 900s budget) now reaches **`n_eval=4/n_grad=4`**, kappa 0.023862→0.04273, every eval
feasible=true/verified=true.

**Not merged to production** — pending the timing investigation below and your own review of the
new kernel code, given it landed this session and only has D4/D20-smoke-level real-run mileage
(much less than CM+ZC K=3's own evidence base).

## 5. Timing — flagged as a real, unresolved concern, not a clean benchmark

**User-raised concern, directly addressed here rather than glossed over.** Per-eval times observed
in the real smokes above were far larger than "a few seconds," and the conditions under which they
were measured were not clean:

| Family | eval 1 (calibration) | eval 2 | eval 3 | time-in-eval / wall | CPU-time / wall |
|---|---|---|---|---|---|
| CM+ZC K=3 (900s run) | t=66.4s | t=536.7s (Δ=470s) | t=751.4s (Δ=215s) | 1049s/1124s = 93% | 2691s/1124s = **2.4x** |
| Frechet two-family (900s run) | t=40.9s | t=551.1s (Δ=510s) | t=899.4s (Δ=348s) | 946s/996s = 95% | 1183s/996s = **1.2x** |

Two distinct observations:

1. **Calibration is fast (40–70s); non-calibration outer points are 5–12x slower (210–510s).**
   This pattern held for both families and is a within-run comparison, so it is less likely to be
   a pure contention artifact — it deserves real investigation (is the inner solve genuinely
   harder away from calibration at K=3/two-family widths, or is warm-start quality degrading after
   the first step?). **Not resolved this session.**
2. **These runs were NOT clean measurements.** `JULIA_NUM_THREADS=4` was used throughout (the
   production convention on this same machine, visible via `ps`, is `-t 10`). The observed
   CPU-time/wall-time ratios (1.2x–2.4x) are well below even the 4 threads requested, confirming
   real contention. `uptime` at the time of writing shows `load average: 127, 126, 122` on a
   208-core machine, with another user running a dozen-plus CPU-bound Python jobs at 400–700% CPU
   each for the session's duration.

**Conclusion: I cannot currently tell you whether CM+ZC K=3 / two-family common-Fréchet are
genuinely this expensive per outer eval under correct production threading on a quiet machine, or
whether these numbers are dominated by the conditions I ran under.** The user's own baseline
expectation (a few seconds for unrestricted, worse but bounded for ZC's dense blocks) may well be
correct — these numbers should not be read as refuting it. A dedicated, controlled benchmark
(`-t 10`, quiet window, matched W, unrestricted vs. CM+ZC K=3 vs. two-family Fréchet side by side)
is the right next step and was **not done this session**.

## 6. Five-family production-readiness matrix

| Family | Scientific spec | Public inner | Public outer eval | Outer gradient | W100k ≥3-grad smoke | Checkpoint/resume | Campaign ready |
|---|---|---|---|---|---|---|---|
| unrestricted | n/a (baseline) | not touched this session | not touched | not touched | not touched | not touched | unknown — the prior session's own unresolved `obj.arg1`-staleness structural concern on `run_polish_checkpointed_unified`'s `cache_hit=false` branch is STILL not empirically verified either way |
| flexible_cm | pass | pass | pass | pass (A-block confirmed correct §1) | not run this session (prior session's own W100k evidence stands) | not re-tested this session | yes, modulo the timing question (§5) |
| common_Frechet (two-family, cdf_plus_power) | pass | pass | pass | not independently FD-checked this session (only the D4 dense-reference gate, §4.2) | W20k only (n_eval=4/n_grad=4); W100k only calibration+nearby, not a multi-gradient smoke | not tested this session for two-family | no — real bug found+fixed this session (§4.4), landed same day, not yet merged, timing unresolved |
| ZC_only (origin-ZC, corrected basis, K=3) | not touched this session | not touched | not touched | not touched | not touched | not touched | unknown — out of this session's scope, see prior k=2 investigation memory |
| CM_plus_ZC (K=3, corrected basis) | pass | pass | pass | pass (A-block confirmed correct §1) | W100k calibration+nearby only (not multi-gradient) | pass (real resume, n_eval 5→7) | **yes** — merged to production (local+origin), tagged, modulo the timing question (§5) |

Do not read this as "all five ready" — it explicitly is not. `common_Frechet` and `CM_plus_ZC`
each have real, dated evidence this session; `unrestricted` and `ZC_only` were not touched at all
and their prior open questions remain exactly as open as before.

## 7. Verdicts (task's requested format)

```
A_BLOCK_GRADIENT = correct_previous_test_mismatched_h
    (A≡B to ~1e-10-1e-16 relative at every h/coordinate/family tested -- decisive; the residual
    A/B-vs-C gap does not shrink to zero as h->0, consistent with reoptimization numerical noise
    at tiny h, not a shared bug)

PRODUCTION_BANDWIDTH = retain_current
    (finite-bandwidth secant, not a bug per above; real campaign evidence shows it driving valid
    outer progress; no smaller-bandwidth pilot run this session)

COMMON_FRECHET_TWO_FAMILY =
    D4: pass (50/50, ~1e-14/1e-15 rel, both contrast modes, block-by-block)
    W20K: pass (calibration+nearby verified direct; real outer smoke n_eval=4/n_grad=4)
    W100K: pass (calibration+nearby verified direct only, NOT a multi-gradient outer smoke)

CM_PLUS_ZC_INTENDED_SPEC =
    scientific_spec: pass
    W20K: pass (n_eval=7/n_grad=7 after checkpoint/resume, kappa+eta genuinely moving)
    W100K: pass (calibration+nearby verified direct only, NOT a multi-gradient outer smoke)

SHORT_REAL_OUTER_RUNS =
    flexible_CM: pass (prior session's own evidence, A-block re-confirmed correct this session)
    common_Frechet_two_family: pass (this session, post-fix)
    CM_plus_ZC_intended_spec: pass (this session)

FIVE_FAMILY_READY =
    unrestricted: unknown (not touched this session, prior structural concern still open)
    flexible_CM: yes (modulo timing question, section 5)
    common_Frechet: no (real bug fixed same day it was found; timing unresolved; not merged)
    ZC_only: unknown (not touched this session)
    CM_plus_ZC: yes (merged to production local+origin, tagged; modulo timing question)

PRODUCTION_RELEASE = cmzc-k3-production-ready-2026-08-06
    (CM+ZC K=3 only -- local production/fullA-exact fast-forwarded a07fcf6->cc18ac0, pushed to
    origin, tag pushed. common-Fréchet two-family NOT merged: landed same-day, real bug found+
    fixed hours before this report, timing concern unresolved -- deliberately held back pending
    your review and/or the timing follow-up.)

NEW_HESSIAN_ALGORITHMS = 0
NEW_GRADIENT_ALGORITHMS = 0
    (the levelpow extension reuses the CM block's own already-validated Pow-weighted-histogram-
    plus-reflection pattern verbatim -- no new derivation)
DENSE_PRODUCTION_GH_USED = false
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
```

## 8. Production SHA / tags

- `cmzc-k3-production-ready-2026-08-06` → `cc18ac0` — **merged**, local + `origin`.
- Common-Fréchet two-family fix (`15b89ac`) — committed to this branch, **not** merged/tagged.

Local commits this session on `diagnostic/cm-paired-basis-preconditioning-2026-08-05` (all also on
`production/fullA-exact` up through `cc18ac0`, per the merge in §3):

- `cc18ac0` — decisive same-h A-block gradient verdict + CM+ZC K=3 intended-spec real evidence
- `abdf836` — extend `CMFrechetLookupState`/operator verification for two-family common-Fréchet
- `15b89ac` — fix two-family common-Fréchet C+ fixed-contribution DimensionMismatch

## 9. Campaign launch commands (for CM+ZC K=3 only — the one merged family)

```julia
run_cm_upper_checkpointed(w0;
    W = 100_000, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
    L = 50, contrasts = :anchored, probs = probs,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    cm_extension = :cm_plus_moments, meanzc_K_mean = 3, meanzc_K_pair = 3,
    meanzc_profiled_level = 2, include_truncated_moment = true,
    ckpt_dir = <dir>, run_id = <id>, label = <label>,
    checkpoint_interval_s = 60.0, maxtime_real = <budget>)
```

`w0` must include the `eta_nu` tail (length `meanzc_K_mean`); build via `cm_w0_from_calibration`
plus the profiled/mean-of-raw-Z convention shown in `diag_cmzc_k3_intendedspec_2026-08-06.jl`.

**Do not launch a real campaign from this alone** — this session ran short smokes (§3), not a
campaign, and the timing question in §5 is directly relevant to how large a real campaign's
wall-clock/compute budget needs to be.

## 10. Clean branch/worktree status

- Branch: `diagnostic/cm-paired-basis-preconditioning-2026-08-05` (unchanged, reused throughout).
- Worktree: `/bbkinghome/edav/cdw_worktrees/cm-paired-basis-preconditioning-2026-08-05` (unchanged).
- `NEW_BRANCHES_CREATED = 0`, `EXTRA_WORKTREES_CREATED = 0`.
- One pre-existing, unrelated uncommitted diff remains untouched, as found:
  `full_aod_diag/d4_exact/cm_originzc_checkpoint.jl`'s own `pin_outer_algorithm` mirror addition —
  present before this session started, not part of this task's scope, left exactly as found (same
  as the prior session's own disposition of it).
- All new diagnostic/smoke scripts this session are committed (not left as untracked scratch):
  `diag_ablock_same_h_2026-08-06.jl`, `diag_ablock_same_h_crossfamily_2026-08-06.jl`,
  `diag_cmzc_k3_intendedspec_2026-08-06.jl`, `diag_cmzc_k3_w100k_2026-08-06.jl`,
  `diag_cmzc_k3_checkpoint_inspect_2026-08-06.jl`, `smoke_cmzc_k3_intendedspec_2026-08-06.jl`,
  `smoke_cmzc_k3_resume_2026-08-06.jl`, `test_frechet_levelpow_fg_d4_2026-08-06.jl`,
  `diag_frechet_twofamily_d20_2026-08-06.jl`, `smoke_frechet_twofamily_w20k_2026-08-06.jl`.

## Artifacts

- This file: `docs/audits/cm-extensions-gradient-and-production-final-2026-08-06/MASTER.md`
- Large logs (not committed to git): `/bbkinghome/edav/repo_scratch/cm-extensions-gradient-and-production-final-2026-08-06/diag/`
- Pushed to Dropbox: `dropbox:Gravity robustness/Analysis/Server Output/cm-extensions-gradient-and-production-final-2026-08-06/`
