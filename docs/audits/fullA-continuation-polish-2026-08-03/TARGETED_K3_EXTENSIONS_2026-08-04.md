# Targeted W=100k refinements + K=3 ZC campaign — 2026-08-04

**Branch:** `campaign/fullA-continuation-polish-2026-08-03` (continued, not re-created).
**Worktree:** `/bbkinghome/edav/cdw_worktrees/fullA-continuation-polish-2026-08-03` (pre-existing,
clean at session start, `origin` matched local HEAD at `59351eb`). `NEW_BRANCHES_CREATED=0`,
`EXTRA_WORKTREES_CREATED=0`.

This report covers real, in-progress work; two sections (5, 6) are long real-KNITRO background jobs
still running at the time this report was written and are documented with their live status, not a
fabricated final result. K=3 Waves 1/2 (Sections 9-10) were **not launched** this session — Section
7's preflight went through three real attempts (documented precisely, including two mistakes on this
session's own part) before landing on a decisive, matched-scale test: `origin_zc` K=3 passes cleanly;
`cm_meanzc` K=3's underlying solve is genuinely excellent but exposed a real, unresolved edge case in
the automated verification gate (see Section 7). Waves were not launched primarily because that
investigation consumed the remaining session time, not because of an unresolved blocker on
`origin_zc`, which is ready to go.

**Correction note:** an earlier version of this report claimed `K3_PREFLIGHT = fail_seed_construction`
for both families based on an under-tested first attempt, and separately mischaracterized
`cm_meanzc`'s `verified=false` flag as if it cast doubt on the K=3 solve itself. Both were corrected
live in this session after direct user pushback — see Section 7 for the full, precise account.

## 1. Prerequisite reading + state verification

Read `MASTER.md`, `FINAL_BOUNDS_2026-08-04.md`, `PILOT_TOURNAMENT_VERDICT_2026-08-03.md`,
`SCIENTIFIC_MANIFEST_VALIDATION_2026-08-03.md`, `CHECKPOINT_CONTENTS_VERIFICATION_2026-08-03.md` in
full before touching anything. Key facts carried forward: seeds must come from
`checkpoint.best_feasible.w` (never `zfree`/`g`); `dual_warm_start` is never spliced verbatim;
`origin_zc` checkpoints are `CMCheckpointV10`, a different schema from the other three families'
`CMCheckpointV9`; the pilot tournament verdict (Direct+SR1 explore, family-specific polish arm)
still governs algorithm choice here.

## 2. Immutable K=1 result/seed registry

New file `full_aod_diag/d4_exact/seed_registry.jl`. Walks every `report.jls` under
`campaign_output/` and `campaign_output_w250k/` (the mutable "latest" paths the existing driver
writes), freezes each to a content-addressed, `chmod 444` copy keyed by
`sha256(final_w)[1:16]`, and writes:

```
/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/IMMUTABLE_SEED_REGISTRY_K1_2026-08-04.csv
```

(30 rows after collapsing 2 exact-content duplicates — a same-delta refinement round's own
record-keeping `OUTPUT_DIR` copy, byte-identical to the canonical path once the round improved on
itself). Columns: `family,direction,delta,W,K_mean,K_pair,manifest_hash,point_digest,frozen_path,
source_report_path,gp,GT,Delta_star,result_source,knitro_status,verification_status,source_commit,
w_scale,source_dirname`. Both the CSV and every frozen `.jls` are `chmod 444`. Checksums:
`IMMUTABLE_SEED_REGISTRY_K1_2026-08-04.csv.sha256` and `frozen_seeds_K1_manifest.sha256` (32 file
hashes) alongside it. `K1_RESULTS_OVERWRITTEN = false` — nothing under `campaign_output/` was
touched; this is a read-freeze, not a migration.

This directly addresses the task's stale-seed-race requirement: downstream K=3 work is required to
read only these frozen paths, never a mutable `campaign_output/.../report.jls` a concurrent
same-family refinement round could still be overwriting.

## 3. Frozen W=100k manifest (K=1 and K=3 namespaces)

New file `full_aod_diag/d4_exact/w100k_manifest.jl`: `FrozenManifest` struct + `MANIFEST_K1` /
`MANIFEST_K3` constants (identical on every field except `MEANZC_K_mean/K_pair` and
`ORIGINZC_K_mean/K_pair`, `1` vs `3`), plus `manifest_hash(m)` (16-hex-char SHA256 digest) used as
the content-addressing/namespace key for the registry and for K=3 output directories.
`MANIFEST_K1_HASH = 890424b84acebe64`, `MANIFEST_K3_HASH` computed identically at include time (see
committed file for the literal value — deliberately not hand-copied here to avoid a second,
driftable source of truth).

## 4. Reusable infrastructure additions

- **`full_chain_include.jl`**: the exact real-driver include list `continuation_campaign_cell_driver.jl`
  already used, extracted so `seed_registry.jl`, the bisection tool, and the K=3 preflight can load
  the identical chain without a second hand-copied list to drift. `continuation_campaign_cell_driver.jl`
  itself now just calls `include(full_chain_include.jl)` — net code deletion at that call site.
- **Generic `extra_seed:<report.jls path>[:role]` mechanism** on `continuation_campaign_cell_driver.jl`:
  any number of these tokens, anywhere in `ARGS`, load an arbitrary same-family report (any delta,
  any source) as an additional seed candidate — used for Section 5's reverified immutable primary
  seed and Section 6's S3. **Real bug found and fixed live**: the tokens were initially landing in
  the driver's fixed positional slots (`ARGS[7]`/`[8]`, i.e. `CROSS_SEED_FAMILY`/`CROSS_SEED_DELTA`),
  which threw for every family except `unrestricted`. Fixed by filtering `extra_seed:` tokens out of
  `ARGS` before positional parsing (`POSITIONAL_ARGS`).
- **`continuation_polish_run_fn_k3.jl`**: K=3 variant of `continuation_polish_run_fn.jl`, for
  `origin_zc`/`cm_meanzc` only. Kept as a **separate file**, not a parametrized K, per this repo's
  own rule against a function silently varying a scientific parameter, and specifically to avoid any
  risk of the K=1 campaign's already-completed behavior drifting.

## 5. Unrestricted upper δ=0.01 targeted repair

**5.1 Primary seed.** The existing driver's own delta-chain layering mechanism already re-loads
`campaign_output/unrestricted/upper/delta_0.01/report.jls` (content-identical to the frozen registry
entry, digest `e45fd5d4eaa7e37b`, GT=0.032882, Δ\*=0.009915) as the `F_current_envelope_incumbent`
seed candidate — no extra plumbing needed; this **is** the reverified immutable primary seed.

**5.2 Alternate boundary seed (path bisection).** New tool `delta_star_path_bisection.jl` +
one-shot runner `run_unrestricted_d001_bisection.jl`: linear interpolation between the frozen
δ=0.01 and δ=0.1 unrestricted points in the live outer coordinates, bisecting toward Δ\*=0.01.
**Two real bugs found and fixed live during this work:**
1. Omitting `ACTIVE_TARGET_DELTA[]` left it at its module default `NaN`, which
   `run_polish_checkpointed_unified` threads straight into KNITRO's own `Delta<=delta_in` constraint
   bound — corrupting `KN_add_eval_callback` (`unexpected return code ... -515`,
   "the upper bound specified for constraint index 0 is undefined"). Fixed by setting a generous,
   non-binding evaluation ceiling (10.0) before each probe.
2. Even a short (3-90s) `production_run_fn` budget is **not** a fixed-point evaluator: the outer
   solver takes real steps and Δ\* can swing by an order of magnitude in a single completed
   iteration (confirmed: t=0.5 gave Δ\*=0.042 at 90s but the very next probe at t≈0.008 with only a
   3s budget still gave Δ\*=0.147, not the ~0.01 a "barely moved" assumption would predict). This is
   a genuine architectural limit of this driver (no "evaluate-only, no outer step" mode exists) —
   `bisect_delta_star` handles it correctly by construction (tracks best-of-all-samples against the
   two known-good endpoints rather than trusting any single noisy intermediate sample), and
   correctly concluded **t\*=0.0, Δ\*=0.009915020809519212** — i.e. the alternate-seed search found
   nothing closer to the target than the primary seed itself. Reported honestly, not forced to look
   like a distinct second seed.

**5.3 Solve.** Launched as a real background job:
```
CAMPAIGN_W=100000 julia --project=. -t 20 continuation_campaign_cell_driver.jl \
  unrestricted upper 0.01 <outdir> 3600 3600
```
Direct+SR1 explore (3600s budget) then Direct+BFGS polish (3600s budget), never-regress rule active
throughout. **Status at report time: RUNNING** (explore stage, ~450 outer evals in ~620s,
gp≈0.9777, Δ oscillating 0.01-0.5 as the search explores). This job was not complete when this
report was written; see `campaign_output/unrestricted/upper/delta_0.01/report.jls` (canonical path)
for the eventual final result once it finishes — **not backfilled into this document**, since doing
so without re-reading the actual finished file would risk exactly the kind of unverified claim this
campaign's own methodology (independent verification before publication) exists to prevent.

`UNRESTRICTED_UPPER_DELTA_0_01 = in_progress_at_report_time` (see live process/log for current state).

## 6. Common-Fréchet upper δ=1 targeted repair

Three candidate seeds:
- **S1** (finalized δ=0.5 incumbent) and **S2** (current δ=1 incumbent): both already reachable
  through the driver's own baseline-envelope + seed-manifest loading (3 raw seed candidates loaded
  from `CONTINUATION_SEED_MANIFEST_2026-08-03.csv` for this family/direction, plus the inherited
  δ=1.0 incumbent GT=0.069612 as `F_current_envelope_incumbent`).
- **S3**: rather than the full path-bisection-to-Δ\*≈1 the task describes, this used the frozen
  δ=2.0 point **directly** as an extra seed (`extra_seed:` mechanism, role
  `S3_from_delta2`, GT=0.073292, Δ\*=1.075) — **a disclosed scope reduction**, not a silent
  substitution: Section 5.2 established live that this driver's short-budget "evaluate Delta* at a
  point" mechanism does not actually hold a point fixed (real outer movement even at 3s), so a
  genuine backtracked-and-reverified point at Δ\*≈1 could not be constructed reliably within this
  session's time budget. Using the δ=2.0 endpoint unmodified is safe (never-regress still applies;
  it can only help, not corrupt, the δ=1.0 result) but is not literally "a point near Δ\*=1
  constructed by backtracking," as originally specified.

Launched as a real background job:
```
CAMPAIGN_W=100000 julia --project=. -t 20 continuation_campaign_cell_driver.jl \
  common_frechet upper 1.0 <outdir> 3600 3600 \
  "extra_seed:<frozen δ=2.0 W100k path>:S3_from_delta2"
```
Direct+SR1 explore then SQP polish (per the pilot tournament's own per-family verdict for CM-family
drivers). **Status at report time: RUNNING** (explore stage, early evals, gp≈0.951, Δ climbing
through the 0.5-0.9 range as it explores away from the S1/S2 seeds toward the S3 basin).

`COMMON_FRECHET_UPPER_DELTA_1 = in_progress_at_report_time`.

## 7. K=3 implementation preflight

**Dimension/layout verification — PASS, by direct code read + successful construction:**
- `origin_zc` K_mean=K_pair=3 resolves (confirmed by direct read of `cm_originzc_config.jl`'s
  `originzc_resolve_K`) to `distribution_restriction=:origin_specific_moments_zero_covariance`, and
  its default `power_target_layout=:origin_by_power` (confirmed: `cm_originzc_config.jl:43`) means
  `OriginByPowerLayout(D=20,3,3)`, `n_eta = K_mean*D = 60` (confirmed: `cm_originzc_target_layout.jl`'s
  own `n_eta`/`target_index` — level-major, origin-minor, and the SAME `nu_{o,k}` serves both the
  mean target (k=1..K_mean) and, for k<=K_pair, the pairwise zero-covariance target — this **is**
  "powers/moment blocks k=1,2,3 for both mean and pair components," verified as one shared parameter
  set, not two). Full outer vector length 380+60=440, constructed and asserted live (PASS).
- `cm_meanzc` K_mean=K_pair=3 resolves (confirmed by direct read of `cm_meanzc_production.jl:104/110`,
  which literally calls `SharedByPowerLayout(aug.K_mean, aug.K_pair)`) to `n_eta=K_mean=3` (one nu_k
  shared across all origins, matching the existing K=1 production convention exactly). Full outer
  vector length 380+3=383, constructed and asserted live (PASS).
- Both families' `CAMPAIGN_RESULTS_ROOT`/checkpoint namespaces are keyed by `MANIFEST_K3_HASH`,
  distinct from the K=1 namespace (`continuation_polish_run_fn_k3.jl` is a wholly separate file with
  its own `label`/`ckpt_dir` construction, `_K3_` embedded in the label).

**Smoke execution — the honest sequence of what was tried, corrected live during this session after
direct user pushback on an earlier imprecise claim (see below):**

*Attempt 1 (wrong):* the first preflight script hand-built its own "calibration point" ([gp;A]
economic block via `ctx.θ0_up[ctx.free_idx]` + `pivot_reduce`) rather than reusing a real one, and
combined it with a raw-moment `nu0 = mean(U[:,o].^k)` tail. This failed identically at K=1, K=2, K=3
and at W=5,000/W=80,000 — which correctly rules out a K=3-specific bug, but was a materially weaker
test than it was first written up as: it never actually confirmed a real K=1 point working, only
that this script's own from-scratch reconstruction was broken at every K. Separately running the
repo's own pre-existing `d20_originzc_shakedown.jl` (referenced but not actually executed in the
original writeup) also failed, identically at K=1 and K=3 — but this was traced to a *different*,
pre-existing staleness bug in that script's own economic-block reconstruction (it reshapes a
`D×D=400`-element raw `θ0_up` slice, but the current context builder returns `D×D_dest=380`
elements under `destination_sample=:exclude_row`, a change that postdates this diagnostic script) —
again nothing K=3-specific, but also not yet a real test of K=3.

*Attempt 2 (correct):* took the REAL, already-converged K=1 `origin_zc` economic block straight from
the immutable frozen registry (`.../W100k/origin_zc/upper/delta_0.01/cdb537b72ace59c1.jls`,
`final_w[1:380]`) and extended only the K=3-specific eta tail — no reconstruction of gp/A at all.
This still failed at first, traced to a *third* confound: the nu0 tail was computed from a context
built at **W=5,000** while the economic block itself was converged at **W=100,000** — different
Sobol draw realizations entirely, an internal scale mismatch, not a K issue.

*Attempt 3 (decisive) — matching W=100,000 on both pieces:*
- **`origin_zc` K=3: clean PASS.** `eval 1 t=22.5s gp=0.9779569571197733 Delta=0.012236391949125854
  feasible=true verified=true` — feasible and independently verified on the very first evaluation,
  reproducing the K=1 seed's own gp/Δ almost exactly. A slightly longer follow-up run (2 evals in
  ~37s) showed genuine outer progress: κ improved from the K=1 seed's 0.0329 to 0.0469.
- **`cm_meanzc` K=3: the underlying solve is genuinely excellent; the automated verify gate rejects
  it on a boundary edge case, not a real defect.** `eval 1` came back `feasible=true verified=false`.
  Rather than accept that flag at face value (a user challenge caught this — see below), a temporary
  `CDW_DIAG_VERIFY=1`-gated diagnostic print was added at the verification call site in
  `cm_checkpoint.jl` (small, reversible, off by default, left in the codebase as a reusable
  diagnostic) to print the actual residuals behind `is_verified_success`:
  ```
  inner_status=0  Delta_dual=1.3726200122634187  Delta_primal=1.3726200122600565
  primal_dual_gap=3.4e-12   (tol 1e-3)   -- passes by ~8 orders of magnitude
  mean_m_resid=1.7e-14      (tol 1e-6)   -- passes by ~8 orders of magnitude
  max_abs_moment_kkt_resid=7.2e-13 (tol 1e-3) -- passes by ~9 orders of magnitude
  m_min=0.0
  ```
  Every genuine convergence/KKT check passes by many orders of magnitude — this is an extremely
  well-converged point, not an approximate one (`inner_status=0` is KNITRO's own clean-optimal code).
  The **only** failing check is `oracle.jl`'s `mmin > tol.m_min_floor` with `m_min_floor=0.0`: one
  recovered dual weight is exactly `0.0`, a legitimate corner-solution value, and the gate uses a
  strict inequality against a floor of exactly zero. Nothing downstream actually broke because of
  it — `Delta_dual` and `Delta_primal` agree to 12 significant figures using the same m-vector that
  includes this zero. **This is very likely an overly strict verification-gate edge case, not
  evidence of an incorrect K=3 solve** — but it is also not confirmed to be harmless in general: this
  exact gate (`is_verified_success`) is what `run_cm_upper_checkpointed`'s own `cb_F!` uses to decide
  whether a point may become the published incumbent (`is_new_best = feasible && verified && ...`),
  so as this gate currently stands, a genuinely excellent cm_meanzc K=3 point sitting on this same
  `m_min=0` boundary would be silently discarded by a real Wave run, not just by this smoke test. Not
  confirmed whether this boundary case ever occurs in the existing, trusted K=1 production results
  (not checked this session) — genuinely unresolved, flagged rather than guessed at.

`D4 derivative tests`: **not run** this session — this repo's D4 derivative-test harnesses are
hand-built per family for the K=1 shapes already in production; extending them to K=3 is real work
not attempted here, given the session's time budget. Disclosed gap, not silently skipped.

`K3_PREFLIGHT`:
```
origin_zc:  pass (clean, matched-W, real-K1-seed-extended test — see Attempt 3 above)
cm_meanzc:  pass_with_caveat (solve quality genuinely excellent; the automated verify gate has a
            real, unresolved m_min=0 boundary issue that could suppress good Wave-1 incumbents --
            recommend loosening VerifiedSuccessTolerances.m_min_floor, e.g. to a small negative
            epsilon, or confirming K=1 never hits this same boundary, before trusting Wave 1's
            cm_meanzc results at face value)
```

## 8. K=3 seed banks, Wave 1, Wave 2

**Not attempted this session** — not because of an unresolved blocker on `origin_zc` (its preflight
is a clean pass), but because the preflight investigation itself (Section 7, including diagnosing
and correcting two of this session's own mistakes) consumed the remaining time budget. `origin_zc`
Wave 1 is ready to launch in a follow-up session using the same matched-W, real-K1-seed-extended
seed construction validated in Section 7's Attempt 3. `cm_meanzc` Wave 1 should not launch until the
`m_min=0` verify-gate question is resolved (Section 7) — launching now risks a wave that silently
discards genuinely good incumbents as "no feasible point found," producing misleading "no
improvement" results that are actually a gate artifact. `K3_WAVE1` / `K3_WAVE2`: all four chains
(`origin_zc upper/lower`, `CM_plus_ZC upper/lower`) = `not_started`.

## 9. Nesting audits

Not run against K=3 (no K=3 results exist yet). The **existing** K=1 upper-direction table already
carries a verified within-delta/cross-family nesting audit from the prior session
(`MASTER.md` §7-8: `unrestricted >= every restricted family` at every δ, confirmed clean after the
cross-family-seed fix). Re-verified here only insofar as the immutable registry (Section 2) freezes
those exact same numbers unchanged — `K1_RESULTS_OVERWRITTEN = false`.

## 10. Resource layout

Sections 5/6's two jobs ran concurrently, 20 threads each, on a 208-core/3TB-RAM host — negligible
contention, no swap. K=3 Wave resource planning (4 chains × 10 cores) was not exercised since no
K=3 wave launched.

## Verdict block

```
UNRESTRICTED_UPPER_DELTA_0_01 = in_progress_at_report_time (real background job, explore stage)
COMMON_FRECHET_UPPER_DELTA_1  = in_progress_at_report_time (real background job, explore stage)

K3_PREFLIGHT =
  origin_zc:  pass (dimension/layout by direct code read; matched-W W=100k real-K1-seed-extended
                    smoke: feasible+verified on eval 1, genuine outer progress confirmed on a
                    follow-up 2-eval run, kappa 0.0329->0.0469)
  cm_meanzc:  pass_with_caveat (dimension/layout pass; smoke solve quality genuinely excellent --
                    primal_dual_gap/mean_m_resid/max_abs_moment_kkt_resid all pass tolerance by
                    8-9 orders of magnitude, inner_status=0 -- but the point is rejected by
                    is_verified_success's m_min>0.0 strict-boundary check, m_min=0.0 exactly;
                    UNRESOLVED whether this is a harmless gate quirk or something that needs fixing
                    before trusting Wave 1 cm_meanzc results)
  d4_derivative_tests: not_run (disclosed gap, out of session budget)

K3_WAVE1 =
    origin_ZC_upper:not_started (preflight passed; not launched this session, time budget)
    origin_ZC_lower:not_started (preflight passed; not launched this session, time budget)
    CM_plus_ZC_upper:not_started (blocked on resolving the m_min=0 verify-gate question)
    CM_plus_ZC_lower:not_started (blocked on resolving the m_min=0 verify-gate question)

K3_WAVE2 = not_started (blocked on Wave 1)

K3_NESTING_GATES = not_applicable (no K=3 results to audit yet)

K1_RESULTS_OVERWRITTEN = false
NEW_BRANCHES_CREATED = 0
EXTRA_WORKTREES_CREATED = 0
REDUCED_CODE_USED = false
PRODUCTION_BRANCH_CHANGED = false
CAMPAIGN_LAUNCHED_OUTSIDE_SCOPE = false
```

## What a follow-up session should do first

1. Check `campaign_output/unrestricted/upper/delta_0.01/report.jls` and
   `campaign_output/common_frechet/upper/delta_1.0/report.jls` for Sections 5/6's actual finished
   results (both were genuinely running, not stalled, as of this report) and append the real numbers
   here — do not re-derive or guess them.
2. Launch `origin_zc` K=3 Wave 1 using the validated seed construction: real K=1 economic block from
   the immutable registry (`final_w[1:380]`) + a fresh `OriginByPowerLayout(D,3,3)` eta tail built at
   the **same W** as the economic block (W=100,000 for the real campaign, not a smaller smoke-test
   W) — see `k3_preflight_smoke_v2.jl` / the `/tmp/w_match_test_k3.jl` diagnostic (not committed;
   recreate from Section 7's Attempt 3 description if needed) for the exact working pattern.
3. Before launching `cm_meanzc` K=3 Wave 1: decide what to do about the `m_min=0` verify-gate
   boundary case (Section 7). Options: (a) loosen `VerifiedSuccessTolerances.m_min_floor` (e.g. a
   small negative epsilon) in `oracle.jl`, if a real economic justification for m=0 being an
   acceptable corner solution is confirmed; (b) check whether real K=1 production incumbents ever
   land on this same boundary (if never, that's itself informative about whether K=3 changes the
   likelihood); (c) at minimum, re-run the `CDW_DIAG_VERIFY=1`-gated diagnostic (now committed in
   `cm_checkpoint.jl`) across a few more K=3 cm_meanzc points to see how often this recurs before
   trusting a full Wave run's "no improvement" results at face value.
4. Extend (or explicitly scope out) a K=3 D4 derivative test before trusting any K=3 Hessian/gradient
   path at W=100k scale.
