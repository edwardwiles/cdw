# Targeted W=100k refinements + K=3 ZC campaign — 2026-08-04

**Branch:** `campaign/fullA-continuation-polish-2026-08-03` (continued, not re-created).
**Worktree:** `/bbkinghome/edav/cdw_worktrees/fullA-continuation-polish-2026-08-03` (pre-existing,
clean at session start, `origin` matched local HEAD at `59351eb`). `NEW_BRANCHES_CREATED=0`,
`EXTRA_WORKTREES_CREATED=0`.

**SESSION ENDED BY USER DIRECTIVE, mid-verification, 2026-08-04 ~14:15 EDT** — the discovery below
was judged large enough to require rethinking the whole campaign's trajectory rather than continuing
to patch individual cells. All background jobs were deliberately killed (`kill -9`, confirmed clean
termination) at the user's explicit instruction; nothing below Section 7 should be read as a
completed deliverable. This report's job is to preserve exactly what was found and verified so the
next session does not have to re-derive it.

## 0. Headline finding: the completed K=1 W=100k campaign's own published bounds
(`FINAL_BOUNDS_2026-08-04.md`) are very likely understated, for a real, root-caused, fixed reason

Section 7 (below) found and fixed a bug in the shared post-solve verification gate
(`oracle.jl::classify_inner_result`, used by all 5 families): it rejected a mathematically valid,
extremely well-converged point (`inner_status=0`, every real residual passing tolerance by 8-9 orders
of magnitude) whenever one recovered dual weight underflowed to exactly `0.0` in `Float64` — which
happens routinely at W=100,000 draws, and turns out to happen **specifically at the divergence-budget
boundary** — i.e. exactly where the true tightest gains-from-trade bound for each cell lives, because
pushing `gp` to its most extreme feasible value is what concentrates the reweighting into the tails.
This never threw an error; `verified=false` just silently failed the incumbent-admission check in
`cb_F!`, the driver logged one line, and the search moved on as if the point were invalid.

**This was checked against the real, already-published campaign, not just a smoke test.** Grepping
the actual completed campaign's own stdout logs (`/bbkinghome/edav/repo_scratch/
fullA-continuation-polish-2026-08-03/campaign_logs/*.log`) for the literal pattern
`feasible=true verified=false` found **224 occurrences vs. only 110 `verified=true` occurrences**,
and sampling showed the SAME signature — `gp` frozen at a stable value, `Delta` sitting almost exactly
at the cell's own delta budget, `verified=false` repeated many times in a row — across essentially
every restricted-family lower-direction cell checked (`origin_zc`, `common_frechet`, `cm_meanzc`,
`flexible_cm`, both δ=1 and δ=2).

**Directly confirmed on one real cell.** `origin_zc lower δ=1.0`'s own real checkpoint
(`campaign_output/origin_zc/lower/delta_1.0/origin_zc_lower_EXPLORE_DIRECT_SR1_latest.jls`) has a
last outer iterate at `g=0.99946`, right in that stuck cluster, while its **published**
`best_feasible` is a far more conservative `gp=0.998132, Delta=0.372`. Re-running this exact cell
fresh from its own real seed, under the now-fixed gate, found and repeatedly re-confirmed a genuinely
feasible, verified boundary point at `gp=0.999481, Delta≈0.995-1.005` — corresponding to
**GT≈0.000779 vs. the published GT=0.002801, roughly a 3.5x tighter lower bound**. This run was
killed mid-polish (user directive, not a failure) after ~1170s of explore, having already converged
cleanly to this improved point multiple times independently.

**Consequence:** `FINAL_BOUNDS_2026-08-04.md`'s numbers (the entire prior session's headline
deliverable) cannot be trusted as the true tightest bounds without re-running under the fixed gate.
The scale of the effect (a systemic pattern across most/all restricted-family lower-direction cells,
not an isolated glitch) is why the user judged this worth stopping to rethink the whole campaign
plan rather than continuing the originally-scoped Section 5/6/8-10 work as if nothing had changed.

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
throughout. **This job DID run to completion** (`final_GT=0.03288167840901757 == inherited_GT`,
`knitro_status=-201`, 4 explore-stage seed attempts + 1 polish attempt, ~6990s total wall,
`result_source=solved` but with `new_GT==inherited_GT`, i.e. no improvement found) — see
`campaign_output/unrestricted/upper/delta_0.01/report.jls`. **But this entire run executed under the
OLD, pre-fix verification gate** (launched ~11:49, the `oracle.jl` fix was applied ~13:0x — editing a
`.jl` file on disk has no effect on an already-running Julia process, confirmed by checking the fix's
edit time against the process start time). Given δ=0.01 is an extremely tight, narrow feasible
region, and Section 0's finding shows this exact gate systematically rejects genuinely-good boundary
points, **this "no improvement" conclusion cannot be trusted as-is** and needs re-running under the
fixed gate before it means anything.

`UNRESTRICTED_UPPER_DELTA_0_01 = converged_no_improvement_UNDER_OLD_GATE — needs re-run, not trustworthy as published`.

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
drivers). **Killed mid-run (user directive, Section 0), never completed.** Last confirmed state
before termination: `eval 140 t=1496.8s gp=0.9508243298023435 Delta=1.0058679567171454 feasible=false
verified=true` — i.e. still exploring, essentially at the end of its explore budget, right at the
δ=1.0 boundary. No usable final result. Ran under the OLD verify gate for its entire lifetime (same
timing issue as Section 5) — must be relaunched from scratch under the fixed gate, not resumed.

`COMMON_FRECHET_UPPER_DELTA_1 = killed_incomplete_UNDER_OLD_GATE — must be relaunched, not resumed`.

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
  The **only** failing check is `oracle.jl`'s `mmin > tol.m_min_floor` with `m_min_floor=0.0`.

  **Root cause, fully confirmed (not left as a guess) — a second, more targeted diagnostic print
  added at `verify_namedtuple_from_operator` (`operator_verification.jl`, gated the same way, off by
  default) traced the exact zero weight to its source:**
  ```
  m_weights[2184]=0.0 exactly -- underlying r[2184]=-999.4002615604248
  r range across all W=100,000 draws = [-1078.28, 139.62]
  ```
  `m_weights` is not a KNITRO decision variable — it is recomputed independently, post-solve, as
  `m[i] = dPsi(r[i])` for every draw, where `r` is built from KNITRO's own converged duals
  (`cc_algo/Psi.jl`'s hybrid divergence conjugate: `dPsi!(r) = exp(r)` for `r<=1`, `e*r` for `r>1`).
  `exp(r)` is mathematically strictly positive for any finite `r` — it can only equal exactly `0.0`
  in `Float64` via underflow (roughly `r < -745`, since `exp(-745)` is already below the smallest
  representable positive double, `~5e-324`). `r[2184]=-999.4` is comfortably past that threshold:
  `exp(-999.4) ~ 1e-434`, a real, finite, positive number in exact arithmetic, ~100 orders of
  magnitude smaller than `Float64` can hold. **`m>0` genuinely holds at the true optimum here — the
  "0.0" is a floating-point representation artifact of one far-tail Monte Carlo draw (out of
  100,000) whose reweighted probability is astronomically small, not evidence of an invalid KNITRO
  solution or a bug in the solve.** This also explains why every aggregate residual stayed tiny: one
  underflowed-to-zero draw among 100,000 has a negligible effect on `mean_m_resid`/`Delta_dual`/KKT.

  **Fix applied and verified live (not left as a recommendation)** — per direct user instruction:
  relax the boundary AND add an explicit finiteness safeguard, since `mmin>=0` alone does no real
  protective work (`m_weights` is mathematically nonnegative by construction on both branches of
  `dPsi!`, so the boundary check was never the thing actually catching a genuine optimization
  failure). Two changes, both committed:
  1. `verify_namedtuple_from_operator` (`operator_verification.jl`, the shared `:operator`-backend
     builder used by all 5 families, the default backend for all 5) now computes
     `m_weights_all_finite = all(isfinite, m_weights)` directly from the full weight vector — this
     is the real safeguard: it catches an actual NaN/Inf anywhere in the recovered weights (a
     genuine sign of a diverged/pathological solve) directly, not indirectly through whichever
     aggregate statistic happens to blow up.
  2. `oracle.jl`'s `classify_inner_result`: `mmin > tol.m_min_floor` → `mmin >= tol.m_min_floor`,
     with the new `m_finite_ok = get(result, :m_weights_all_finite, true)` added to the `ok` gate.
     `:dense_reference` backend results (non-default, no families use it by default) don't populate
     the new field and fall back to `true` — behavior on that explicit opt-in path is unchanged, not
     regressed.
  **Verified live, both directions:**
  - The exact previously-rejected `cm_meanzc` K=3 point now returns `verified=true`
    (`gp=0.9514357371433328 Delta=1.3726200122634187`, identical numbers, only the classification
    changed).
  - A known-good `origin_zc` K=1 point re-run after the fix gives **bit-identical** output to before
    (`kappa=0.03288167840901757`, `Delta=0.010270108273628202`, `verified=true`) — no regression.

`D4 derivative tests`: **not run** this session — this repo's D4 derivative-test harnesses are
hand-built per family for the K=1 shapes already in production; extending them to K=3 is real work
not attempted here, given the session's time budget. Disclosed gap, not silently skipped.

`K3_PREFLIGHT`:
```
origin_zc:  pass (clean, matched-W, real-K1-seed-extended test — see Attempt 3 above)
cm_meanzc:  pass (verify-gate fix applied to oracle.jl/operator_verification.jl and verified live:
            the previously-rejected point now returns verified=true; a known-good K=1 point is
            confirmed bit-identical/unregressed after the change)
```

## 8. K=3 seed banks, Wave 1, Wave 2

**Not attempted this session** — purely a time-budget matter now, not an unresolved blocker on either
family. Both `origin_zc` and `cm_meanzc` K=3 preflight are clean passes (Section 7): `origin_zc` via
the matched-W, real-K1-seed-extended seed construction validated in Attempt 3; `cm_meanzc` via that
same construction plus the verify-gate fix (`oracle.jl`/`operator_verification.jl`, applied and
verified live to cause no regression). Both are ready to launch Wave 1 in a follow-up session.
`K3_WAVE1` / `K3_WAVE2`: all four chains (`origin_zc upper/lower`, `CM_plus_ZC upper/lower`) =
`not_started`.

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
VERIFY_GATE_BUG = found_root_caused_fixed_verified (oracle.jl/operator_verification.jl -- see §0/§7)
VERIFY_GATE_BUG_REAL_WORLD_IMPACT = confirmed (224 verified=false vs 110 verified=true in the real
    completed campaign's own logs; direct re-run of origin_zc lower delta=1.0 found a ~3.5x tighter
    verified boundary point the original run had repeatedly hit and rejected -- see §0)
SESSION_STOPPED_BY_USER = true (2026-08-04 ~14:15 EDT, to rethink campaign trajectory given the
    above -- all background jobs killed cleanly, not a crash or silent failure)

UNRESTRICTED_UPPER_DELTA_0_01 = converged_no_improvement_UNDER_OLD_GATE -- ran to completion but
    entirely under the pre-fix verify gate; not trustworthy as published, needs re-run
COMMON_FRECHET_UPPER_DELTA_1  = killed_incomplete_UNDER_OLD_GATE -- must be relaunched from scratch

K3_PREFLIGHT =
  origin_zc:  pass (dimension/layout by direct code read; matched-W W=100k real-K1-seed-extended
                    smoke: feasible+verified on eval 1, genuine outer progress confirmed on a
                    follow-up 2-eval run, kappa 0.0329->0.0469)
  cm_meanzc:  pass (dimension/layout pass; smoke solve quality genuinely excellent throughout --
                    primal_dual_gap/mean_m_resid/max_abs_moment_kkt_resid all pass tolerance by
                    8-9 orders of magnitude, inner_status=0; the verify-gate rejection was a real
                    bug (strict m_min>0 against a mathematically-nonnegative-by-construction
                    quantity, tripped by benign Float64 underflow of one far-tail draw's weight,
                    r=-999.4) -- FIXED live in oracle.jl/operator_verification.jl (relaxed to >=,
                    paired with an explicit m_weights-all-finite safeguard so genuine NaN/Inf
                    failures are still caught directly rather than only via aggregate residuals);
                    verified to flip the previously-rejected point to verified=true AND to leave a
                    known-good K=1 case bit-identical/unregressed)
  d4_derivative_tests: not_run (disclosed gap, out of session budget)

K3_WAVE1 =
    origin_ZC_upper:not_started (preflight passed; not launched this session, time budget)
    origin_ZC_lower:not_started (preflight passed; not launched this session, time budget)
    CM_plus_ZC_upper:not_started (preflight passed after verify-gate fix; not launched, time budget)
    CM_plus_ZC_lower:not_started (preflight passed after verify-gate fix; not launched, time budget)

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

**Superseded by Section 0.** The K=3-specific next-steps this list originally contained (launch
`origin_zc`/`cm_meanzc` K=3 Wave 1, extend D4 tests to K=3) are still correct in isolation, but they
are no longer the right *priority* — they're downstream of a question that now needs answering first:
how much of the existing, published K=1 W=100k campaign (`FINAL_BOUNDS_2026-08-04.md`) needs to be
re-run under the fixed verify gate, and by how much do the bounds actually move. Concretely, in
priority order:

1. **Decide the re-verification strategy for the completed K=1 campaign.** Section 0 found the
   `feasible=true verified=false`-near-a-delta-boundary pattern in 224 real log lines across
   essentially every restricted-family lower-direction cell. Before re-running everything at full
   budget, consider a cheap triage pass: for each cell, load its own `EXPLORE_*_latest.jls` /
   `POLISH_*_latest.jls` checkpoint(s), reconstruct the last iterate (see the working pattern in
   Section 0's `origin_zc lower δ=1.0` re-run — seed from `continuation_campaign_cell_driver.jl`
   using the cell's own `best_feasible.w` as primary seed, NOT the fragile z-space
   `checkpoint.zfree`/`cm_a_from_z` reconstruction path, which hit an unrelated evaluation-error bug
   this session and was abandoned in favor of a real re-run instead) and re-run each cell for real
   under the fixed gate, comparing the new incumbent against the currently-published one.
2. **Re-run Sections 5/6 from scratch** (`unrestricted upper δ=0.01`, `common_frechet upper δ=1.0`)
   under the fixed gate — both of this session's own runs executed entirely under the pre-fix code
   and their conclusions (`converged_no_improvement` / incomplete) are not trustworthy as-is.
3. Once the K=1 re-verification picture is clear, launch `origin_zc` K=3 Wave 1 (preflight already
   passed — real K=1 economic block from the immutable registry + a fresh `OriginByPowerLayout(D,3,3)`
   eta tail built at the SAME W as the economic block, W=100,000 for the real campaign) and
   `cm_meanzc` K=3 Wave 1 (preflight now also passes after the verify-gate fix).
4. Extend (or explicitly scope out) a K=3 D4 derivative test before trusting any K=3 Hessian/gradient
   path at W=100k scale.
5. Consider whether the verify-gate fix (`oracle.jl`/`operator_verification.jl`, already committed to
   this branch) should be evaluated for merge into `production/fullA-exact` on its own, ahead of and
   independent of any K=3 work, given its real-world impact on already-published K=1 results.
