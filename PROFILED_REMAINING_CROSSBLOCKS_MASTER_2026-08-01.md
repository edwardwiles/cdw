# Profiled economic block, remaining cross-blocks (H_EF/H_EZ) — master status (2026-08-01)

## Branch / commits / worktree

- Continuation branch: `architecture/profiled-economic-block-remaining-crossblocks-2026-08-01`
- Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-economic-block-remaining-crossblocks-2026-08-01`
- Forked from: `architecture/profiled-economic-block-all-families-complete-2026-08-01` @ `6f6a878` (the
  recorded HEAD in the task prompt; confirmed via `git worktree add ... -b <new> 6f6a878`, working
  tree clean at fork time).
- Source branch `6f6a878` itself was **not modified** — worked entirely in this new worktree/branch.
- Not merged, not pushed to `origin`, no production default changed, no campaign launched.

## What this session actually did

The handoff (`profiled_all_family_phase4-6_hec_bugfix_2026-08-01.zip`, and this branch's own commit
history) established that H_EC's `use_profiled_correction=true` path had two real formula bugs — the
correction-term multiplier used the structured-formulation `pi_vec[j]` instead of the homogeneous
formulation's `Lam_homog[j] = kappa0[j]*Pmat[o,slot]`, and the France/cf row was missing a constant
`denom_cf_scaled*nu_diff` additive term — both found and fixed in `winner_pair_cross_hessian_cm_block!`
via a direct ForwardDiff cross-check the user pushed for. The handoff explicitly flagged, but did NOT
fix, the **identical bug pattern** in three sibling functions:

- `winner_pair_cross_hessian_colsum!` / `winner_pair_cross_hessian_esum!` (H_EF, common Fréchet)
- `winner_pair_cross_hessian_zc_block!` / `winner_pair_cross_hessian_zc_block_threaded!` (H_EZ,
  ZC-only and CM+ZC)

and that these functions' own existing profiled test files validated against a brute-force reference
that itself used `pi_vec` — i.e. they were passing by being self-consistently wrong, not by being
correct.

### 1. Fixed all four production functions with the same `Lam_homog`/`denom_cf_scaled` pattern

**`winner_pair_cross_hessian_colsum!`** (`full_aod_diag/d4_exact/winner_pair_cross_hessian.jl:619-681`):
the bilateral multiplier now switches to `Lam_homog[j]` in lockstep with the correction term switching
to `MSumX[target_slot[j],l]` (previously the correction term switched but the multiplier stayed
`pi_vec[j]`); the France row gains an additive `wctx.denom_cf_scaled*sumNu` term before subtracting
`Lam_homog[jcf]*MSumX[target_slot[jcf],l]` — `sumNu` (the sum-over-origins analog of H_EC's `nu_diff`)
is the correct constant-term partner here, confirmed by `MSumX`'s own pre-existing field docstring
("H_EF's colsum!/esum! sum-over-x analog of H_EC's per-x difference").

**`winner_pair_cross_hessian_esum!`** (`winner_pair_cross_hessian.jl:700-748`): same pattern, using
`T0_slot[target_slot[j]]`/`Lam_homog[j]` for the bilateral rows and `+ wctx.denom_cf_scaled*t0` for the
France row's additive constant (`t0` is the un-binned analog of `sumNu`).

**`winner_pair_cross_hessian_zc_block!`** (`winner_pair_cross_hessian.jl:493-604`, serial) and
**`winner_pair_cross_hessian_zc_block_threaded!`** (`threaded_cross_hessian.jl:220-330`, threaded
twin): the bilateral multiplier switches to `Lam_homog[j]` against the `TZ[d,x]` correction; the
France row gains `+ wctx.denom_cf_scaled*NuZ[x]` before subtracting `Lam_homog[jcf]*TZ[d_cf,x]` (`NuZ`
is the Z-feature analog of `nu_diff`/`sumNu`/`t0`). Both the serial and threaded versions received the
identical edit, so their pre-existing bit-for-bit-agreement test continues to certify true agreement,
not two independently-wrong implementations that happen to match.

The derivation for each "sum" function's correction/constant-term pairing was obtained by direct
structural analogy to H_EC's own (o,refIndex1)-DIFFERENCE formula — every "difference" quantity in
H_EC (`nu_diff`, `corr = MCScum[d,o,l]-MCScum[d,refIndex1,l]`) has an exact "sum" or "un-binned"
counterpart already built and documented on `WinnerBinCrossScratch`/`WinnerZCCrossScratch`
(`sumNu`/`MSumX`, `t0`/`T0_slot`, `NuZ`/`TZ`) — no new scratch fields or O(W) passes were needed; this
was purely a bugfix to which existing quantities get multiplied by which existing coefficients.

### 2. Rewrote every profiled test that used `pi_vec` as its own reference

- `test_profiled_hef_correction_d4_2026-08-01.jl`: both `expected_new` lines (colsum!/esum!) switched
  from `wctx.pi_vec[j]` to `wctx.Lam_homog[j]`.
- `test_profiled_hez_correction_d4_2026-08-01.jl`: the one `expected` line switched from
  `wctx.pi_vec[j]` to `wctx.Lam_homog[j]`.
- `test_profiled_hez_threaded_d4_2026-08-01.jl`: **not edited** — it only ever compares threaded vs.
  serial output bit-for-bit, never against an independent formula, so it needed no reference change;
  it continues to pass, now certifying agreement between two CORRECT implementations.

### 3. Found and fixed a second, independent bug: the France-row test was passing vacuously

`test_profiled_france_row_d4_2026-08-01.jl` (added by the handoff branch to specifically exercise the
France/cf row's profiled path for H_EC/H_EZ/H_EF) called `build_winner_pair_ctx(cf; bi_slot=bi_slot)`
with **no `gpσ`/`denom_cf` keywords**, i.e. both silently defaulted to `0.0`. That makes
`wctx.Lam_homog[cf.cf_col] = kappa0[cf.cf_col]*0.0 = 0` and `wctx.denom_cf_scaled =
kappa0[cf.cf_col]*0.0 = 0` identically — so the (already-fixed, for H_EC) production formula collapses
to the bare "keep" term regardless of whether `Lam_homog`/`denom_cf_scaled` are wired correctly at
all. The test's own (un-fixed, `pi_vec`-based) reference then only matched because this particular D4
fixture has `cf.usePMM=false`, making `wctx.pi_vec[cf.cf_col]==0` too — both sides were accidentally
zero, and the test provided **no actual discriminating power** over the exact bug this whole session
is about. Confirmed by direct grep: real production wiring
(`cm_hessian_architectures.jl:1305`, `hessian_cm_structured!`'s profiled branch) computes genuinely
nonzero `gpσ = gp^σ` (`gp=θ_full[3+D]`, `σ=θ_full[2]`) and `denom_cf = gpσ*ctx.γ.LPrime[bi]` from the
real calibration θ, never the `0.0` default.

Fixed by computing the same real `gpσ`/`denom_cf` this session's own production wiring uses at the top
of the test, passing them into all three `build_winner_pair_ctx` calls (H_EC/H_EZ/H_EF sections), and
switching every brute-force reference from `pi_vec` to `Lam_homog`/`denom_cf_scaled` with the correct
additive constant term. Added an explicit `check(... "is genuinely nonzero (test has real teeth)" ...)`
assertion in each section so a future accidental reversion to a vacuous default is caught immediately
rather than silently passing again. Re-run: all three sections now pass with `gpσ_france≈0.905`,
`denom_cf_france≈1.608` (genuinely nonzero), matching to machine/near-machine precision
(`max|Δ|` between `2.9e-15` and `1.3e-10`, see `key_results/`).

## Verification performed (and its limits — read before trusting)

**What this session verified, and how:**

- Every one of the four fixed functions' bilateral-column output matches an **independently
  re-derived** brute-force computation, built directly from raw `cf`/`Bidx`/`S`/`Z` data (NOT reusing
  `MSumX`/`T0_slot`/`TZ_buf`/`QCScum`/etc., so a bug shared between the reference and the code under
  test cannot hide) — `test_profiled_hef_correction_d4_2026-08-01.jl`,
  `test_profiled_hez_correction_d4_2026-08-01.jl`, both across `{anchored,orthonormal}` contrasts ×
  `L∈{10,20}` × `{calib,perturbed}` points where applicable. All PASS, `max|Δ|` in the `1e-15`–`1e-10`
  range (raw-data brute force at D4 scale, not bit-identical — same tolerance discipline the
  pre-existing H_EC/H_EZ tests already used).
- The France/cf row specifically (the second, independent bug even within the profiled-correction
  fix) is now verified with **genuinely nonzero** `gpσ`/`denom_cf`, not the previously-vacuous
  `0.0`/`0.0` defaults, for all three of H_EC/H_EZ/H_EF (`test_profiled_france_row_d4_2026-08-01.jl`).
- Threaded H_EZ (`winner_pair_cross_hessian_zc_block_threaded!`) remains bit-identical to the serial
  version at both `use_profiled_correction={true,false}`, across worker counts `{1,2,4}` and 3 random
  points, INCLUDING the France row (`test_profiled_hez_threaded_d4_2026-08-01.jl`, unmodified, still
  passes — now certifying two correct implementations agree, not two wrong ones).
- **Zero regression**: re-ran the full pre-existing D4 suite this session's edits touch or could
  affect (`test_profiled_hec_correction_d4_2026-08-01.jl`, `test_winner_pair_cross_hessian_cm_d4.jl`,
  `test_winner_pair_cross_hessian_zc_d4.jl`, `test_profiled_flexcm_d4_hessian_gate_2026-08-01.jl`) —
  all still ALL PASS, unchanged. `test_threaded_cross_hessian_d4.jl`'s common-Fréchet section throws
  `FieldError: type OperatorPsiBundle has no field moments!` — confirmed via a direct side-by-side run
  against the **unmodified** `architecture/profiled-economic-block-all-families-complete-2026-08-01`
  worktree that this is a **pre-existing failure at the forked-from HEAD**, not something introduced
  this session (same error, same location, before any of this session's edits).

**What this session explicitly did NOT verify, and should not be assumed correct on the strength of
the above alone** (this repo's own verification hierarchy — see CLAUDE.md-adjacent memory on
`lfd_ok`/hand-coded-reference-pairs-are-insufficient — explicitly says a hand-coded production/
reference pair is not, by itself, sufficient; H_EC's own bug was found only once the user pushed for
a ForwardDiff cross-check of the real production code path, not from its own initial brute-force
gate, which "passed" the same way these did before today):

- **No ForwardDiff check of the actual homogeneous dual objective** (mission's own explicit
  requirement #1 in its "verification hierarchy" section) was built for H_EF or H_EZ this session.
  Building one requires an element-type-generic (`Dual`-compatible) rewrite of the scalar `Ψ`
  contraction — production's `obj.Psi!`/`ddPsi!` are hardcoded `Vector{Float64}`-in-place kernels, not
  callable with `ForwardDiff.Dual` arguments — the exact same obstacle the master doc for H_EC
  describes overcoming ("writing element-type-generic copies of the moment-contraction functions").
  That work was not attempted this session; the brute-force checks above are real and independently
  derived, but they are the SAME tier of evidence H_EC's own tests provided before its bug was found,
  not the stronger tier that actually caught it.
- **No explicit dense reduced-G D4 Hessian gate** and **no real D4 KNITRO solve** for common Fréchet,
  ZC-only, or CM+ZC under a genuinely reduced/anchor-omitting economic layout — none of the three
  families have a reduced-layout orchestrator wiring (`hessian_cm_frechet_structured!`/
  `archA_partitioned_hess_cb_builder`-equivalent "gather" branch) at all yet; only flexible CM has one
  (built by the PRIOR session, unchanged by this one). This session fixed the underlying PRIMITIVE
  formulas those three families' future gather branches will need, but did not build the branches
  themselves.
- **`hessian_cm_structured_v2!`** (threaded flexible-CM) was not touched — still no profiled/gather
  branch, mechanical port from the now-confirmed-correct serial branch, not done.
- **Production operator FG (`:cm_lookup`)** wiring for any restricted family — untouched; this
  session, like the prior one, only closed correctness gaps in the `:dense_reference`-backend/
  `:winner_bin`-cross-Hessian-backend combination already used by the existing tests.
- **Genuine D20 omit-ROW reduced contexts** for common Fréchet / ZC-only / CM+ZC — not constructed;
  dual dimensions, bounds, initial values, names, verification slices, Hessian offsets, checkpoint
  metadata for these three families are unaudited (same status as the prior session's own accounting).
- **Full-formulation no-overhead gate** (allocation/timing) — not run.
- **Restriction-preparation audit** (mission's own explicit ask, "prove every profiled early branch
  prepares current restriction state before H_CC/H_CF/H_FF/H_CZ/H_ZZ"): grepped every
  `build_bin_tables!`/`prefix_sum_tables!` call site
  (`cm_hessian_architectures.jl:1293-1294,1345-1346`, `cm_hessian_threaded.jl:219-220`). The ONLY
  profiled/gather branch that currently exists at all (flexible CM's, `hessian_cm_structured!`) already
  has its own `build_bin_tables!(...; fill_S=false)`/`prefix_sum_tables!(...; fill_S=false)` call,
  fixed by the PRIOR session specifically because `fill_cm_HCC!` needs fresh `cctx.CT` — confirmed
  present and unchanged. There is currently no separate profiled/gather branch for H_CF/H_FF (common
  Fréchet's `_fill_frechet_level_blocks!`, which reads `cctx.CT`/`T1` built by the SAME
  `build_bin_tables!`/`prefix_sum_tables!` call as the non-profiled H_CC path since no profiled branch
  exists there yet) or H_CZ/H_ZZ (ZC families) to audit — this ask is not yet applicable until those
  gather branches are built, since the non-profiled path they'd fall back to already calls
  `build_bin_tables!`/`prefix_sum_tables!` unconditionally at the top of `hessian_cm_structured!`
  (`cm_hessian_architectures.jl:1345-1346`). No new checksum test was written since there is no new
  branch yet to write one against; the one existing branch's own prior-session fix was re-confirmed
  present, not re-derived.

## Deliverables

- This file (`PROFILED_REMAINING_CROSSBLOCKS_MASTER_2026-08-01.md`).
- Exact homogeneous cross-block derivation for H_EF/H_EZ's constant-term fix: see "What this session
  actually did" §1 above (structural analogy to H_EC's difference-formula fix, using
  `WinnerBinCrossScratch`/`WinnerZCCrossScratch`'s own pre-existing "sum"/"un-binned" twin fields).
- H_EF/H_EZ independent brute-force gates: PASS (see `key_results/`).
- France-row gates with genuinely nonzero `gpσ`/`denom_cf` (the second bug this session found, in the
  TEST not the production code): PASS (see `key_results/`).
- Zero-regression confirmation against the full pre-existing D4 suite this session's edits touch.
- Commits/ancestry: see `git log --oneline 6f6a878..HEAD` in this worktree.
- Raw logs: `key_results/*.log` (full stdout of every test run this session, including the pre-existing
  common-Fréchet threaded-test failure confirmed present at the unmodified fork point too).

## Final verdict (mission's own format)

```text
H_EC =
    corrected_and_independently_verified   # unchanged from the source branch (6f6a878) -- this
                                            # session did not touch winner_pair_cross_hessian_cm_block!

H_EF =
    Lam_homog_correct:            pass   # winner_pair_cross_hessian_colsum!/_esum!, fixed + brute-force verified
    France_constant_correct:      pass   # denom_cf_scaled*sumNu / denom_cf_scaled*t0, fixed + verified with
                                          # genuinely nonzero gpσ/denom_cf (test itself was fixed from vacuous)
    ForwardDiff:                  not_run   # see "What this session explicitly did NOT verify" -- the
                                             # mission's own stated top verification tier, not attempted
    D4_KNITRO:                    not_run   # no reduced-layout orchestrator/gather branch exists yet
                                             # for common Fréchet to solve through

H_EZ =
    Lam_homog_correct:            pass   # winner_pair_cross_hessian_zc_block!/_threaded!, fixed + verified
    France_constant_correct:      pass   # same as H_EF, both serial and threaded, genuinely nonzero test
    serial_ForwardDiff:           not_run
    threaded_ForwardDiff:         not_run
    ZC_only_D4_KNITRO:            not_run   # no reduced-layout gather branch exists yet for ZC-only
    CM_plus_ZC_D4_KNITRO:         not_run   # same, CM+ZC

PRODUCTION_FG =
    fail_common_frechet   # :cm_lookup not wired for any restricted family (unchanged from prior session)
    fail_ZC_only
    fail_CM_plus_ZC
    # flexible_cm: :dense_reference-only closure from the prior session, also not :cm_lookup -- see
    # that session's own PRODUCTION_FG verdict, unchanged here.

DENSE_ECONOMIC_G_MATERIALIZATIONS =
    not_audited_this_session   # no new production FG path was wired; existing counters unchanged

OPTIMIZED_KERNEL_REUSE =
    existing_full_width_then_gather   # unchanged design (prior session) -- this session only fixed the
                                       # PRIMITIVE formulas the future H_EF/H_EZ gather branches will call;
                                       # no new gather branch was written this session

OLD_FULL_PATH_OVERHEAD =
    none_expected_not_remeasured   # use_profiled_correction=false byte-for-byte unchanged in every
                                    # touched function (regression-verified); allocation/timing not
                                    # independently re-profiled this session

D20_SMALLW =
    not_run   # all families

PRODUCTION_DEFAULT_CHANGED = false
CAMPAIGN_LAUNCHED = false
```

## Recommended next steps (in order, for whoever continues this)

1. Build the ForwardDiff-of-the-actual-homogeneous-objective gate for H_EF and H_EZ that this session
   did not attempt — the single highest-value remaining gap, since it is the ONLY tier of evidence
   that actually caught H_EC's original bug (the brute-force tier this session used did NOT catch
   H_EC's bug on its own first pass either). Requires an element-type-generic rewrite of the scalar
   `Ψ` contraction (production's `obj.Psi!`/`ddPsi!` are `Vector{Float64}`-hardcoded); the existing
   `homogeneous_dual_contraction`/`homogeneous_transpose_contraction!`
   (`homogeneous_contraction_2026-07-31.jl`) already implement the FULL (unreduced) economic-block `E`
   formula generically enough to reuse as a starting derivation, though they too are `Float64`-typed
   as written and would need a small `Dual`-compatible variant.
2. Build the reduced-layout Hessian-orchestrator "gather" branch for common Fréchet
   (`hessian_cm_frechet_structured!`), mirroring flexible CM's own branch in
   `hessian_cm_structured!` — this is what actually lets a genuinely reduced common-Fréchet KNITRO
   solve be attempted at all; H_EF's formula is now correct, but nothing currently calls it from a
   reduced context.
3. Build the equivalent gather branch for ZC-only/CM+ZC (`archA_partitioned_hess_cb_builder` or
   wherever that family's Hessian callback lives) — structurally analogous but a separate codebase
   surface, per the prior session's own note.
4. Port the confirmed-correct serial `hessian_cm_structured!` profiled branch into
   `hessian_cm_structured_v2!` (threaded flexible-CM) — mechanical now, not yet done.
5. Wire `:cm_lookup` (the TRUE production inner-FG-backend default) for at least flexible CM before
   claiming any of this is production-ready — still entirely unaddressed across both this session and
   the prior one.
6. Only after (1)-(5): D20 omit-ROW reduced contexts, D20 small-W gates, no-overhead gate, per the
   mission's own ordering.
