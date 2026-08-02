# Profiled restricted-family outer-gradient layer — master report (2026-08-01)

## Mission recap

Parallel workstream to the inner restricted-family pipeline (`architecture/profiled-restricted-inner-endtoend-2026-08-01`
or its descendant). This branch owns profiled outer-coordinate decoding, the shared economic A/gp
fixed-dual gradient, family adapters, the gravity-pivot chain rule, outer-gradient caches, gradient
verification, and the matched outer-search A/B harness — nothing inside inner FG/Hessian/cross-block/
restriction-kernel code.

Branch: `architecture/profiled-restricted-outer-gradient-2026-08-01`, based on `f439109`
(`diagnostic/profiled-scales-unrestricted-outer-ab-2026-08-01`'s tip). See
`PROFILED_RESTRICTED_OUTER_GRADIENT_SOURCE_SNAPSHOT_2026-08-01.md` for the full base-commit
verification.

## 1. What was built

| File | Role |
|---|---|
| `full_aod_diag/d4_exact/profiled_outer_gradient_layout_contract_2026-08-01.jl` | Five-accessor interface + `validate_family_layout_contract` + `structural_checksum` |
| `full_aod_diag/d4_exact/profiled_lfix_incremental_2026-08-01.jl` (edited) | Extracted `build_price_winner_base_cache` (pure code motion) and `profiled_composite_gradient_from_cache` (pure code motion) out of `build_profiled_lfix_cache`/`profiled_composite_gradient_at_incremental`, so the shared engine can reuse both without duplicating the winner-update mechanism. Zero formula change — proved by the unrestricted regression gate below. |
| `full_aod_diag/d4_exact/profiled_shared_economic_gradient_engine_2026-08-01.jl` | `build_shared_profiled_lfix_cache` (contract-driven generalization of `build_profiled_lfix_cache`) + `shared_family_outer_gradient` + `assert_shared_gradient_method_identity` |
| `full_aod_diag/d4_exact/profiled_family_adapters_2026-08-01.jl` | `UnrestrictedFamilyCtx` (real) + `MockRestrictedFamilyCtx` (synthetic, four families) |
| `full_aod_diag/d4_exact/profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl` | Independent, non-incremental full-rebuild fixed-dual reference generalized to any contract-satisfying family, diagnostic namespace (`diag_` prefix) |
| `full_aod_diag/d4_exact/PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl` | Family-generic profiled outer KNITRO search, one shared gradient call site |
| `full_aod_diag/d4_exact/test_profiled_layout_contract_interface_2026-08-01.jl` | 8 interface tests (all pass) |
| `full_aod_diag/d4_exact/test_profiled_shared_engine_unrestricted_regression_2026-08-01.jl` | Real-KNITRO D4/D20 regression (bit-identical) |
| `full_aod_diag/d4_exact/test_profiled_mock_family_gate_2026-08-01.jl` | Mock-family numerical gate (machine precision) |
| `full_aod_diag/d4_exact/test_profiled_ab_harness_smoke_2026-08-01.jl` | Harness mechanical smoke test |

## 2. Gate results (all re-run live this session, not carried over from any prior claim)

### 2a. Interface tests (`PROFILED_RESTRICTED_OUTER_GRADIENT_PREINTEGRATION_GATE_2026-08-01.csv`)

8/8 PASS: no-hard-coded-offset, restriction-range-disjoint-and-after-economic,
validate-passes-well-formed, incorrect-checksum-throws, wrong-dual-length (validator and cache
builder both), anchor-coordinate-cannot-appear, shared-gradient-method-identity.

### 2b. Unrestricted regression, real KNITRO

- **D4** (`PROFILED_UNRESTRICTED_SHARED_ENGINE_REGRESSION_2026-08-01_D4.csv`): `max_abs_err=0.0`,
  `cos_sim=1.0` at calibration and a random small perturbation. `h_used`/`switch_mass`/`cache.q0`
  bit-identical between the shared engine and the pre-refactor code.
- **D20, real `:exclude_row` data, W=20,000** (`PROFILED_UNRESTRICTED_SHARED_ENGINE_REGRESSION_2026-08-01_D20_W20000.csv`):
  same result, `max_abs_err=0.0` at both points. This is the strongest evidence the refactor changed
  nothing about the unrestricted family: a real 361-dimensional profiled outer gradient from a real
  KNITRO solve, bit-for-bit identical before and after.

### 2c. Mock restricted-family gate (`PROFILED_RESTRICTION_MOCK_FAMILY_GATE_2026-08-01.csv`)

9/9 PASS across `flexible_CM`/`common_Frechet`/`ZC_only`/`CM_plus_ZC`:

- **(A) code-level no-coupling**: holding `restriction_contrib0` fixed while randomizing the raw
  restriction-`beta` slice by 1,000,000× leaves the A/gp gradient **bit-identical** — the shared
  engine provably never reads that slice directly.
- **(B) shared engine vs. independent full-rebuild reference, matched per-coordinate bandwidth**: A-block
  (11 coordinates) `max_rel_err` between `3.3e-15` and `5.5e-15` — genuine machine precision. `gp`
  (analytic in the shared engine, FD in the reference regardless of bandwidth) differs by
  `2e-7`–`4.8e-6`, matching the pre-existing analytic-vs-FD gap already present in the unrestricted
  family's own gate.
- **(sanity)** confirms `restriction_contrib0`'s *magnitude* does change the gradient baseline — see
  the boundary-of-claim section below.

### 2d. D4 / D20-small-W gates, restriction-parameter regression, performance gate

See `PROFILED_RESTRICTED_OUTER_GRADIENT_D4_GATE_2026-08-01.csv`,
`PROFILED_RESTRICTED_OUTER_GRADIENT_D20_SMALLW_GATE_2026-08-01.csv`,
`PROFILED_RESTRICTION_PARAMETER_GRADIENT_REGRESSION_2026-08-01.csv`,
`PROFILED_RESTRICTED_OUTER_GRADIENT_PERFORMANCE_GATE_2026-08-01.csv`. Unrestricted rows are real,
live results; restricted-family rows are either the mock-gate result (formula/mechanism validation,
labeled explicitly) or `BLOCKED_PENDING_INNER` (no live restricted-family inner context exists yet)
— never a fabricated pass.

Performance: shared-engine refactor adds **~0.05 MiB (0.2%)** allocation over the pre-refactor
unrestricted call (from `validate_family_layout_contract`'s checksum), and is 2.2× **faster** wall
time at D4 warm (0.067s vs 0.150s — within normal JIT/measurement noise, not a claimed real
speedup). Zero dense economic-`G` materializations anywhere in this branch's new code (grep-confirmed).

### 2e. Outer A/B harness

Family-generic; unrestricted arm smoke-tested (D4, `maxit_override=8`): harness wiring confirmed
correct, `gp` never entered the KNITRO free-variable set, both PASS checks green. The four
restricted-family arms throw a loud, explicit "not ready" error rather than silently falling back to
the unrestricted evaluator. **No campaign was launched** — see
`PROFILED_ALL_FAMILY_OUTER_AB_PROTOCOL_2026-08-01.md`.

## 3. A genuine finding: an inherited "machine precision" claim did not reproduce as committed

While building the full-rebuild reference, I re-ran the pre-existing, already-committed
`test_profiled_incremental_vs_fullrebuild_2026-08-01.jl` gate script (from the base commit, not
authored by this branch) to use as ground truth. Its own output —
`cos_sim=0.9998982414935192`, `max_rel_err=2.048` at calibration — is **unchanged** from what was
already recorded in the repo's own `PROFILED_INCREMENTAL_VS_FULLREBUILD_2026-08-01_D4.csv` before I
touched anything (confirmed via `git diff`, substantive fields byte-identical, only wall-clock timing
columns changed run-to-run). What does **not** match is the prose elsewhere:
`PROFILED_UNRESTRICTED_OUTER_AB_MASTER_2026-08-01.md` and
`PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md` both describe this exact comparison as
"D4 cosine similarity 1.0000000000 (max rel err ~6e-6) ... machine precision." That claim is not what
the committed gate script's own output shows, and is not what re-running it produces.

**Root cause, isolated directly (not guessed)**: `profiled_composite_gradient_at` (the full-rebuild
comparator in that file) uses one **fixed** `h=0.01` for every coordinate; the incremental method
(`profiled_composite_gradient_at_incremental`/the shared engine) uses an **adaptive**, per-coordinate
`h` from `profiled_select_bandwidth`. Comparing the two at mismatched bandwidths reproduces exactly
the observed `cos_sim≈0.9999` gap. Re-running the identical comparison at a **matched** bandwidth
(each coordinate's own adaptively-selected `h`, reused in both arms) collapses the gap to `~1e-16` —
genuine machine precision (verified directly, see §2c above, which is exactly this comparison
generalized to the restricted-family mock case). This is very likely what the earlier "machine
precision" claim actually meant to describe (a matched-bandwidth comparison), but that is not what
the currently-committed script computes or what its own output file records. Flagging this here per
this repo's own standing culture of catching and correcting exactly this kind of gap between a
narrative claim and the artifact that supposedly backs it — **not** asserting the earlier session did
anything in bad faith; the underlying formula is correct (my own from-scratch independent reference
confirms it), only the comparator's bandwidth choice was mismatched.

**Recommendation for whoever next touches
`test_profiled_incremental_vs_fullrebuild_2026-08-01.jl`/its master-doc claim**: either (a) update
the comparator to accept the incremental method's own selected `h` per coordinate (small, mechanical
change, exactly mirroring `diag_profiled_full_rebuild_gradient`'s `h::Union{Float64,AbstractVector{Float64}}`
signature added in this branch), or (b) correct the prose to describe the actual `cos_sim≈0.9999`
result instead of a `1.0000000000` figure the script does not produce.

## 4. On the restriction-invariance claim (task §3/§11) — the exact scope that was tested

Two claims are easy to conflate; only one is true, and only that one is claimed here:

- **TRUE, tested (§2c-A)**: at a FIXED restriction contribution, changing what's stored in the
  restriction-labeled `beta` slice has zero effect on the A/gp gradient (the shared engine never
  reads that slice for anything other than via the caller-supplied `restriction_contrib0`).
- **NOT true in general, tested and documented as such (§2c-sanity)**: the A/gp gradient's *value*
  does depend on `restriction_contrib0`'s magnitude, because `Psi` (the smoothed complementarity
  function) is nonlinear — a different fixed baseline in `q` genuinely shifts a finite-difference
  estimate taken around it. The task text's phrasing ("the restriction-dual contribution is constant
  across +/- probes and cancels from the central difference") is correct as a statement about *how*
  the restriction contribution enters the incremental cache (once, as a coordinate-independent
  additive term in `q0`, never revisited during the coordinate loop) — it is not a claim that the
  resulting gradient value is independent of that constant, and this report does not claim that.

## 5. File-ownership boundary

The only edit to a pre-existing file is `full_aod_diag/d4_exact/profiled_lfix_incremental_2026-08-01.jl`
— two pure extractions (`build_price_winner_base_cache`, `profiled_composite_gradient_from_cache`),
zero formula changes, proved by the bit-identical unrestricted regression. No file on the forbidden
list (`winner_pair_cross_hessian.jl`, `threaded_cross_hessian.jl`, `core_exact_hessian.jl`,
`cm_hessian_threaded.jl`, `cm_hessian_architectures.jl`, `zc_gram_blas_candidates.jl`, restriction
moment constructors, reduced economic FG/Hessian primitives, family inner callback builders) was
read for anything beyond understanding the interface this branch needs to consume, and none was
edited.

## 6. What remains, honestly

Everything downstream of "a live restricted-family inner context exists" is blocked on the inner
workstream, by design (task's own explicit sequencing). The mock-family gates establish that the
shared engine's *mechanism* is correct and ready; they cannot and do not establish that any real
restricted family's *specific* restriction moments are genuinely A/gp-invariant in the inner
workstream's actual implementation — that must be re-verified once real contexts are available (see
`PROFILED_RESTRICTED_OUTER_GRADIENT_INTEGRATION_2026-08-01.md`).

## 7. Verdict block

```text
INNER_FILES_MODIFIED = none

SHARED_ECONOMIC_A_GP_GRADIENT = one_method_all_families
  (profiled_composite_gradient_from_cache; assert_shared_gradient_method_identity PASS;
   verified via methods() returning exactly one applicable method)

UNRESTRICTED_REGRESSION = pass
  (D4 and real D20/W=20000, bit-identical, max_abs_err=0.0)

FAMILY_ADAPTERS =
  flexible_CM:pass_mock
  common_Frechet:pass_mock
  ZC_only:pass_mock
  CM_plus_ZC:pass_mock
  (all four MOCK -- real inner contexts not yet available; see INTEGRATION doc)

GRAVITY_PIVOT_CHAIN_RULE = pass
  (reused unchanged from gravity_pivot_on_retained_2026-07-31.jl; exercised in every
   gradient call via profiled_composite_gradient_from_cache -> outer_dim_profiled(pe)
   pathway; no separate family-specific pivot written)

D4_FIXED_DUAL_GATE = pass_unrestricted_mock_restricted
  (unrestricted: real, bit-identical to pre-refactor; restricted: mock, machine
   precision vs matched-bandwidth full-rebuild reference; live restricted D4 contexts
   blocked pending inner)

D20_SMALLW_GATE = pass_unrestricted_only
  (real D20/:exclude_row/W=20000, bit-identical; restricted families
   blocked_pending_inner)

RESTRICTION_PARAMETER_GRADIENTS = unchanged
  (proof by construction: shared_family_outer_gradient's output length is always
   exactly outer_dim_profiled(pe), structurally cannot contain a restriction-parameter
   slot; restriction beta slice never read directly, confirmed by randomization test;
   no live family-specific restriction-parameter gradient exists yet to regress
   bitwise against, blocked pending inner)

DENSE_ECONOMIC_G_MATERIALIZATIONS = 0
  (grep-confirmed over every new file in this branch)

OUTER_AB_HARNESS = ready
  (family-generic; unrestricted arm smoke-tested and working; four restricted arms
   are explicit not-ready placeholders, never silent fallbacks)

PRODUCTION_DEFAULT_CHANGED = false
CAMPAIGN_LAUNCHED = false
```
