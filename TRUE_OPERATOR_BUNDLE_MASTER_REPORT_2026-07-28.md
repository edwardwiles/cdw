# True no-H operator bundle: master report (2026-07-28)

Branch: `work/true-operator-bundle-no-H-2026-07-28`, isolated worktree, based on inherited
`cleanup/remove-legacy-CC-H-G-storage-2026-07-28@84a4198`. A separate real campaign
(`worktrees/campaign-five-family-bounds-2026-07-28`, confirmed via live `/proc/*/cwd` PID
`3765858` running `verify_direction_wiring.jl` at session start) was never touched, read, or
disturbed.

## Executive summary

**Part A (true no-H operator bundle): COMPLETE for all five families.** `OperatorPsiBundle`
(no `H`/`H_copy`/`K`/`ones`/`moments!`/`jac_h` field, structurally -- not merely empty) is now the
production bundle type for unrestricted, flexible CM, common Fréchet, CM+ZC, and ZC-only,
validated by real KNITRO gates at both D=4 (synthetic, exact 0.0 agreement on every quantity
checked) and real D=20/W=100,000 production data (exact agreement on the full inner solve's
accepted status, delta*, and complete dual vector). Wiring the four families beyond flexible CM
surfaced three real, pre-existing latent bugs (none introduced by this bundle), all found and
fixed, each verified by the gate scripts' own structural checks failing loudly and unambiguously
(clean `FieldError`/`MethodError`, never silent wrong answers).

**Part B (gravity moment resolution): RESOLVED.** The earlier session's "0.003 residual" finding
was a diagnostic artifact (testing `CS.reconstruct_full` in isolation, bypassing the real
production pivot decoder), confirmed and root-caused via a real 100-point numerical test against
the actual pivot decoder (`pivot_expand`, as used by `run_cm_upper_checkpointed`). The inner
gravity moment is retained (not removed) because it remains the only enforcement mechanism
visible to the shared family-level entry points this task touches, which have no way to know
whether their caller already pivoted -- a scoped, conservative resolution, not a blanket removal.

## Part A: what was built

### The bundle itself

`full_aod_diag/d4_exact/operator_psi_bundle.jl` defines `OperatorPsiBundle` -- a flat struct
carrying the same control-flow scalar fields `PsiObjectiveBundleImplicit` has (needed because the
shared Hessian/verification code across all five families does unconditional flat-field reads,
`@unpack M, arg0, arg2, ddPsi! = obj`/bare `obj.Psi!`, for BOTH bundle types), but genuinely absent
`H`, `H_copy`, `moments!`, `jac_h`. `K` is renamed `payoff` (the forbidden-field list names the
literal field name `K`, not just the legacy `[K|ones|G]` layout). `economic_state`/
`restriction_state` are added as bundle-owned state HANDLES -- aliases into the same shared boxes
(`cctx.core_cf_ref`, `cctx`/`octx` itself) the existing Hessian/verification plumbing already
reads, not duplicated storage.

`prime_operator!(obj, θ_econ, ctx, core_cf_ref; restriction_state)` is the single shared priming
function used by flexible CM, common Fréchet, CM+ZC, and ZC-only: builds the economic
compressed-factual state (`cf_build`), fills `payoff` and the gravity column directly (O(W), never
O(W*d)), publishes `core_cf_ref[] = cf`. Unrestricted uses its own inline analog (see below) rather
than this function directly, because its own established convention differs (a bare scalar
`grav_raw`, not a materialized gravity column).

### Per-family wiring

Each family's `build_*_production_context` (or, for unrestricted, a standalone wrapper) gained a
`moment_representation::Symbol` kwarg (`:operator`, production default | `:dense_reference`,
explicit reference-gate opt-in). This is a genuine type-level branch in the CONSTRUCTOR/factory
function, not a runtime flag inside a shared bundle -- `OperatorPsiBundle` itself has zero
knowledge of "dense_reference"; the factory chooses which of two entirely different concrete
types to build.

Priming dispatches at the actual call site (`inner_loop_internal_*_production`/`_operator`) on
`obj isa OperatorPsiBundle`, calling `prime_operator!` (or the unrestricted inline analog) instead
of the legacy `moments_fn(...)`/`obj.H[:,1]`/`select_G_from_H` sequence.

### Real bugs found and fixed while generalizing (all pre-existing, none introduced by this bundle)

1. **`archC_frechet_hess_cb_builder`'s serial branch** (`cm_frechet_hessian.jl`) hardcoded the
   dense-only `_archC_prep_for_hessian!` instead of the shared dense-G-free
   `_prep_dual_index_for_archC!` dispatcher -- its own THREADED branch, and flexible-CM's
   analogous `archC_hess_cb_builder` in BOTH branches, already used the correct dispatcher. This is
   a real, pre-existing disconnection between common Fréchet's own duplicated Hessian-callback
   code and flexible CM's shared one (see the architecture note below) -- not present in flexible
   CM itself, and not something this task's bundle work introduced.
2. **`build_bin_tables_threaded!`** (`cm_hessian_threaded.jl`, genuinely shared infrastructure used
   by flexible CM's own production-default threaded path) had a strict
   `H::AbstractMatrix{Float64}` method signature -- a `MethodError` on `nothing`, at DISPATCH time,
   even though `H`'s contents are never read when `fill_S=false`. Fixed to
   `Union{Nothing,AbstractMatrix{Float64}}` with the same explicit fail-fast guard the serial
   `build_bin_tables!` already had.
3. **`archA_partitioned_hess_cb_builder`** (origin-ZC) unconditionally unpacked `H`, `H_copy`, AND
   a third, previously-unnoticed dense-bundle-only scratch field (`∂∂f_∂∂x`, used purely as
   per-call n×n scratch, never relying on a prior value). `H`/`H_copy` fixed via the same
   `_dense_H_or_nothing`/`_dense_H_copy_or_nothing` dispatch pattern; the scratch field replaced
   entirely with a closure-owned persistent matrix (built once per KNITRO solve, not per callback
   invocation), used identically by both bundle types -- removing the dependency rather than
   working around it.

Each of these was latent and harmless against a real dense `H` (a redundant read, or an unused
unpack) -- none is a correctness bug in the existing dense-reference production path. All three
were caught immediately and unambiguously by the gate scripts' own structural proof (a clean
Julia `FieldError`/`MethodError`, not a silent wrong numerical answer), exactly the safety property
a genuinely-absent field is supposed to provide.

### Architecture note (user-requested)

Flexible CM and common Fréchet do **not** share one parameterized Hessian-callback implementation
the way an idealized architecture would (`[E | CM | CM-F]`, one set of functions differing only in
the block constructing the CM-F/level piece). Common Fréchet has its own separate
`hessian_cm_frechet_structured!`/`_v2!`/`archC_frechet_hess_cb_builder`
(`cm_frechet_hessian.jl`/`cm_frechet_hessian_threaded.jl`) -- structurally parallel to flexible
CM's `hessian_cm_structured!`/`_v2!`/`archC_hess_cb_builder`, but genuinely duplicated code, not a
shared function with a swapped-in level-block piece. Bug #1 above is a direct, concrete
consequence of this: a fix applied to flexible CM's shared function did not automatically apply to
Fréchet's separate copy, because there is no shared function to fix once. Unifying these into one
shared, parameterized implementation is a real, identifiable architecture improvement -- out of
scope for this task (which is about the bundle storage layer, not the Hessian-assembly call
graph), but is the concrete next refactor to prevent this exact class of duplication-drift bug
systematically rather than catching each instance one at a time, as this session had to.

### Gate results

See `FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md` and `FINAL_FIVE_BY_SEVEN_TRUE_OPERATOR_MATRIX_2026-07-28.md`/`.csv`
for the full per-family table. Summary:

```
D=4 (synthetic, real KNITRO):            ALL FIVE FAMILIES PASS, exact (0.0) agreement
D=20/W=100,000 (real production data):   ALL FIVE FAMILIES PASS, exact (0.0) agreement
```

Every family's real full inner solve (both the dense-reference and operator bundles, from the
identical calibration outer point, D=20/W=100,000 production data) reached the SAME accepted
KNITRO status and the SAME converged dual point to machine precision. This is the direct
machine-precision solution-equivalence proof: the no-H bundle yields the identical solution to the
dense reference, on real data, in every family.

## Part B: gravity moment resolution

Full algebra trace: `GRAVITY_PIVOT_VS_GRAVITY_MOMENT_ALGEBRA_2026-07-28.md`.
Full decision + evidence: `GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md`.
Real 100-point test: `GRAVITY_PIVOT_VALID_COORDINATE_TESTS_2026-07-28.csv`.

Summary: `gravity_elimination.jl`'s pivot (`pivot_expand`) is a genuine, exact elimination -- it
nulls the outer gravity equality for ANY choice of free coordinates, not just calibration
(confirmed algebraically and by 100 real perturbations: max residual 1.0e-17). `CS.reconstruct_full`
itself is a plain per-coordinate scatter with zero gravity awareness (confirmed: the only
definition in the repo). The REAL production driver (`run_cm_upper_checkpointed`) always calls
`pivot_expand` before ever constructing the vector it hands to `reconstruct_full`; the earlier
session's finding tested `reconstruct_full` in isolation, bypassing that step -- an invalid-point
diagnostic, not a pivot bug (100 naive-bypass perturbations: max residual 1.2e-3, reproducing the
earlier finding's own qualitative pattern with the SAME bypassed-pivot mechanism).

**Verdict: `same_equality_exactly_eliminated`, scoped.** The inner gravity moment is retained (not
removed) in the family-level entry points this task's Part A touches (`archC_base_state` and
siblings, `OperatorPsiBundle`'s own priming) because those entry points have no way to know
whether their caller already pivoted -- removing the moment there would silently drop gravity
enforcement for any future caller that doesn't route through `pivot_expand` first. The narrower,
concrete next step (named, not done this session) is: add a scalar outer-reconstruction assertion
to `run_cm_upper_checkpointed`'s own provably-pivoted inner solve specifically, then remove the
now-redundant inner moment there alone.

## Deliverables

- `TRUE_OPERATOR_BUNDLE_MASTER_REPORT_2026-07-28.md` (this document)
- `FLEXIBLE_CM_NO_H_BUNDLE_GATE_2026-07-28.md`
- `FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`
- `OPERATOR_VS_DENSE_INNER_SOLVE_EQUIVALENCE_2026-07-28.csv`
- `OPERATOR_BUNDLE_FIELD_AND_ALLOCATION_PROOF_2026-07-28.json`
- `GRAVITY_PIVOT_VS_GRAVITY_MOMENT_ALGEBRA_2026-07-28.md`
- `GRAVITY_PIVOT_VALID_COORDINATE_TESTS_2026-07-28.csv`
- `GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md`
- `FINAL_FIVE_BY_SEVEN_TRUE_OPERATOR_MATRIX_2026-07-28.md` + `.csv`
- `SHA256_MANIFEST_true_operator_bundle_2026-07-28.txt`
- 8 real KNITRO gate scripts (D=4) + 5 real KNITRO gate scripts (D=20/W=100,000), all in
  `full_aod_diag/d4_exact/test_operator_no_H_bundle_equivalence_*.jl`

## Final verdict block

```
OPERATOR_BUNDLE =
    unrestricted:OperatorPsiBundle
    flexible_cm:OperatorPsiBundle
    common_frechet:OperatorPsiBundle
    cm_plus_zc:OperatorPsiBundle
    zc_only:OperatorPsiBundle

LEGACY_H_FIELDS_IN_PRODUCTION = none
LEGACY_H_ALLOCATIONS_IN_PRODUCTION = 0
PRODUCTION_MOMENTS_CALLS = 0
PRODUCTION_SELECT_G_CALLS = 0

OPERATOR_DENSE_EQUIVALENCE =
    fixed_point_callbacks:pass_exact_all_five_families_d4_and_d20
    complete_inner_solves:pass_exact_all_five_families_d4_and_d20

GRAVITY_RELATIONSHIP = same_equality_exactly_eliminated
GRAVITY_RESIDUAL_ON_VALID_PIVOT_POINTS = machine_zero_all

FIVE_FAMILY_ARCHITECTURE = complete_true_operator_no_H

PRODUCTION_MERGE = port_ready_waiting_for_campaign
    -- a separate real campaign (worktrees/campaign-five-family-bounds-2026-07-28) was confirmed
       active (live PID) at session start and throughout -- not merged, not pushed, per this
       project's standing "confirm before pushing to production" requirement, which also applies
       independent of campaign status.
```
