# Five-family no-H bundle gate (2026-07-28) — FINAL: ALL FIVE FAMILIES PASS

> **CORRECTION (2026-07-29):** the `unrestricted: WIRED_AND_GATED` line below is misleading and was
> found FALSE live on 2026-07-29. What this document's own equivalence-gate scripts
> (`test_operator_no_H_bundle_equivalence_unrestricted[_d20].jl`) validated is real: a companion
> `OperatorPsiBundle` built from `ctx.obj`'s scalar fields agrees with the dense reference to machine
> precision, both at D=4 and real D=20/W=100,000. What did NOT happen, despite the "WIRED_AND_GATED"
> label: neither production driver (`c10_d20_production_driver.jl` /
> `c10_d20_production_driver_unified.jl`) ever actually called any such conversion -- `ctx.obj`
> stayed the dense `PsiObjectiveBundleImplicit` in every real solve for this family, unlike the other
> four, which genuinely do construct `OperatorPsiBundle` inside their own
> `build_*_production_context`. "The equivalence gate passes" and "the production driver uses the
> gated type" are different claims; this doc conflated them for unrestricted only. Fixed 2026-07-29
> (see `full_aod_diag/d4_exact/compressed_live.jl::build_unrestricted_operator_ctx`, wired into
> `run_polish_checkpointed_unified`) -- unrestricted's production driver now genuinely defaults to
> `OperatorPsiBundle`, matching what this document always claimed.
>
> **SECOND CORRECTION (2026-07-29):** the same conflation applies to `cm_plus_zc` (cm_meanzc) and
> `zc_only` (origin_zc) below, found while investigating a report that origin_zc's production runs
> still used the dense bundle. Confirmed live: `build_cm_meanzc_production_context`
> (`cm_meanzc_production.jl`) and `build_originzc_production_context`
> (`cm_originzc_production.jl`) both hardcoded `moment_representation::Symbol = :dense_reference`
> as their DEFAULT, and their real production drivers (`run_cm_upper_checkpointed`'s `is_meanzc`
> branch, `run_originzc_upper_checkpointed`) never passed the kwarg at all -- so, exactly like
> unrestricted, `ctx.obj` stayed dense in every real solve for these two families despite this
> doc's "WIRED_AND_GATED" label and despite both families' own standalone equivalence gates
> genuinely passing (they call the builder with the kwarg passed EXPLICITLY, which the real driver
> never did). `common_frechet` is the one exception among the four `run_cm_upper_checkpointed`
> families that was NOT false: its own default was independently flipped and validated end-to-end
> through the real driver in a separate 2026-07-29 task (`FRECHET_OPERATOR_DEFAULT_INVESTIGATION_
> 2026-07-29.md`) before this one started. `flexible_cm` was also never false (resolves via the
> shared `MOMENT_REPRESENTATION[]` global, independently confirmed `:operator` and never mutated in
> any production code path). Fixed 2026-07-29 for cm_meanzc and origin_zc (kwarg threaded through
> both drivers with a `nothing`-sentinel default so it never silently overrides either builder's own
> resolved default; `moment_representation` defaults flipped to `:operator` in both builders,
> validated D=4 + real D=20/W=100,000 through both real drivers end-to-end,
> `test_meanzc_originzc_driver_wiring_2026-07-29.jl`) -- all five families now genuinely default to
> `OperatorPsiBundle` in production, matching what this document always claimed.

## Status

```
unrestricted:    WIRED_AND_GATED  (OperatorPsiBundle, D=4 + real D=20/W=100,000, exact agreement)
flexible_cm:     WIRED_AND_GATED  (OperatorPsiBundle, D=4 [serial+threaded] + real D=20/W=100,000)
common_frechet:  WIRED_AND_GATED  (OperatorPsiBundle, D=4 + real D=20/W=100,000)
cm_plus_zc:      WIRED_AND_GATED  (OperatorPsiBundle, D=4 + real D=20/W=100,000)
zc_only:         WIRED_AND_GATED  (OperatorPsiBundle, D=4 + real D=20/W=100,000)
```

All five families now construct the genuine no-H `OperatorPsiBundle` in production mode
(`moment_representation=:operator`), validated against the unchanged dense reference
(`moment_representation=:dense_reference`, still `PsiObjectiveBundleImplicit`) at both D=4
(synthetic) and real production-scale D=20/W=100,000 data. This supersedes the earlier
mid-session draft of this document (flexible-CM only) once the user asked for -- and this session
delivered -- the full generalization to all five families, not a scoped-down subset.

## D=4 real-KNITRO results (all exact, 0.0 abs-diff)

| family | structural proof | FG/Hessian callbacks | full inner solve |
|---|---|---|---|
| unrestricted | PASS | exact | exact |
| flexible_cm (serial) | PASS | exact | exact |
| flexible_cm (threaded, production default) | PASS | n/a (delta*/dual only) | exact |
| common_frechet | PASS | exact | exact |
| cm_plus_zc | PASS | exact | exact |
| zc_only | PASS | exact | exact |

## Real D=20/W=100,000 results (production Sobol data, destination_sample=:exclude_row, L=50)

| family | dense nStatus | operator nStatus | delta*/dual agreement | dense solve time | operator solve time |
|---|---|---|---|---|---|
| unrestricted | 0 | 0 | exact (0.0) | 6.6s | 1.8s |
| flexible_cm | -103 | -103 (matching) | exact (0.0) | 25.4s | 9.9s |
| common_frechet | 0 | 0 | exact (0.0) | 25.5s | 9.7s |
| cm_plus_zc | 0 | 0 | exact (0.0) | 25.9s | 11.1s |
| zc_only | 0 | 0 | exact (0.0) | 15.1s | 5.5s |

Every family's operator-bundle inner solve reached the SAME accepted KNITRO status as the dense
reference and the SAME converged dual point to machine precision (both |delta zeta*| and
max|delta lambda*| exactly 0.0 in every case) -- this is the direct answer to "does the no-H
bundle yield the same solution as the dense reference on real data": yes, in every family tested,
to machine precision. Incidentally, the operator-bundle solve was also consistently 2-3x faster
wall-clock than the dense-reference solve at this scale in every family -- not the point of this
task, but a real, observed side effect worth recording (not independently investigated/attributed
to a specific mechanism this session; the dense-reference path retains legacy per-callback work
this bundle type structurally cannot do).

## What was built, per family

Each family's `build_*_production_context` function gained a `moment_representation` kwarg
(`:operator`, production default | `:dense_reference`, explicit reference-gate opt-in), mirroring
flexible CM's own pattern:
- `build_cm_production_context` (flexible CM)
- `build_cm_frechet_production_context` (common Frechet)
- `build_cm_meanzc_production_context` (CM+ZC)
- `build_originzc_production_context` (ZC-only)
- a standalone wrapper for unrestricted (this family has no separate production-context
  builder -- `ctx.obj` IS the production bundle directly; the wrapper constructs a companion
  `OperatorPsiBundle` from its scalar fields, matching the same pattern)

Priming dispatches on `obj isa OperatorPsiBundle` to `prime_operator!` (flexible CM, common
Frechet, CM+ZC, ZC-only) or an inline analog matching unrestricted's own scalar-gravity
convention (see `operator_psi_bundle.jl`'s header for why `prime_operator!` itself doesn't fit
that family cleanly).

## Real bugs found and fixed while generalizing (none introduced by this bundle; all pre-existing)

1. **`archC_frechet_hess_cb_builder`'s serial branch** (common Frechet's own, separate copy of the
   Hessian-callback builder) hardcoded the dense-only `_archC_prep_for_hessian!` instead of the
   shared dense-G-free `_prep_dual_index_for_archC!` dispatcher its own threaded branch (and
   flexible-CM's analogous function, in both branches) already used. A real disconnection between
   Frechet's own duplicated Hessian-callback code and flexible-CM's shared one -- see the master
   report's architecture note on this.
2. **`build_bin_tables_threaded!`** (shared infrastructure, `cm_hessian_threaded.jl`, used by
   flexible CM's own production-default threaded path too) had a strict
   `H::AbstractMatrix{Float64}` method signature -- rejected `nothing` at dispatch time even when
   `fill_S=false` means `H`'s contents are never read. Fixed to `Union{Nothing,...}` with the same
   explicit fail-fast guard `build_bin_tables!` (serial) already had.
3. **`archA_partitioned_hess_cb_builder`** (origin-ZC) unconditionally unpacked `H`, `H_copy`, AND
   a third scratch field, `∂∂f_∂∂x` (a previously-unnoticed dense-bundle-only field). `H`/`H_copy`
   fixed via the same `_dense_H_or_nothing`/`_dense_H_copy_or_nothing` dispatch; the scratch field
   replaced entirely with a closure-owned persistent scratch matrix (built once per KNITRO solve),
   used identically for both bundle types.

All three were latent, pre-existing gaps -- harmless against a real dense H (redundant reads or
unused unpacks), only surfaced as hard failures once a genuinely H-less bundle exercised them. The
gate scripts' own structural checks (fieldnames, obj.H throws) caught every one immediately and
unambiguously (a clean FieldError/MethodError, never a silent wrong answer).

## Architecture note (flagged per user request)

Flexible CM and common Frechet do **not** share a single parameterized Hessian-callback
implementation the way an ideal architecture would want (`[E | CM | CM-F]` with one set of
functions differing only in the block that constructs the CM-F/level piece). Common Frechet has
its own separate `hessian_cm_frechet_structured!`/`_v2!`/`archC_frechet_hess_cb_builder`
(`cm_frechet_hessian.jl`/`cm_frechet_hessian_threaded.jl`) -- largely structurally parallel to
flexible CM's `hessian_cm_structured!`/`_v2!`/`archC_hess_cb_builder`, but genuinely duplicated
code, not a shared function with a swapped-in level-block piece. This is exactly why bug #1 above
could happen at all: a fix applied to flexible CM's shared function did not automatically apply to
Frechet's separate copy. Unifying these into one shared parameterized implementation is a real,
identifiable architecture improvement -- out of scope for this task (which was about the bundle
storage layer, not the Hessian-assembly call graph), but worth naming explicitly as the natural
next refactor if this class of duplication-drift bug is to be prevented systematically rather than
caught one instance at a time.
