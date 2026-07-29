# Investigation: is common Fréchet's `moment_representation=:dense_reference`-only default still justified? (2026-07-29)

## Question

The master report flagged, as the highest-priority remaining gap, that `common_frechet` is
hardcoded to `moment_representation=:dense_reference` (cannot use the true no-H `OperatorPsiBundle`
path `flexible_cm` defaults to) because of a documented history of reproduced `nStatus=-400`
failures. User asked: why should this still be a problem, given the FG-callback harmonization just
completed makes common Fréchet's `[E|C]` block literally the same code as flexible CM's?

## Finding: two different mechanisms were conflated

1. **The actual historical bug** (`archC_frechet_base_state`'s HISTORY comment,
   `cm_frechet_cplus.jl:113-133`, first found 2026-07-26, re-confirmed 2026-07-27) is about
   `skip_cm_fill_ref`/`skip_fill` — an optimization that left a STILL-DENSE
   `PsiObjectiveBundleImplicit`'s CM/level columns of `obj.H` stale (unfilled) while using the
   `:cm_frechet_lookup` FG backend, on the (twice-falsified) claim that nothing reads those columns
   anymore. Something in the Hessian/gradient callback chain silently depended on those stale
   columns being non-garbage, producing `nStatus=-400` specifically at non-calibration outer θ
   points; root cause was never identified, and the skip was permanently reverted (dense fill now
   always runs).
2. **`moment_representation=:operator`** (`OperatorPsiBundle`, `operator_psi_bundle.jl`) is
   structurally different: the bundle has **no dense `H` field at all** (confirmed by this repo's
   own `test_operator_no_H_bundle_equivalence_frechet_d20.jl`, which asserts
   `:H, :H_copy, :G, :K, :ones` are absent from `fieldnames`). Anything that still depended on those
   columns would hit a hard `MethodError`/`FieldError` at construction/dispatch time, not a silent
   wrong answer — and the 2026-07-28 five-family generalization session
   (`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`) found and fixed exactly three such hard failures
   across all five families (one specifically in common Fréchet's own
   `archC_frechet_hess_cb_builder`).

`build_cm_frechet_production_context`'s own docstring (added in commit `6a8fadc`, 2026-07-28
11:53) cites mechanism (1)'s history as the reason to keep `:dense_reference` as the DEFAULT for
mechanism (2) as well — a defensible conservative choice at the time it was written, but the two
mechanisms are not the same code path, and the citation does not by itself demonstrate mechanism
(2) inherits mechanism (1)'s bug.

## The actual test-coverage gap

Every existing no-H-bundle equivalence gate for common Fréchet
(`test_operator_no_H_bundle_equivalence_frechet_d20.jl`, commit `ad1cd43`, same day) runs the real
full inner solve **only from the calibration outer θ** (`x_free_calib`) — exactly the "missed at
the calibration point, caught beyond it" pattern the HISTORY comment itself warns about. Nobody
had re-tested `:operator` at the specific non-calibration outer points
(`near_delta1_perturbed`, `hard_point_x1.01`) that reproduced mechanism (1)'s failure twice.
`flexible_cm`'s own D=20 no-H gate has the identical calibration-only limitation — this is not a
common-Fréchet-specific gap in test design, just one that matters more here given this family's
specific failure history.

## What this task ran to close the gap

New script: `investigate_frechet_operator_nonlocal_gate_2026-07-29.jl`. Builds both a
`:dense_reference` and an `:operator` `build_cm_frechet_production_context` at real
D=20/W=80,000/L=50/`destination_sample=:exclude_row` (same scale as the original bug reports), and
runs the REAL full inner solve (`archC_frechet_base_state`, exactly the function whose HISTORY
comment documents the bug) from three outer θ points — calibration, `near_delta1_perturbed`, and
`hard_point_x1.01` (`x_free_calib .* 1.01`, the exact perturbation that originally exposed the
crash) — for both `:anchored` and `:orthonormal` contrasts (6 cells total).

## Result

```
anchored     calib                  status(d/o)=0/0 match=true |Δζ*|=0.000e+00 pass=true
anchored     near_delta1_perturbed  status(d/o)=0/0 match=true |Δζ*|=0.000e+00 pass=true
anchored     hard_point_x1.01       status(d/o)=0/0 match=true |Δζ*|=0.000e+00 pass=true
orthonormal  calib                  status(d/o)=0/0 match=true |Δζ*|=0.000e+00 pass=true
orthonormal  near_delta1_perturbed  status(d/o)=0/0 match=true |Δζ*|=0.000e+00 pass=true
orthonormal  hard_point_x1.01       status(d/o)=0/0 match=true |Δζ*|=0.000e+00 pass=true
```

All 6 cells: no `nStatus=-400`, no crash, `:operator` reproduces `:dense_reference`'s accepted
status and dual solution EXACTLY (`0.000e+00` on both `ζ*` and `λ*`) — including at the two specific
outer points that reproduced the historical failure under mechanism (1), at the same real-data
scale (D=20, W=80,000) the original bug reports used.

## Interpretation

This is real, direct evidence that `moment_representation=:operator` does NOT reproduce the
historical `skip_cm_fill_ref` failure for common Fréchet, at least in this configuration
(`W=80,000`, `L=50`, `destination_sample=:exclude_row`, both contrasts, the two specific
historically-failing outer points, `cm_hessian_backend=:structured`,
`inner_fg_backend=:cm_frechet_lookup`). Combined with the mechanistic argument (no dense array to
go stale, any real remaining dependency would hard-crash rather than silently corrupt) and the
now-completed FG-callback harmonization (common Fréchet's `[E|C]` block is now literally the same
code flexible_cm's is), this substantially undermines the case for keeping `:dense_reference` as
the default out of caution inherited from the old mechanism.

**This does NOT, on its own, prove there is no remaining issue anywhere** (a different W, L,
`destination_sample`, or a code path not exercised by `archC_frechet_base_state` — e.g. the outer
gradient / `composite_gradient_at_Cplus_frechet` path, which was NOT tested here — could still
depend on something). Per this project's own standing lesson from this exact bug's history ("a
passing D=4-only gate is NOT sufficient evidence... do not re-attempt without first root-causing"),
the honest characterization is: **the specific historical failure mode was not reproduced under the
structurally-different mechanism, at the scale and points that used to break it** — a real, positive
finding, not a full clearance of every possible remaining dependency.

## Recommendation

Flipping `build_cm_frechet_production_context`'s `moment_representation` default from
`:dense_reference` to `:operator` (matching flexible_cm) is very likely safe based on this
evidence, but this task did NOT make that change — it is a production-default flip with a
documented failure history attached to it (twice), and per this project's standing "confirm before
changing a documented-risk production default" posture, that decision is left to the user rather
than made unilaterally. If the user wants to proceed: (1) flip the default in
`cm_frechet_level.jl`, (2) also test the outer-gradient path
(`composite_gradient_at_Cplus_frechet`/`build_lfix_base_cache_cm_frechet_C!`) under `:operator`,
not just the inner dual solve, since that path was not covered by this investigation, (3) run at
one more `(W, destination_sample)` combination for robustness, (4) update
`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`'s status table accordingly.
