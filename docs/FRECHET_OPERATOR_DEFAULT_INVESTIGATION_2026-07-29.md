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

**This does NOT, on its own, prove there is no remaining issue anywhere** (a different W, L, or
`destination_sample` could still depend on something). Per this project's own standing lesson from
this exact bug's history ("a passing D=4-only gate is NOT sufficient evidence... do not re-attempt
without first root-causing"), the honest characterization is: **the specific historical failure
mode was not reproduced under the structurally-different mechanism, at the scale and points that
used to break it** — a real, positive finding, not a full clearance of every possible remaining
dependency.

## Part 2: the outer gradient (`composite_gradient_at_Cplus_frechet`/`cm_frechet_production_gradient_cplus`)

The user pushed back on treating the outer gradient as an open question at all, correctly pointing
out that `build_lfix_base_cache_C!`/`composite_gradient_at_Cplus_from_cache` are literally the same
shared functions for both families and never touch `obj.H`/`G` outside an explicit, default-off
`validate_dense` self-check — confirmed by reading both functions in full. The only Fréchet-specific
piece, `frechet_cm_level_fixed_contribution`, is bins-based (not H-based) for both its CM part (now
calling the same `cm_fixed_value_contribution` as flexCM) and its level part. So there was no real
mechanism by which the outer gradient specifically could depend on dense H/G.

Ran it anyway (`investigate_frechet_outer_gradient_operator_gate_2026-07-29.jl`): real D=20/W=80,000,
both contrasts, calibration + both historically-failing outer points, comparing
`cm_frechet_production_gradient_cplus`'s full 380-dimensional output (dense vs operator):

```
anchored     calib                  max|Δg|=0.000e+00  n_g=380  pass=true  (2.1x speedup)
anchored     near_delta1_perturbed  max|Δg|=0.000e+00  n_g=380  pass=true
anchored     hard_point_x1.01       max|Δg|=0.000e+00  n_g=380  pass=true
orthonormal  calib                  max|Δg|=0.000e+00  n_g=380  pass=true
orthonormal  near_delta1_perturbed  max|Δg|=0.000e+00  n_g=380  pass=true
orthonormal  hard_point_x1.01       max|Δg|=0.000e+00  n_g=380  pass=true
```

All 6 cells exact. Confirms the conceptual argument: the outer gradient never had a real
bundle-type dependency; the only thing that mattered was the inner solve it's built on top of
(already covered in Part 1).

## Part 3: default flipped, with a real side-finding along the way

`build_cm_frechet_production_context`'s `moment_representation` default was flipped to `:operator`
in `cm_frechet_level.jl` (matching flexible_cm). The real production driver
(`run_cm_upper_checkpointed`) never passes this kwarg explicitly and already satisfies the other
prerequisites (`cm_hessian_backend=:structured` is its own default, `inner_fg_backend` defaults to
`CM_FRECHET_INNER_FG_BACKEND_DEFAULT[] = :cm_frechet_lookup`) — so flipping this one default is
sufficient to change production behavior with no other call-site changes needed.

**A confirmation run at this point produced a genuine `nStatus=-500` (`KN_RC_CALLBACK_ERR`) — a
real Julia exception thrown inside the KNITRO callback, not a false alarm to wave away.**
Root-caused via `debug_frechet_operator_coldstart_2026-07-29.jl` (bypasses KNITRO to surface the
real exception instead of KNITRO's caught-and-coded `-500`): the underlying error was

```
hessian_core_winner_pair!: workers=6 not in workspace's precomputed worker_counts
```

`build_winner_pair_parallel_workspace`'s default `worker_counts = [1, 2, 4, 8, 10, 19, 20]`
(`core_exact_hessian.jl:602`) does not include 6, and the confirmation script happened to run with
`-t 6` (my own earlier passing investigation scripts used `-t 8`). This is a **pre-existing
constraint of the shared winner-pair Hessian backend, identical for both `:dense_reference` and
`:operator`** — completely unrelated to this task's flip. Re-run at `-t 8`: clean pass, `nStatus=0`
both, `|Δζ*|=0.000e+00`. Recorded here as an operational gotcha worth knowing about (any real run —
this family or flexible_cm, either `moment_representation`, whenever the winner-pair backend is
active — must use a thread count from that precomputed list or pass its own `worker_counts` at
workspace construction time), not as a defect in this task's change.

## Outcome

User confirmed: proceed with the flip (no dense fallback anywhere). Done in this task:

1. `moment_representation` default flipped `:dense_reference` -> `:operator` in
   `cm_frechet_level.jl`, matching flexible_cm.
2. Outer-gradient path tested under `:operator` (Part 2 above) — exact agreement, 6/6 cells.
3. Confirmed the real production driver's own call site (`run_cm_upper_checkpointed`, which never
   passes `moment_representation` explicitly) now builds and solves through the `OperatorPsiBundle`
   correctly end-to-end (`confirm_frechet_operator_default_flip_2026-07-29.jl`), matching the
   explicit `:dense_reference` sibling to `0.000e+00`.
4. Found and root-caused an unrelated pre-existing gotcha along the way (Part 3's `worker_counts`
   constraint) — recorded, not fixed (out of scope; affects both bundle types equally and both
   families that use the winner-pair backend).

Not done (out of scope / lower priority, left as follow-on items): a second `(W,
destination_sample)` combination for extra robustness beyond what's tested here; updating
`FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md`'s status table to reflect the default flip (that doc
already lists common_frechet as gated, just not as the *default*).

`:dense_reference` remains fully available as an explicit opt-in (nothing was deleted from that
code path) for reference/debug use.
