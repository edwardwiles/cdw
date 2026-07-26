# Common-Fréchet CM driver port — 2026-07-25/26 (Parts II & V)

## Part II: moment-construction layer + CMConfig wiring

New file `cm_frechet_level.jl`: dense reference (`precalc_frechet_level_dense`) and Architecture-B
bin-lookup fast path (`fill_frechet_level_columns_from_bins!`, mirroring `fill_cm_columns_from_bins!`
exactly) for the `L` level columns, plus `build_cm_frechet_level_augmented_obj` (Architecture-A,
concatenates `[CM level]` and hands the combined matrix to `wrap_moments_with_cm` **unchanged** —
that function is already generic on the supplied moment matrix) and
`build_cm_frechet_production_context` (Architecture-B, production speed, real dispatch between
`cm_hessian_backend=:dense_reference`/`:structured`).

`CMConfig` (`cm_config.jl`) gained one new field, `marginal_restriction::Symbol = :common_flexible`
(default byte-identical to pre-existing behavior; `:common_frechet` opt-in).
`build_cm_production_context_v2` dispatches `:common_frechet` to the new production context, adding
one new branch, touching no existing branch's behavior.

Startup manifest (`print_frechet_startup_manifest`) prints exactly the task-specified fields:
```
marginal_restriction = common_frechet
frechet_feature_set = cdf_only
frechet_basis = cm_contrasts_plus_common_level
frechet_grid_size = <L>
cm_contrast_count = D-1
frechet_level_count = 1        # PER THRESHOLD; total_marginal_moments = D*L confirms the multiply-out
total_marginal_moments = D*L
```
Confirmed live at D=4/L=10 (`total_marginal_moments=40`) and D=20/L=10 (`total_marginal_moments=200`).

## Part V: the real public checkpointed driver

`run_cm_upper_checkpointed` (`cm_checkpoint.jl`) gained a `marginal_restriction::Symbol =
:common_flexible` kwarg, with the same "hard-refuse on resume mismatch" discipline the existing
`cm_extension`/`destination_sample` checks already use (restriction-column count/meaning differs —
no safe override). Currently requires `cm_extension=:cm_only` (guarded, explicit error) — not yet
combined with the meanzc extension (orthogonal but unvalidated combination, disclosed scope
boundary, not attempted this session).

Dispatch added at every family-specific call site: pcx construction, the verified/screened value
path (`cm_frechet_production_value_verified_screened`), and both gradient backends
(`cm_frechet_production_gradient(_cplus)`) — new functions in `cm_frechet_cplus.jl`, mirroring the
existing CM-only functions' signatures/bodies exactly, reusing `cm_screen_precheck!` (screening)
and `build_cm_bin_ctx` (Architecture-C bin tables) **unchanged**.

`build_cm_frechet_production_context` now returns top-level `bins`/`cctx` fields matching
`build_cm_production_context`'s own `pcx` shape (previously only nested under `aug`), so it's a
drop-in for every function expecting the plain-CM `pcx` shape.

### Checkpoint schema: `CMCheckpointV8`

Bumps `CM_CHECKPOINT_SCHEMA` 6→8 (skips 7, already claimed by `cm_originzc_checkpoint.jl` —
checked the whole tree for `CMCheckpointV*` name collisions before bumping, per this project's own
standing gotcha). Adds one field, `marginal_restriction`, to `CMCheckpointV6`'s layout; permanent
upgrade path `upgrade_schema6_to_v8` (every schema-6 file is `:common_flexible` by construction,
since `:common_frechet` didn't exist at that schema — not a guess).

## Live validation (real public driver, not a test-script bypass)

Real D=20/W=80,000/L=10/`destination_sample=:exclude_row`/`cm_hessian_backend=:structured`
(winner-pair) run through `run_cm_upper_checkpointed` with `marginal_restriction=:common_frechet`:
**10 real KNITRO outer evaluations, 6 gradients, genuine feasible progress** (`gp: 0.9878→0.9702`),
checkpoint written correctly (`schema=8`, `marginal_restriction=common_frechet`).

Resume gate (`smoke_frechet_resume.jl`, 6/6 PASS): mismatched `marginal_restriction` on resume
correctly refused; matched resume correctly continues `n_eval`/`n_grad` counters (`10→12`, `6→8`);
resumed checkpoint still reports the correct schema/mode.

**A genuine false alarm along the way, ruled out empirically, not asserted away**: an initial
`W=2000` smoke attempt failed with what looked like a Fréchet-specific numerical blowup
(`nStatus=-300`/unbounded at the very first point, byte-identical wild `x_free0` values). A
side-by-side control run of **plain flexible CM at the identical small W failed with the exact same
symptom** — confirmed a known small-W conditioning issue (this project's own prior finding: `W=2,000`
is debugging-only), not a bug in this port. Documented so a future session doesn't re-discover it
from scratch.

## Known, disclosed gap

`run_cm_upper_checkpointed` uses the **legacy** `pivot_reduce`/`pivot_expand` (log-A_od,
`legacy_z`) outer-coordinate parametrization throughout — it has zero reference to
`outer_coordinate_layout.jl`/the new transformed-A machinery. This session's `:common_frechet`
wiring targets this legacy-coordinate driver only; it has **not** been integrated with (or
validated against) the separate, now-canonical transformed-A unified driver
(`c10_d20_production_driver_unified.jl`-family). Genuine, disclosed follow-up task, not attempted
this session (out of scope given the remaining time budget once transformed-A landed mid-session).
