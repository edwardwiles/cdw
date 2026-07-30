# Production bundle construction call graph (2026-07-30)

Base SHA: `production/fullA-exact @ 79b941c` (both 2026-07-29 incidents from
`dense_bundle_incident_postmortem_2026-07-29.zip` already fixed at this SHA — this document audits
the *pattern* that allowed them, per the architecture-hardening task brief, not re-discovers the
already-fixed bugs themselves).

There are **3 real top-level driver functions**, not 5 — `run_cm_upper_checkpointed` is the shared
entry point for three of the five families (`flexible_cm`, `common_frechet`, `cm_meanzc`), branching
internally on `marginal_restriction`/`cm_extension`. `origin_zc` and `unrestricted` each have their
own driver.

## flexible_cm

- **Public campaign runner**: `campaign_cm_family_runner.jl` → dispatches to
  `run_cm_upper_checkpointed`/`run_cm_lower_checkpointed` (`cm_checkpoint.jl:1270`, thin
  `find_smallest=false` wrapper around the same function).
- **Standalone/checkpoint-resume runner**: `run_cm_upper_checkpointed` (`cm_checkpoint.jl:591`),
  called with `marginal_restriction=:common_flexible` (default), `cm_extension=:cm_only` (default).
- **Production context builder**: `build_cm_production_context` (`cm_production_bundle.jl:65`).
- **Bundle constructor**: called with `moment_representation` **not passed** at all
  (`cm_checkpoint.jl:929-931`, the `else` branch of the `is_meanzc ? ... : is_frechet ? ... : ...`
  ternary) — resolves to the builder's own default,
  `moment_representation::Symbol = MOMENT_REPRESENTATION[]` (`cm_production_bundle.jl:88`, a shared
  `Ref{Symbol}`, `no_dense_g_counters.jl:120`, currently `:operator`).
- **Moment-representation default**: shared global `MOMENT_REPRESENTATION[]` — the one family
  whose default is a mutable, inspectable switch rather than a private literal (postmortem §1's
  "control case").
- **FG/Hessian backend defaults**: `CM_INNER_FG_BACKEND_DEFAULT[]`, `CM_CROSS_HESSIAN_BACKEND_DEFAULT[]`.
- **Verification backend**: `test_verification_backend_default_cm.jl` gates it; live default read
  off `pcx.cctx`/`resolve_flexible_cm_manifest`.
- **Live stash**: `CM_LIVE_PCX_STASH` (`cm_hessian_subblock_profiling.jl`), opt-in via
  `CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]`.

## common_frechet

- **Public campaign runner / standalone runner**: same `run_cm_upper_checkpointed`, with
  `marginal_restriction=:common_frechet`.
- **Production context builder**: `build_cm_frechet_production_context` (`cm_frechet_level.jl:306`).
- **Bundle constructor**: `moment_representation` **is** threaded through from the driver
  (`cm_checkpoint.jl:922`, `mr_kwargs`), but only because `moment_representation::Union{Nothing,Symbol}
  = nothing` at the driver level (`cm_checkpoint.jl:707`) and `nothing` means "omit the kwarg" —
  i.e. every real caller that doesn't pass this kwarg explicitly (all of them except one 2026-07-29
  wiring test) resolves to the builder's own default, `moment_representation::Symbol = :operator`
  (`cm_frechet_level.jl:341`, a private literal, flipped from `:dense_reference` on 2026-07-29 per
  `FRECHET_OPERATOR_DEFAULT_INVESTIGATION_2026-07-29.md`).
- **FG/Hessian backend defaults**: `CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]`,
  `CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[]`, `cm_hessian_backend::Symbol=:dense_reference`
  (this one is the Hessian *architecture* choice, not the bundle-struct choice — a different axis,
  unaffected by this task).
- **Live stash**: same `CM_LIVE_PCX_STASH`.

## cm_meanzc (CM+ZC)

- **Public campaign runner / standalone runner**: same `run_cm_upper_checkpointed`, with
  `cm_extension != :cm_only`, `is_meanzc=true`.
- **Production context builder**: `build_cm_meanzc_production_context` (`cm_meanzc_production.jl:151`).
- **Bundle constructor**: same `mr_kwargs` threading as `common_frechet` — driver default `nothing`
  omits the kwarg, resolves to the builder's own private-literal default,
  `moment_representation::Symbol = :operator` (`cm_meanzc_production.jl:155`, flipped from
  `:dense_reference` on 2026-07-29 per the postmortem's "second incident" fix, commit `45576dc`).
- **Live stash**: `CMZC_LIVE_PCX_STASH` referenced by profiling/gate scripts.

## origin_zc (ZC-only)

- **Public campaign runner / standalone / checkpoint-resume runner**:
  `run_originzc_upper_checkpointed` (`cm_originzc_checkpoint.jl:463`).
- **Production context builder**: `build_originzc_production_context`
  (`cm_originzc_production.jl:48`).
- **Bundle constructor**: same `nothing`-omits-kwarg pattern
  (`cm_originzc_checkpoint.jl:505,643`) → builder's own private-literal default,
  `moment_representation::Symbol = :operator` (`cm_originzc_production.jl:50`, also flipped
  2026-07-29, commit `45576dc`).
- **Live stash**: `ORIGINZC_LIVE_PCX_STASH`.

## unrestricted

- **Public campaign runner**: `campaign_unrestricted_runner.jl`.
- **Standalone/checkpoint-resume runner**: `run_polish_checkpointed_unified`
  (`c10_d20_production_driver_unified.jl:129`).
- **Production context builder**: `build_unrestricted_operator_ctx` (`compressed_live.jl:447`) —
  note this is *not* a from-scratch context builder like the four restricted families' own
  `build_*_production_context`; it's a post-hoc converter called on an already-built dense `ctx`
  (`compressed_live.jl:212`, `ctx = build_unrestricted_operator_ctx(ctx; moment_representation=...)`).
- **Bundle constructor**: driver's own kwarg
  `moment_representation::Symbol = MOMENT_REPRESENTATION[]` (`c10_d20_production_driver_unified.jl:154`)
  passed straight through to `build_unrestricted_operator_ctx` — this is the family fixed by the
  postmortem's "first incident" (commit `1d9d2ed`); prior to that fix the conversion call did not
  exist in the driver at all.
- **Live mechanism**: the driver's own returned `ctx`, not a separate global stash.

## The structural pattern common to all three real driver functions

Every one of the 3 real driver functions still has an independently-implemented
`moment_representation`-shaped kwarg on its own signature (`run_cm_upper_checkpointed`,
`run_originzc_upper_checkpointed`: `Union{Nothing,Symbol}=nothing`;
`run_polish_checkpointed_unified`: `Symbol=MOMENT_REPRESENTATION[]`), each threading through to a
per-family builder that has *its own, separately defaulted* `moment_representation` kwarg
(`MOMENT_REPRESENTATION[]` shared global for `flexible_cm`/`unrestricted`; private `:operator`
literals for `common_frechet`/`cm_meanzc`/`origin_zc`). All five of today's defaults happen to
resolve to `:operator`. **Nothing in the type system, the function signatures, or the call graph
makes it structurally impossible for a future edit to flip any one of these five independent
defaults back to `:dense_reference`** — exactly the postmortem's root cause (§1: "two decoupled
defaults for one logical switch"), still present in its entirety even though the specific instances
found in the postmortem are fixed. This is the pattern this task's API split (§2-§4) removes by
construction: production runners will no longer accept the kwarg at all, so there is no default left
to independently flip.
