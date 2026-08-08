# Operational notes for Claude Code sessions in this repo

This is the canonical repo as of 2026-08-03 (origin `git@github.com:edwardwiles/cdw.git`). Fuller
project history/context (including the 2026-08-03 repo cleanup that made this the canonical repo)
lives in `/bbkinghome/edav/gravity_robustness/CLAUDE.md` and `/bbkinghome/edav/repo_salvage/` on
this server, if that path is reachable from your session.

## ⚠️ Never let a function default a scientific parameter — require it, error if omitted

**The rule, stated by the user directly (2026-08-03):** no function anywhere in the EK/Ricardo
FULL/REDUCED call chain may give a default value to a parameter that changes *what economic
problem is being solved* — sigma, W, L, K_mean/K_pair, draw_design, draw_seed, gravity exclusions
(`exclude_diagonal_gravity`/`gravity_exclude_cells`), `destination_sample`, CM `contrasts`/
`probs`. A caller who omits one must get a hard error, not a silently-substituted value. In Julia
this is a plain keyword argument with **no** `= value` (`sigma::Float64`, not
`sigma::Float64 = 2.5`) — Julia raises `UndefKeywordError` immediately if it's omitted, before the
function body runs at all. The single source of truth for what these values *should* be is
`scientific_manifest/ScientificManifest.jl` + `configs/fullA_production_2026-08-03.toml` —
production runners load it (`SCIENTIFIC_MANIFEST_TOML` env override; hard error, not a default, if
the file is missing) rather than hardcoding their own copy of the same numbers.

**Do not repeat the anti-pattern this codebase already committed once**, on 2026-08-01: flip a
dangerous default (sigma `nothing`→2.5) at the top production-driver layer only, and leave the
deeper shared context builders (`d20_real_setup`/`d20_real_setup_design`/
`build_ad_context_real_d20`, `full_aod_diag/d4_exact/context_real_d20.jl` +
`draw_design.jl`) with their own, *different*, unfixed default — with an explicit comment
justifying it as "~200 unrelated diagnostic/benchmark scripts depend on their old defaults." That
reasoning is backwards: the deeper a function sits in the call chain, the more callers silently
inherit whatever default lives there, so it is exactly the layer that most needs no default, not
the one safest to leave alone. Fixed for real 2026-08-03 (branch
`hardening/require-scientific-params-2026-08-03`) by removing the default at *every* layer — root
builders through the three real production entry points
(`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`/`run_polish_checkpointed_unified`)
— which does mean those ~200 old scripts now throw `UndefKeywordError` if run. That is the
intended consequence, not a regression to silently work around.

**Before touching any "default" — including one that looks obviously dangerous — trace the full
call chain and check for an existing resolver/validator; do not assume from one layer.** Two
concrete lessons from the same session, both confirmed live:
1. A pass through this exact call chain concluded the live FULL W100k 10x10 campaign was silently
   running at sigma=2.5, based on checking only the campaign script's call site (no explicit
   `σHat`) and the deepest function's default (`nothing`→2.5). Wrong — an *intermediate* function
   (`run_cm_upper_checkpointed`) had its own default of `σHat=3.0`, which is what actually governed
   behavior since nothing overrode it. Corrected same session after re-tracing every layer.
2. Conversely, `meanzc_K_mean`/`meanzc_K_pair` still default to `0` in `run_cm_upper_checkpointed`
   and were *not* changed, because tracing the call chain found `meanzc_resolve_K`
   (`cm_meanzc_config.jl`) already hard-errors `"cm_extension=:cm_plus_moments requires
   meanzc_K_mean >= 1, got 0"` if a caller turns that extension on without setting K — `0` is a
   real, meaningful value for the (far more common) case where the extension is off, not a silent
   placeholder. Removing that default would only have added noise, not safety.

**Deliberately does not extend to Melitz** (`src/melitz/`, `scripts/melitz_*`, `melitz/`,
`test/melitz/`) — EK/Ricardo and Melitz are different models with separate config surfaces, kept
that way on purpose (user directive, 2026-08-03). Do not make Melitz code depend on
`ScientificManifest`, and do not give Melitz's own settings the same treatment without being
asked — it wasn't part of this hardening pass.

## Reusable production utility: multistart seed generator — use this, don't reinvent it

**If a task needs multiple randomized-but-verified starting points for a FULL-A outer campaign
(origin-ZC / CM+ZC / common-Fréchet, any combination), a reusable, tested, production-validated
generator already exists.** Do not hand-invent candidate `A_od`/`gp` points, guess nu values, or
write a one-off campaign-specific script for this — that exact anti-pattern (repeated across many
past sessions) is precisely what this utility replaces.

- **Code**: `full_aod_diag/d4_exact/multistart_seed_generator.jl` (implementation) +
  `full_aod_diag/d4_exact/test_multistart_seed_generator.jl` (tests, run standalone the same way
  every other `test_*.jl` file in this directory runs). Include both the same way every other
  `cm_*` file in this directory is included (self-`include`s nothing external beyond the standard
  d4_exact family machinery — its own header comment lists every file it needs already loaded).
- **Full docs, API reference, worked examples, and the exact live `gp`/nu-policy formulas**:
  `docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md` — read this before calling
  the generator, not just this pointer.
- **Entry point**:
  ```julia
  generate_multistart_seeds(ctx; M, direction, delta_max, family_specs, W, rng_seed, A_scale,
      gp_scale=1.0, max_attempts=200, include_calibration=true, min_seed_distance=0.0,
      output_dir) -> MultiStartSeedSet
  ```
  `ctx` must be built via `d20_real_setup_design(...)` (explicit `σHat`, `inner_lower_limit=-10.0`,
  the draw design/gravity settings you actually intend) — nothing scientific is defaulted inside
  the generator itself. `family_specs` is a plain caller-supplied `Vector{FamilySeedSpec}` (never
  hard-wired) — `production_five_family_seed_specs(ctx)` gives the CURRENT five-family production
  comparison (Origin-ZC/CM+ZC at K_mean=3×{K_pair=0,K_pair=3}, plus single-family common-Fréchet)
  as one convenience preset, not the only option.
- **Before trusting a specific `A_scale`/`gp_scale`**: these are NOT tuned defaults — an
  aggressive scale can make every random draw genuinely infeasible (confirmed live: 0/10 vs 10/10
  acceptance between two scale choices at the same `W`). Scan a small grid at your own `W`/
  `delta_max` first (MASTER.md documents the exact scan pattern used to find a working scale), do
  not assume a value from an old campaign or from the docs' own example call transfers unchanged.
- **Nu is not something you invent per family**: it is derived deterministically from a companion
  LFD solve at the same candidate point (never a fixed constant, never randomized) — MASTER.md
  section 5 has the exact mechanism and why. Each origin-ZC/CM+ZC family costs TWO real KNITRO
  solves per candidate (the companion, then the restricted family) — budget campaign wall-clock
  accordingly, especially at `W=100_000`.
- **Reproducibility is a tested contract**: identical `(rng_seed, family_specs, W, A_scale,
  gp_scale)` always gives bit-identical accepted seeds and digests, independent of attempt
  completion order — verified live at real D20/W=20000 (two independent process-cold runs, exact
  diff match) and covered by the test suite's own reordering test. If a campaign needs a different
  seed set, change `rng_seed`, don't work around the generator.
