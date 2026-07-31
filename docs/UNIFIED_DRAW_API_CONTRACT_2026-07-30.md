# Unified draw API contract (2026-07-30)

## Typed design representation

`full_aod_diag/d4_exact/draw_design_types.jl`:

```julia
abstract type DrawDesign end
struct PseudorandomDesign <: DrawDesign; seed::Int; end
struct RandomizedSobolDesign <: DrawDesign; seed::Int; end
struct ScrambledHaltonDesign <: DrawDesign; seed::Int; end
struct PrecomputedDrawDesign <: DrawDesign; U::Matrix{Float64}; already_transformed::Bool; label::String; end
```

The public interface stays **Symbol**-based (`:pseudorandom`/`:sobol_randomized`/`:halton_scrambled`/
`:precomputed`), matching the pre-existing `draw_design::Symbol` checkpoint-provenance convention
used throughout the campaign/checkpoint layer (`cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`,
`c10_d20_production_driver*.jl`) -- that convention was already architecturally correct (provenance
metadata, never a numerical branch; see the reachability audit) and changing ~14 struct fields and
~100 call sites to a typed interface would have added surface area without fixing anything. Instead,
`resolve_draw_design(design::Symbol, seed::Int; ...)` in `draw_design.jl::d20_real_setup_design`
converts the Symbol to a typed `DrawDesign` **exactly once**, before any draw is generated.

## The one authoritative draw API

```julia
generate_randoms!(Uexp::Matrix{Float64}, design::DrawDesign) -> Uexp
```

One method per `DrawDesign` subtype (4 total -- verified as the only multi-method function in the
whole traced pipeline, see `POST_DRAW_METHOD_IDENTITY_PROOF_2026-07-30.md`). The caller
preallocates `Uexp` (`W x D`, `Float64`); every design fills the same shape, scalar type, and
storage layout (column-major `Array`, Julia's native layout). `:pseudorandom` fills it fully
in-place via the pre-existing `genExpRands!` (zero extra allocation). `:sobol_randomized`/
`:halton_scrambled` call the pre-existing, unmodified `sobol_U`/`halton_U` generators
(`qmc_draws.jl`) and `copyto!` the result in -- one top-level copy, not eliminated by this task
(see the resource gate CSV for the measured cost, negligible next to context construction).
`:precomputed` validates shape and `copyto!`s the caller's matrix, applying the shared transform
only if `already_transformed=false`.

## The one shared inverse-CDF transform

Task step 3 asked for a single shared implementation such as `U[i] = -log1p(-U[i])`. This branch
uses `U[i] = -log(1 - U[i])` instead -- **a deliberate deviation from the illustrative snippet**,
not an oversight. `-log(1-u)` is exactly what production's pre-existing `genExpRands!` already
computed, and is exactly what the (now-deleted) QMC path's `exp_from_uniform01` already computed
too (confirmed by direct reading of both, prior to any change). `log1p(-u)` is mathematically
identical but is **not bit-for-bit identical** to `log(1-u)` for arbitrary floating-point `u` (they
use different underlying algorithms and can differ in the last ULP). Task step 4 says: "Prove that
the pseudorandom sequence remains identical under the same RNG algorithm and seed, or stop and
obtain an explicit decision before changing it." Switching to `log1p` would not be provably
bit-identical to the pre-existing pseudorandom sequence without an exhaustive re-verification this
session did not have a safe way to perform without risking a silent behavior change to already-
validated production results -- so the existing `-log(1-u)` form was kept, extracted into one
shared function (`transform_unit01_to_exp1!`, `prepare_cc/genRands.jl`) that both `genExpRands!`
and every draw design now call, satisfying "one shared implementation" without touching the
underlying arithmetic. A defensive `clamp` to `prevfloat(1.0)` was added and proven to be a
mathematical no-op for anything `rand!` can produce (Float64 uniform sampling never returns exactly
1.0, and `prevfloat(1.0)` is the largest double strictly below 1.0 -- there is no representable
value in between), so this is still bit-for-bit identical to the pre-existing `genExpRands!`.

## Global RNG state

Task step 4 also asks to "Replace global `Random.seed!(seed)` with an explicit RNG object where
possible." This was done for exactly the designs where it was already true or risk-free:

- `:sobol_randomized`/`:halton_scrambled` already saved and restored the caller-visible global RNG
  state around their internal `Random.seed!`/`rand`/`shuffle` calls (`qmc_draws.jl`'s pre-existing
  `_with_saved_global_rng` wrapper, unmodified) -- production construction with these designs
  already did not leak global RNG mutation to the caller, unchanged by this task.
- `:precomputed` never touches the RNG at all.
- `:pseudorandom` **still mutates global RNG state** (`Random.seed!(draw_seed)` in
  `d20_real_setup_design`, then `Random.seed!(seedU)` inside `master_prepare_cc`), exactly as
  before this task. This is a **deliberate deferral, not an oversight**: production's pseudorandom
  results have already been validated against this exact global-RNG call sequence, and swapping it
  for an explicit `Xoshiro(seed)`-threaded-through-`rand!` design (the behavior-preserving version
  of this refactor) would need to be run and diffed against real historical results before being
  trusted -- exactly the "stop and obtain an explicit decision" the task anticipates for this case.
  Flagged here for a future session with the time budget to do that verification properly, rather
  than attempted speculatively in this one.

## Draw design determines only how `U` is filled

Everything downstream of `generate_randoms!`/`master_prepare_cc`'s `U=nothing` branch is single-
method (see `POST_DRAW_METHOD_IDENTITY_PROOF_2026-07-30.md`): `build_ad_context_real_d20`,
`d20_real_setup`, screen construction, threshold construction, objective/bundle construction, and
every downstream diagnostic/production function all take the completed `U` (or its absence, for
the pseudorandom in-function-draw case) as their only draw-related input.

## Standardized draw manifest

`draw_design.jl::draw_design_meta` returns: `draw_design`, `randomization_method`, `draw_seed`,
`D`, `W`, `scalar_type`, `matrix_layout`, `transform_convention` (+ version), `sobol_jl_version`,
`julia_version`, `generation_code_sha` (git HEAD, best-effort), `artifact_path` (provenance
passthrough for a caller-loaded precomputed matrix), `checksum_uniform`, `checksum_transformed`,
`n_at_boundary`, `n_inf_transformed`, `timing`. Every field required by task §11 is present.
