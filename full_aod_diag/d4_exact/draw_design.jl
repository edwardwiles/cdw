# ============================================================================
# Production draw-design resolver -- the ONLY production entry point that
# selects among draw designs. As of the "unify random-draw production
# pipeline" task (2026-07-30), this file is a thin resolver only (task §9):
# it resolves the requested design, fills/transforms the one preallocated `U`
# matrix, and calls the SINGLE `d20_real_setup` implementation
# (context_real_d20.jl) for every design. It does not construct or patch
# screens, threshold state, context fields, objectives, or moments -- all of
# that lives in exactly one place (context_real_d20.jl::d20_real_setup),
# reached identically regardless of which design supplied `U`.
#
# History: this file previously routed :sobol_randomized/:halton_scrambled
# through a parallel qmc_context_real_d20.jl pipeline
# (build_ad_context_real_d20_qmc/master_prepare_cc_qmc/d20_real_setup_qmc)
# that had already drifted from the pseudorandom path (missing
# threshold_state, missing screen construction, patched back in here as a
# 3rd copy of the screen-building block). See
# docs/DRAW_DESIGN_PIPELINE_CALL_GRAPH_2026-07-30.md and
# docs/QMC_PSEUDORANDOM_DUPLICATION_REACHABILITY_2026-07-30.md for the full
# audit. That pipeline is deleted; every design now reaches the identical
# post-draw code path (see docs/POST_DRAW_METHOD_IDENTITY_PROOF_2026-07-30.md).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> d20_real_setup, build_ad_context_real_d20, D4X_ROOT, AD_PARAMS
include(joinpath(@__DIR__, "draw_design_types.jl"))   # -> DrawDesign hierarchy, resolve_draw_design, generate_randoms!
include(joinpath(@__DIR__, "qmc_draws.jl"))           # -> pseudorandom_U (diagnostic use only), halton_U, sobol_U
using Sobol
using SHA   # AUD-11 fix: stable cross-process/cross-version checksums (stdlib, no Project.toml entry needed)
# sha256_of_matrix now lives in oracle.jl (context_fingerprint, the AUD-08 fix, is its primary
# unconditional consumer and oracle.jl is the more universally-included file); guard include so
# draw_design_meta works even if this file is used in isolation from oracle.jl.
isdefined(Main, :sha256_of_matrix) || include(joinpath(@__DIR__, "oracle.jl"))

const VALID_DRAW_DESIGNS = (:pseudorandom, :sobol_randomized, :halton_scrambled, :precomputed)

const DRAW_DESIGN_DESCRIPTIONS = Dict(
    :pseudorandom     => "Julia default RNG (MersenneTwister via Random.rand!) -- production baseline, byte-for-byte unmodified from the pre-existing d20_real_setup/Random.seed!(draw_seed) path",
    :sobol_randomized => "Sobol.jl SobolSeq deterministic base sequence + an independent Cranley-Patterson random shift (mod 1) per seed -- a plain randomized-QMC shift, NOT Owen/digital scrambling (Sobol.jl does not implement digital scrambling)",
    :halton_scrambled => "cc_algo/rhalton.jl scrambled Halton sequence -- genuine Owen-style per-digit scrambling (independent random digit permutation per radix digit, per dimension), ported from Art B. Owen's R code",
    :precomputed      => "Caller-supplied W x D draw matrix (diagnostic synthetic matrix, persisted historical artifact, or a same-seed cross-design comparison matrix) -- see PrecomputedDrawDesign",
)

"""
    draw_design_meta(design, seed, D, W, U; timing=NamedTuple()) -> NamedTuple

Per-context metadata block logged for every draw design: design name,
randomization method (named honestly, see DRAW_DESIGN_DESCRIPTIONS), seed, D,
W, package versions, the transform convention, and checksums of both the
recovered raw-uniform draws and the transformed (Exp(1)) productivity draws.

The raw uniform draws are recovered from the final Exp(1) matrix via the
transform's own inverse (`U01 = 1 - exp(-U)`) rather than threaded through
separately -- `transform_unit01_to_exp1!` is a bijection on [0,1) x [0,Inf), so this
is exact (not an approximation), and it means the :pseudorandom path (whose
raw U01 is overwritten in place by genExpRands! and never separately
returned) gets the same checksum treatment as the QMC paths with no change to
genExpRands!/drawU.jl.
"""

"Best-effort git HEAD SHA of the generation code producing this manifest -- cached after first
call (repo HEAD doesn't change mid-process); `missing` if `git` isn't available (e.g. a packaged
deployment without a .git directory)."
const _GENERATION_CODE_SHA = Ref{Union{Missing,String}}(missing)
function _generation_code_sha()
    if _GENERATION_CODE_SHA[] === missing
        _GENERATION_CODE_SHA[] = try
            strip(read(`git -C $(D4X_ROOT) rev-parse HEAD`, String))
        catch
            missing
        end
    end
    return _GENERATION_CODE_SHA[]
end

function draw_design_meta(design::Symbol, seed::Int, D::Int, W::Int, Uexp::AbstractMatrix{Float64};
        timing::NamedTuple = NamedTuple(), artifact_path::Union{Nothing,AbstractString} = nothing)
    U01_recovered = 1.0 .- exp.(-Uexp)
    return (
        draw_design = design,
        randomization_method = DRAW_DESIGN_DESCRIPTIONS[design],
        draw_seed = seed,
        D = D,
        W = W,
        scalar_type = eltype(Uexp),
        matrix_layout = "W x D, column-major (Julia native Array layout)",
        transform_convention = "Exp(1) via U[i] = -log(1 - U01[i]) (prepare_cc/genRands.jl::transform_unit01_to_exp1!, the single shared implementation used identically by all designs, version=v1-2026-07-30)",
        sobol_jl_version = design == :sobol_randomized ? string(pkgversion(Sobol)) : missing,
        julia_version = string(VERSION),
        generation_code_sha = _generation_code_sha(),
        artifact_path = artifact_path,
        checksum_uniform = sha256_of_matrix(U01_recovered),
        checksum_transformed = sha256_of_matrix(Uexp),
        n_at_boundary = count(x -> x <= 0.0 || x >= 1.0, U01_recovered),
        n_inf_transformed = count(!isfinite, Uexp),
        timing = timing,
    )
end

"""
    d20_real_setup_design(; W, δ=1.0, find_smallest=true, draw_design=:pseudorandom,
                           draw_seed=20260719, outer_loop_opt=..., inner_loop_opt=...,
                           needs_outer_moment_jacobian=false, build_screen=true,
                           log_draw_meta=true, U_precomputed=nothing,
                           precomputed_already_transformed=true) -> ctx

Production entry point selecting among the validated draw designs. Returns the SAME NamedTuple
shape `d20_real_setup` already returns (so every existing downstream function -- evaluate_fullA,
compute_winners, build_pivot_elimination, composite_gradient_at_fast_buffered,
build_ranged_screen_context, the whole c10_d20_production_driver.jl machinery -- works unchanged),
plus three extra fields: `draw_design`, `draw_seed`, `draw_meta` (see `draw_design_meta`).

`draw_design = :pseudorandom` (the default) takes the EXACT same code path production already
takes -- `Random.seed!(draw_seed); d20_real_setup(...)` (U drawn internally by
`master_prepare_cc`) -- with no change to timing-relevant control flow or RNG semantics.
`draw_design in (:sobol_randomized, :halton_scrambled)` generates the full `W x D` Exp(1) matrix
via `generate_randoms!` and passes it to the SAME `d20_real_setup` as `U`.
`draw_design == :precomputed` requires `U_precomputed` (a `W x D` matrix; set
`precomputed_already_transformed=false` if it is a raw uniform-[0,1) matrix rather than
already-Exp(1)) and routes it through the identical path -- this replaces every historical
diagnostic script's direct call into the now-deleted `d20_real_setup_qmc`.

`log_draw_meta` defaults to `true` (matching this task's "log per-context metadata" requirement)
but can be set `false` for a caller that wants a literal zero-added-instruction :pseudorandom path
(skips the checksum/hash computation entirely).

`exclude_diagonal_gravity`/`σHat` (2026-07-30, sigma=3 campaign prep, merged from
production/fullA-exact) are unrelated to draw design and pass straight through to
`d20_real_setup`'s own kwargs of the same name, identically for every draw design -- exactly one
copy of this wiring exists (in `d20_real_setup`), not one per draw design, which is the invariant
this whole file exists to enforce.
"""
function d20_real_setup_design(; W::Int, δ::Float64 = 1.0, find_smallest::Bool = true,
        draw_design::Symbol = :pseudorandom, draw_seed::Int = 20260719,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
        needs_outer_moment_jacobian::Bool = false, build_screen::Bool = true,
        log_draw_meta::Bool = true,
        U_precomputed::Union{Nothing,AbstractMatrix{Float64}} = nothing,
        precomputed_already_transformed::Bool = true,
        # Provenance only (task §11/§10): a caller that loaded U_precomputed from a persisted
        # artifact should pass its path here so draw_meta records it; this file does not itself
        # implement artifact loading (no such loader exists anywhere in this codebase today --
        # see reachability audit).
        artifact_path::Union{Nothing,AbstractString} = nothing,
        # Part A (2026-07-23): passthrough to d20_real_setup's own destination_sample kwarg
        # (default :exclude_row, matching that function's new default). CM/originZC checkpoint
        # callers (cm_checkpoint.jl, cm_originzc_checkpoint.jl) explicitly pass :all_legacy here
        # since their moment/pivot-elimination layers are not rectangularized in this release.
        destination_sample::Symbol = :exclude_row,
        # exclude_diagonal_gravity (2026-07-30, user-directed fix): passthrough to d20_real_setup's/
        # d20_real_setup_qmc's own kwarg of the same name. `false` default reproduces every
        # pre-existing caller's behavior bit-exactly.
        exclude_diagonal_gravity::Bool = false,
        # gravity_exclude_cells (2026-07-31, Brazil-Korea gravity-exclusion task): passthrough to
        # d20_real_setup's own kwarg of the same name. Empty default reproduces every pre-existing
        # caller bit-exactly.
        gravity_exclude_cells::AbstractVector{<:Tuple{Int,Int}} = Tuple{Int,Int}[],
        # σHat passthrough (2026-07-30, sigma=3 campaign prep) to d20_real_setup's/
        # d20_real_setup_qmc's own kwarg of the same name. `nothing` default reproduces
        # AD_PARAMS.σHat=2.5 unchanged for every pre-existing caller.
        σHat::Union{Nothing,Float64} = nothing,
        # inner_lower_limit passthrough (2026-08-06, lower-limit/hotpath task) to d20_real_setup's
        # own kwarg of the same name -- REQUIRED, no default, for the same reason. Production
        # value -10.0.
        inner_lower_limit::Float64)
    draw_design in VALID_DRAW_DESIGNS ||
        error("d20_real_setup_design: draw_design must be one of $(VALID_DRAW_DESIGNS), got :$(draw_design)")

    D = D20_REAL

    if draw_design == :pseudorandom
        # ---- EXACT existing production call sequence: U is drawn INSIDE d20_real_setup (via
        # master_prepare_cc's internal Random.seed!(seedU); drawU(...)), not here. The outer
        # Random.seed!(draw_seed) below is unchanged pre-existing behavior -- see
        # UNIFIED_DRAW_API_CONTRACT_2026-07-30.md for why this design is not routed through
        # generate_randoms!/an explicit RNG object (that would risk perturbing the exact
        # pseudorandom sequence production has already validated results against; flagged there
        # as a deliberate deferral, not an oversight). Do not add anything ahead of this that
        # could perturb the global RNG state consumed by d20_real_setup's internal draw. ----
        Random.seed!(draw_seed)
        t_ctx = @elapsed ctx0 = d20_real_setup(W = W, δ = δ, find_smallest = find_smallest,
            outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
            needs_outer_moment_jacobian = needs_outer_moment_jacobian, build_screen = build_screen,
            destination_sample = destination_sample, exclude_diagonal_gravity = exclude_diagonal_gravity,
            gravity_exclude_cells = gravity_exclude_cells, σHat = σHat,
            inner_lower_limit = inner_lower_limit)
        timing = (uniform_and_transform = NaN, ctx_build = t_ctx,
                  pairwise = ctx0.screen_setup_wall.pairwise, witness = ctx0.screen_setup_wall.witness)
    else
        design = resolve_draw_design(draw_design, draw_seed;
            U_precomputed = U_precomputed, precomputed_already_transformed = precomputed_already_transformed)
        Uexp = Matrix{Float64}(undef, W, D)
        t_gen = @elapsed generate_randoms!(Uexp, design)
        t_ctx = @elapsed ctx0 = d20_real_setup(W = W, δ = δ, find_smallest = find_smallest,
            outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
            needs_outer_moment_jacobian = needs_outer_moment_jacobian, build_screen = build_screen,
            destination_sample = destination_sample, U = Uexp,
            exclude_diagonal_gravity = exclude_diagonal_gravity, gravity_exclude_cells = gravity_exclude_cells,
            σHat = σHat, inner_lower_limit = inner_lower_limit)
        # Screens and threshold_state are now built ONCE, inside d20_real_setup itself, for
        # every design -- no compensating patch here (task §9: "It must not construct or patch
        # screens, threshold state, context fields").
        timing = (uniform_and_transform = t_gen, ctx_build = t_ctx,
                  pairwise = ctx0.screen_setup_wall.pairwise, witness = ctx0.screen_setup_wall.witness)
    end

    meta = log_draw_meta ? draw_design_meta(draw_design, draw_seed, ctx0.D, W, ctx0.U; timing = timing, artifact_path = artifact_path) : nothing
    return merge(ctx0, (draw_design = draw_design, draw_seed = draw_seed, draw_meta = meta))
end
