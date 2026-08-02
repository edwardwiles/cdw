# ============================================================================
# Typed draw-design representation (task "unify random-draw production pipeline",
# 2026-07-30, §2-3). Resolves the existing Symbol-based public interface
# (:pseudorandom / :sobol_randomized / :halton_scrambled, plus the new
# :precomputed) into one of these types EXACTLY ONCE, at
# draw_design.jl::d20_real_setup_design -- nothing downstream of that
# resolution point branches on draw design again; every design produces a
# final W x D Exp(1) matrix through the SAME generate_randoms! call surface
# and the SAME context/setup functions.
#
# Kept as a Symbol-facing public API (VALID_DRAW_DESIGNS in draw_design.jl is
# unchanged) rather than exposing these types to every one of the ~100
# existing call sites -- checkpoints already persist draw_design as a Symbol
# (provenance metadata, see docs/QMC_PSEUDORANDOM_DUPLICATION_REACHABILITY_2026-07-30.md
# §"DRAW_METADATA_REQUIRED"), and that convention is correct and left alone.
# ============================================================================

abstract type DrawDesign end

"Julia default RNG (MersenneTwister/Xoshiro256++ depending on Julia version, via `Random.rand!`) --
production baseline. Global-RNG-state semantics (Random.seed! before drawing) are UNCHANGED from
pre-existing production behavior; see UNIFIED_DRAW_API_CONTRACT_2026-07-30.md for why this design
does not go through the RNG-state-purity refactor the other three designs already have."
struct PseudorandomDesign <: DrawDesign
    seed::Int
end

"Sobol.jl `SobolSeq` deterministic base sequence + an independent Cranley-Patterson random shift
(mod 1) per seed -- a plain randomized-QMC shift, NOT Owen/digital scrambling (Sobol.jl does not
implement digital scrambling). Does not mutate caller-visible global RNG state."
struct RandomizedSobolDesign <: DrawDesign
    seed::Int
end

"`cc_algo/rhalton.jl` scrambled Halton sequence -- genuine Owen-style per-digit scrambling.
Does not mutate caller-visible global RNG state."
struct ScrambledHaltonDesign <: DrawDesign
    seed::Int
end

"""
A caller-supplied draw matrix (diagnostic synthetic matrices, persisted historical artifacts, or
a same-seed cross-design comparison matrix) routed through the SAME unified pipeline as every
generated design, per task §10 ("Do not create `_qmc` context functions for injected matrices").

`already_transformed=true` (default): `U` is the final `W x D` Exp(1) matrix, used as-is.
`already_transformed=false`: `U` is a `W x D` uniform-[0,1) matrix; `generate_randoms!` applies
the shared `transform_unit01_to_exp1!` before use -- the manifest requirement (task §10, "prevent
an already transformed artifact from being transformed twice") is enforced by this single flag
rather than by inspecting the matrix's own value range, which cannot reliably distinguish the two
cases (a legitimate uniform-[0,1) draw and a legitimate Exp(1) draw can both, in principle,
contain any nonnegative-adjacent values near the boundary).
"""
struct PrecomputedDrawDesign <: DrawDesign
    U::Matrix{Float64}
    already_transformed::Bool
    label::String
end
PrecomputedDrawDesign(U::Matrix{Float64}; already_transformed::Bool = true, label::AbstractString = "precomputed") =
    PrecomputedDrawDesign(U, already_transformed, String(label))

"Round-trip Symbol <-> DrawDesign, matching draw_design.jl::VALID_DRAW_DESIGNS."
design_symbol(::PseudorandomDesign) = :pseudorandom
design_symbol(::RandomizedSobolDesign) = :sobol_randomized
design_symbol(::ScrambledHaltonDesign) = :halton_scrambled
design_symbol(::PrecomputedDrawDesign) = :precomputed

"""
    resolve_draw_design(design::Symbol, seed::Int; U_precomputed=nothing,
                         precomputed_already_transformed=true, precomputed_label="precomputed")
        -> DrawDesign

The one place a Symbol (or, per task §2, a String -- accepted via `Symbol(design)` by the caller)
becomes a typed `DrawDesign`. Called exactly once, in `d20_real_setup_design`, before any draw is
generated.
"""
function resolve_draw_design(design::Symbol, seed::Int;
        U_precomputed::Union{Nothing,AbstractMatrix{Float64}} = nothing,
        precomputed_already_transformed::Bool = true,
        precomputed_label::AbstractString = "precomputed")
    design == :pseudorandom && return PseudorandomDesign(seed)
    design == :sobol_randomized && return RandomizedSobolDesign(seed)
    design == :halton_scrambled && return ScrambledHaltonDesign(seed)
    if design == :precomputed
        U_precomputed === nothing &&
            error("resolve_draw_design: draw_design=:precomputed requires U_precomputed to be given")
        return PrecomputedDrawDesign(Matrix{Float64}(U_precomputed), precomputed_already_transformed, String(precomputed_label))
    end
    error("resolve_draw_design: unknown draw_design :$(design)")
end

"""
    generate_randoms!(Uexp::Matrix{Float64}, design::DrawDesign) -> Uexp

The one authoritative draw API (task §3): caller preallocates `Uexp` (`W x D`, `Float64`), this
fills it with the final Exp(1)-transformed draw matrix for `design`. Every design writes the same
shape/scalar type/storage layout; the inverse-CDF transform is the single shared
`transform_unit01_to_exp1!` (prepare_cc/genRands.jl) in every branch, not reimplemented per design.

Note on allocation (see DRAW_PIPELINE_W500K_RESOURCE_GATE_2026-07-30.csv for measurements):
the :pseudorandom branch is fully in-place (zero extra allocation beyond `Uexp` itself, via the
pre-existing `genExpRands!`). The Sobol/Halton branches call the pre-existing `sobol_U`/`halton_U`
generators (qmc_draws.jl, UNCHANGED numerics) and `copyto!` their result into `Uexp` -- one
top-level copy, not newly introduced by this refactor (those generators already allocate several
internal temporaries of their own; this does not add to that).
"""
function generate_randoms!(Uexp::Matrix{Float64}, design::PseudorandomDesign)
    Random.seed!(design.seed)
    genExpRands!(Uexp)
    return Uexp
end

function generate_randoms!(Uexp::Matrix{Float64}, design::RandomizedSobolDesign)
    W, D = size(Uexp)
    copyto!(Uexp, sobol_U(W, D; seed = design.seed))
    return Uexp
end

function generate_randoms!(Uexp::Matrix{Float64}, design::ScrambledHaltonDesign)
    W, D = size(Uexp)
    copyto!(Uexp, halton_U(W, D; seed = design.seed))
    return Uexp
end

function generate_randoms!(Uexp::Matrix{Float64}, design::PrecomputedDrawDesign)
    size(design.U) == size(Uexp) ||
        error("generate_randoms!: PrecomputedDrawDesign matrix is $(size(design.U)), expected $(size(Uexp))")
    copyto!(Uexp, design.U)
    design.already_transformed || transform_unit01_to_exp1!(Uexp)
    return Uexp
end
