# ============================================================================
# Fixed-Fréchet-aware Lfix/composite-gradient outer path (port-prep
# 2026-07-24). Structural analogue of `lfix_cm_aware.jl`'s
# `cm_fixed_contribution`/`build_lfix_base_cache_cm`/`composite_gradient_at_fast_cm`
# for the fixed-Fréchet CDF+POWER block, task brief §4/§5's
# `∂h_ω^FF/∂A = 0` requirement: the appended block's dual contribution is
# folded into `LFixBaseCache.q0` ONCE at the base point, exactly like
# flexible-CM's own CM block, so the standard (non-restriction-aware)
# `lfix_incremental_at`/`dest_contrib_*`/`gamma_component_analytic` machinery
# is reused completely UNCHANGED for the per-coordinate outer-gradient sweep.
#
# Simplification versus `cm_fixed_contribution`: that function computes
# `λ_C*'C_s` via a specialized O(W*(D-1)) cumulative-suffix-sum LOOKUP kernel
# (`cm_lookup_kernels.jl`) specifically because flexible-CM's production fast
# path (Architecture B) never materializes a dense `W x ncm` CM matrix. This
# port's `:cdf_power` feature set DOES use the dense (Architecture A) moment
# path (`build_cm_frechet_augmented_obj_basis`, see
# FIXED_FRECHET_STRUCTURED_POWER_HESSIAN_2026-07-24.md §5's disclosed
# scoping) -- `aug.CM` (a genuine `W x ncm` matrix) is therefore ALREADY
# resident in memory, so `frechet_fixed_contribution` below is a plain dense
# matrix-vector product (`aug.CM * λ_ff`, O(W*ncm)) rather than a bespoke
# lookup kernel -- simpler, and correct by construction from the SAME `CM`
# matrix already validated against the structured Hessian (gate P2,
# test_frechet_power_hessian_d4_gates.jl).
# ============================================================================

"""
    frechet_fixed_contribution(base::BaseDualState, aug) -> Vector{Float64}

`out[s] = λ_FF*' h^FF_s` for every draw `s` -- the fixed-Fréchet block's
constant dual contribution at the base point, via a dense matvec against
`aug.CM` (the `W x ncm` moment matrix `build_cm_frechet_augmented_obj_basis`
already materializes). `base.λstar`'s Fréchet-block sub-vector is sliced at
`aug.ncore : aug.ncore-1+aug.ncm`, the same layout convention
`wrap_moments_with_cm`/`build_cm_frechet_augmented_obj_basis` establish (core
columns `1:ncore-1`, appended block `ncore:ncore-1+ncm`, gravity last).
"""
function frechet_fixed_contribution(base::BaseDualState, aug)
    ncore = aug.ncore; ncm = aug.ncm
    @assert length(base.λstar) >= ncore - 1 + ncm "base.λstar too short for aug's (ncore,ncm) -- was base solved against aug.obj_cm?"
    λ_ff = base.λstar[ncore:ncore-1+ncm]
    return aug.CM * λ_ff
end

"""
    build_lfix_base_cache_cm_frechet(x_free0, ctx_cm, base, aug; validate_dense=false) -> LFixBaseCache

Fixed-Fréchet analogue of `build_lfix_base_cache_cm`. `ctx_cm.obj` must be
the SAME `aug.obj_cm` `base` was solved against. `build_lfix_base_cache`
(generic, unchanged) is called first -- it structurally only ever indexes
`base.λstar[1:D*D_dest]` (the core block), so it silently and correctly
ignores the appended Fréchet tail, exactly as it does for flexible-CM's own
CM tail (see `lfix_cm_aware.jl`'s docstring for the same argument, verified
there, not re-derived independently here).
"""
function build_lfix_base_cache_cm_frechet(x_free0::AbstractVector, ctx_cm, base::BaseDualState, aug;
                                           validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    ff_contrib0 = frechet_fixed_contribution(base, aug)
    return with_q0(cache0, cache0.q0 .- ff_contrib0)
end

"""
    cm_frechet_production_gradient(x_free0, fpcx, pe; base=nothing, kwargs...) -> (g, meta)

Reference-backend (non-C+) outer-gradient entry point for
`marginal_mode=:frechet_reference, frechet_feature_set=:cdf_power`. Thin
wrapper: builds (or reuses a caller-supplied) `base`/`cache`, delegates all
per-coordinate work to the UNCHANGED `composite_gradient_at_fast`. Task
brief §8's state-reuse discipline applies here identically to
`cm_frechet_base_state` -- pass `base=` whenever the caller already holds a
verified `BaseDualState` for this exact point.

NOT YET BUILT in this port-prep pass: a `:cplus`-backend analogue
(`cm_frechet_production_gradient_cplus`) mirroring `lfix_cm_cplus.jl`'s
`cm_production_gradient_cplus`, and a fast-lookup (non-dense-matvec)
contribution kernel for the `:cdf_only` archB moment path. Both are
disclosed follow-ups in the port-readiness report, not silent gaps -- this
reference-backend path is what the D=20 gates and outer shakedown in this
branch actually exercise.
"""
function cm_frechet_production_gradient(x_free0::AbstractVector, fpcx, pe;
        base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    fpcx.mode === :frechet_reference || error("cm_frechet_production_gradient: fpcx.mode=$(fpcx.mode), expected :frechet_reference")
    fpcx.cfg.frechet_feature_set === :cdf_power || error("cm_frechet_production_gradient: only :cdf_power is wired in this port-prep pass (got :$(fpcx.cfg.frechet_feature_set))")
    base = base === nothing ? cm_frechet_base_state(x_free0, fpcx) : base
    cache = build_lfix_base_cache_cm_frechet(x_free0, fpcx.ctx_cm, base, fpcx.aug)
    return composite_gradient_at_fast(x_free0, fpcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
end
