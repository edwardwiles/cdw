# ============================================================================
# Overnight task (2026-07-22), Priority 1: CM-aware C+ (factorized) gradient backend.
#
# See docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md for the full derivation this file
# implements. Summary: the CM-augmented base-point dual scalar
#     q_s(theta) = -zeta* - lambda_G*'G_s(theta) - lambda_C*'C_s
# has a CONSTANT `lambda_C*'C_s` term across every outer coordinate probe (lfix_cm_aware.jl's
# own already-proven result -- NOT re-derived here, `cm_fixed_contribution` is reused
# verbatim). `lfix_cm_aware.jl::build_lfix_base_cache_cm` proved this decomposition survives
# swapping the Reference economic-block builder (`build_lfix_base_cache`) for ANY backend that
# (a) only ever indexes `base.lambda*[1:D^2]` for the economic core columns and (b) uses the
# identical counterfactual-column tail check `oci-1 >= D^2+1`. Backend C+'s
# `build_lfix_base_cache_C`/`build_lfix_base_cache_C!` (lfix_factorized.jl,
# lfix_factorized_workspace.jl) satisfy both conditions by direct inspection (see the trace
# doc's Section 5) -- this file composes them with the SAME `cm_fixed_contribution` the
# Reference CM path already uses, changing NEITHER.
#
# PURELY ADDITIVE: does not modify lfix_cm_aware.jl, lfix_factorized.jl,
# lfix_factorized_workspace.jl, composite_gradient_at_Cplus, or cm_production_bundle.jl. Every
# function here is new. `cm_gradient_backend=:reference` (the production default, unchanged)
# never reaches any function defined in this file.
# ============================================================================
include(joinpath(@__DIR__, "gradient_workspace.jl"))          # GradWorkspacePool, GradWorkspace, build_grad_workspace_pool, resize_pool_if_needed!
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))   # LFixFactorizedWorkspace, LFixBaseCacheC, build_lfix_base_cache_C!, lfix_incremental_at_Cplus!, a_block_fd_component_Cplus!, select_bandwidth_C, gamma_component_analytic(::LFixBaseCacheC,...)
# Expected already included by caller (matches lfix_cm_aware.jl's own convention, not
# re-included here to avoid depending on include-idempotence for files with heavier deps):
#   cm_lookup_kernels.jl, lfix_cm_aware.jl (cm_fixed_contribution, cm_bin_indices_for),
#   cm_production_bundle.jl (archC_base_state, archC_verified_state, CMExpectedSolveFailure,
#   build_cm_production_context)

"""
    with_q0_C(cache::LFixBaseCacheC, q0_new::Vector{Float64}) -> LFixBaseCacheC

Field-generic analog of `lfix_cm_aware.jl::with_q0`, for Backend C/C+'s cache struct
(`LFixBaseCacheC`, which also carries its own `q0::Vector{Float64}` field, `lfix_factorized.jl:72`).
Built from `fieldnames(LFixBaseCacheC)`, not a hardcoded positional list, for the same
future-proofing reason `with_q0` gives.
"""
function with_q0_C(cache::LFixBaseCacheC, q0_new::Vector{Float64})
    vals = Any[f === :q0 ? q0_new : getfield(cache, f) for f in fieldnames(LFixBaseCacheC)]
    return LFixBaseCacheC(vals...)
end

"""
    build_lfix_base_cache_cm_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCacheC

CM-aware analog of `lfix_cm_aware.jl::build_lfix_base_cache_cm`, for Backend C+. Same contract:
`ctx_cm.obj === aug.obj_cm` (the CM-augmented objective `base` was actually solved against).
Calls `build_lfix_base_cache_C!` UNCHANGED against `ctx_cm` (see this file's header / the trace
doc for why this silently and correctly ignores the CM tail of `lambda*`), then folds in the SAME
`cm_fixed_contribution` (lfix_cm_aware.jl) the Reference CM path uses -- one shared CM-algebra
implementation for both backends, not a second derivation. `ctx`/`aug`/`bins` are the plain
(non-CM-swapped) context/aug/bins, exactly as `build_lfix_base_cache_cm` requires (kept separate
from `ctx_cm` because `aug.origins`/`aug.z` are computed off the plain `ctx.U`).
"""
function build_lfix_base_cache_cm_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                      base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned};
                                      validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    cm_contrib0 = cm_fixed_contribution(base, ctx, aug, bins)
    return with_q0_C(cache0, cache0.q0 .- cm_contrib0)
end

"""
    composite_gradient_at_Cplus_from_cache(x_free0, ctx, pe, pool, cache; base, threaded, h_mode, h0, bandwidth_cache, multi_method) -> (g, meta)

Structural twin of `lfix_factorized_workspace.jl::composite_gradient_at_Cplus`, differing ONLY in
accepting a PRE-BUILT `cache::LFixBaseCacheC` instead of building one internally via
`build_lfix_base_cache_C!` -- the one thing `composite_gradient_at_Cplus` itself does not allow
(it has no `cache=` override point, unlike `composite_gradient_at_fast`'s established `cache=`
pattern). This is the ONLY reason this near-duplicate exists: production's own
`composite_gradient_at_Cplus` (the function the live unrestricted-path driver calls) is never
touched. Every other line (coordinate loop, threading discipline, `:static` scheduling,
bandwidth-cache locking) is copied verbatim from `composite_gradient_at_Cplus` so the two stay
comparable for benchmarking.
"""
function composite_gradient_at_Cplus_from_cache(x_free0::AbstractVector, ctx, pe, pool::GradWorkspacePool,
        cache::LFixBaseCacheC; base::BaseDualState,
        threaded::Bool = false, h_mode::Symbol = :cached, h0::Float64 = 0.01,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing,
        multi_method::Symbol = :top3)
    h_mode in (:fixed, :cached) || error("composite_gradient_at_Cplus_from_cache: h_mode must be :fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_Cplus_from_cache: h_mode=:cached requires a bandwidth_cache Dict")

    D = ctx.D
    # Ddest (destination count) -- Ddest==D unless row_idx excludes ROW (Part A, 2026-07-23).
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    D2 = D * Ddest; W = cache.W
    z0 = log.(reshape(x_free0[2:end], D, Ddest))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])
    h_used = zeros(D2); cache_hits = falses(D2)

    nT = Threads.maxthreadid()
    resize_pool_if_needed!(pool, W; nT = nT)
    bandwidth_cache_lock = ReentrantLock()

    function do_coord!(k::Int)
        tid = Threads.threadid()
        tws = pool.slots[tid]
        if h_mode == :fixed
            h = h0
            h_used[k] = h
        else
            local h, is_hit
            lock(bandwidth_cache_lock) do
                is_hit = haskey(bandwidth_cache, k)
                h = is_hit ? bandwidth_cache[k] : NaN
            end
            if is_hit
                cache_hits[k] = true
            else
                h, _, _ = select_bandwidth_C(cache, ctx, pe, w0, k)
                lock(bandwidth_cache_lock) do
                    bandwidth_cache[k] = h
                end
                cache_hits[k] = false
            end
            h_used[k] = h
        end
        g[k] = a_block_fd_component_Cplus!(tws, cache, ctx, pe, w0, k, h_used[k])
        return nothing
    end

    if threaded
        CS.guard_enter_coord_pool!()
        try
            Threads.@threads :static for k in 2:D2
                do_coord!(k)
            end
        finally
            CS.guard_exit_coord_pool!()
        end
    else
        for k in 2:D2
            do_coord!(k)
        end
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, h_mode = h_mode,
               threaded = threaded, cache_hits = cache_hits)
end

"""
    composite_gradient_at_Cplus_cm(x_free0, ctx_cm, pe, ctx, aug, bins, pool, ws; base=nothing, cache=nothing, kwargs...) -> (g, meta)

CM-aware entry point for Backend C+, structural twin of `lfix_cm_aware.jl::composite_gradient_at_fast_cm`
(which wraps the Reference `composite_gradient_at_fast`) but wrapping
`composite_gradient_at_Cplus_from_cache` instead. `base` defaults to `archC_base_state` (the SAME
Architecture-C inner-solve path `cm_production_gradient` uses -- the inner-solve/Hessian-backend
choice is orthogonal to the outer gradient backend selected here, see the trace doc Section 1).
"""
function composite_gradient_at_Cplus_cm(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned},
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace, cctx;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archC_base_state(x_free0, ctx_cm, cctx) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end

"""
    cm_production_gradient_cplus(x_free0, pcx, ctx, pe, pool, ws; base=nothing, kwargs...) -> (g, meta)

`:cplus`-backend analog of `cm_production_bundle.jl::cm_production_gradient`, the production
CM `cb_G!` entry point when `cm_gradient_backend=:reference` (unchanged default). `pcx` is the
SAME `build_cm_production_context(...)` return value both backends share; `pool`/`ws` are the
caller-owned `GradWorkspacePool`/`LFixFactorizedWorkspace` (built once per `(D,W)`, threaded
through every call, matching every other `_Cplus`/`_KBplus`/`_Aplus` production entry point's own
convention in `c10_d20_production_driver.jl`).
"""
function cm_production_gradient_cplus(x_free0::AbstractVector, pcx, ctx, pe, pool::GradWorkspacePool,
        ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing, kwargs...)
    base = base === nothing ? archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx) : base
    cache = build_lfix_base_cache_cm_C!(ws, x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
    return composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
end
