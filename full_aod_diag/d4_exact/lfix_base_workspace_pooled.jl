# ============================================================================
# Production-gate addendum, Section 9's fair comparison table requires BOTH candidates to have
# BOTH persistence mechanisms (persistent base workspace + pooled coordinate buffers) -- the
# original price-tensor-elimination report's Backend A benchmark only had the persistent
# workspace ALONE (no GradWorkspacePool). This file is the missing "Backend A+": identical to
# gradient_workspace.jl's own composite_gradient_at_fast_pooled, except sourcing `cache` from
# lfix_base_workspace.jl's persistent build_lfix_base_cache! instead of the allocating
# build_lfix_base_cache -- the two-tensor analogue of composite_gradient_at_Cplus.
# ============================================================================
include(joinpath(@__DIR__, "lfix_base_workspace.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))

"""
    composite_gradient_at_Aplus(x_free0, ctx, pe, grad_pool::GradWorkspacePool, ws::LFixBaseWorkspace;
                                 base=nothing, threaded=false, h_mode=:cached, bandwidth_cache=nothing, multi_method=:top3) -> (g, meta)

"Backend A+": persistent two-tensor base cache (via `ws`, this session's own `LFixBaseWorkspace`)
+ the SAME per-coordinate `GradWorkspacePool` production already uses. Identical to
`composite_gradient_at_fast_pooled` except line 1: `build_lfix_base_cache!(ws, ...)` instead of
the allocating `build_lfix_base_cache(...)`.
"""
function composite_gradient_at_Aplus(x_free0::AbstractVector, ctx, pe, pool::GradWorkspacePool, ws::LFixBaseWorkspace;
        base::Union{Nothing,BaseDualState} = nothing, threaded::Bool = false,
        h_mode::Symbol = :cached, h0::Float64 = 0.01,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing,
        multi_method::Symbol = :top3)
    h_mode in (:fixed, :cached) || error("composite_gradient_at_Aplus: h_mode must be :fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_Aplus: h_mode=:cached requires a bandwidth_cache Dict")

    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache!(ws, x_free0, ctx, base; validate_dense = false)
    D = ctx.D; D2 = D^2; W = cache.W
    z0 = log.(reshape(x_free0[2:end], D, D))
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
            g[k] = a_block_fd_component_ws!(tws, cache, ctx, pe, w0, k, h; multi_method = multi_method)
        else
            local h, is_hit
            lock(bandwidth_cache_lock) do
                is_hit = haskey(bandwidth_cache, k)
                h = is_hit ? bandwidth_cache[k] : NaN
            end
            if is_hit
                cache_hits[k] = true
            else
                h, _, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
                lock(bandwidth_cache_lock) do
                    bandwidth_cache[k] = h
                end
                cache_hits[k] = false
            end
            h_used[k] = h
            g[k] = a_block_fd_component_ws!(tws, cache, ctx, pe, w0, k, h; multi_method = multi_method)
        end
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
               threaded = threaded, cache_hits = cache_hits, tie_fallback = false)
end
