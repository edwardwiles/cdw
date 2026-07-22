# ============================================================================
# Finalization task Phase 4: persistent-workspace ("+") version of Backend :kbplus, mirroring
# lfix_factorized_workspace.jl's Backend C+ design exactly -- one persistent workspace per
# context, refilled in place, zero allocation on a warm call at a fixed (D,W). The only new
# array vs Backend C+'s own workspace is `USigmaPow` (W×D).
# ============================================================================
include(joinpath(@__DIR__, "lfix_kbplus.jl"))

"""
    LFixKBPlusWorkspace

Persistent backing store for Backend :kbplus's arrays: `WinnerRefCache`'s own (`logCC0`,
`mulU`, winner/runnerup/third indices+scores, `margin`) plus the L_fix-specific bookkeeping
(same as `LFixFactorizedWorkspace`) plus `USigmaPow` (W×D, `Uσ^(-μ)`, the one array this
backend needs that C+ does not).
"""
mutable struct LFixKBPlusWorkspace
    D::Int
    W::Int
    valid::Bool
    logCC0::Matrix{Float64}
    mulU::Matrix{Float64}
    winner::Matrix{Int}
    sw::Matrix{Float64}
    runnerup::Matrix{Int}
    sr::Matrix{Float64}
    third::Matrix{Int}
    st3::Matrix{Float64}
    margin::Matrix{Float64}
    USigmaPow::Matrix{Float64}
    SW::Vector{Float64}
    denom::Vector{Float64}
    CONST_d::Vector{Float64}
    contrib0::Matrix{Float64}
    q0::Vector{Float64}
    Uσ_bi::Vector{Float64}
    cf_contrib0::Vector{Float64}
end

"`build_lfix_kbplus_workspace(D, W)` -- one-time allocation of every persistent array Backend :kbplus needs."
function build_lfix_kbplus_workspace(D::Int, W::Int)
    return LFixKBPlusWorkspace(D, W, false,
        Matrix{Float64}(undef, D, D), Matrix{Float64}(undef, W, D),
        Matrix{Int}(undef, W, D), Matrix{Float64}(undef, W, D),
        Matrix{Int}(undef, W, D), Matrix{Float64}(undef, W, D),
        Matrix{Int}(undef, W, D), Matrix{Float64}(undef, W, D),
        Matrix{Float64}(undef, W, D),
        Matrix{Float64}(undef, W, D),
        Vector{Float64}(undef, W), Vector{Float64}(undef, D), Vector{Float64}(undef, D),
        Matrix{Float64}(undef, W, D), Vector{Float64}(undef, W),
        Vector{Float64}(undef, W), Vector{Float64}(undef, W))
end

"Mirror of `ensure_lfix_factorized_workspace!` for :kbplus -- rebuilds only on a genuine (D,W) change."
function ensure_lfix_kbplus_workspace!(ws_ref::Base.RefValue{LFixKBPlusWorkspace}, D::Int, W::Int)
    ws = ws_ref[]
    if ws.D != D || ws.W != W
        ws_ref[] = build_lfix_kbplus_workspace(D, W)
    end
    return ws_ref[]
end

"""
    build_winner_ref_KB!(ws, x_free0, ctx; check_ties=true) -> WinnerRefCache

In-place-backed ranking scan for :kbplus -- IDENTICAL formulas/order to
`lfix_factorized_workspace.jl::build_winner_ref!` (ranking is unaffected by the ratio-vs-exp
value-reconstruction change), writing into this file's own `LFixKBPlusWorkspace` buffers.
"""
function build_winner_ref_KB!(ws::LFixKBPlusWorkspace, x_free0::AbstractVector, ctx; check_ties::Bool = true)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    D = ctx.D; U = ctx.U; W = size(U, 1)
    (ws.D == D && ws.W == W) || throw(DimensionMismatch(
        "build_winner_ref_KB!: workspace is (D=$(ws.D),W=$(ws.W)), context needs (D=$D,W=$W) -- call ensure_lfix_kbplus_workspace! first"))
    μ = θ_full0[1]; σ = θ_full0[2]
    constCons0, _, _ = constCons_matrix(θ_full0, ctx)
    ws.logCC0 .= log.(constCons0)
    ws.mulU .= μ .* log.(U)
    UPow = U .^ (-μ)   # transient, O(W*D), matches build_winner_ref's own tie-check factor -- not persisted (only needed here)

    winner = ws.winner; sw = ws.sw; runnerup = ws.runnerup; sr = ws.sr
    third = ws.third; st3 = ws.st3; margin = ws.margin
    scol = Vector{Float64}(undef, D)   # O(D), negligible -- not the allocation target
    n_tied = 0; tied = Tuple{Int,Int}[]
    @inbounds for d in 1:D
        for s in 1:W
            for o in 1:D
                scol[o] = ws.logCC0[o, d] + ws.mulU[s, o]
            end
            i1, s1, i2, s2, i3, s3 = top3_scan(scol)
            winner[s, d] = i1; sw[s, d] = s1
            runnerup[s, d] = i2; sr[s, d] = s2
            third[s, d] = i3; st3[s, d] = s3
            margin[s, d] = s2 - s1
            if check_ties
                pmin = constCons0[1, d] / UPow[s, 1]
                for o in 2:D
                    p = constCons0[o, d] / UPow[s, o]
                    (p < pmin) && (pmin = p)
                end
                c = 0
                for o in 1:D
                    (constCons0[o, d] / UPow[s, o] <= pmin) && (c += 1)
                end
                if c > 1
                    n_tied += 1
                    length(tied) < 5 && push!(tied, (s, d))
                end
            end
        end
    end
    check_ties && n_tied > 0 && throw(TiedWinnerError(n_tied, tied))
    return WinnerRefCache(D, W, μ, σ, ws.logCC0, ws.mulU, winner, sw, runnerup, sr,
                          third, st3, margin, collect(x_free0), θ_full0)
end

"""
    build_lfix_base_cache_KB!(ws, x_free0, ctx, base; validate_dense=false) -> LFixBaseCacheKB

In-place-backed twin of `lfix_kbplus.jl::build_lfix_base_cache_KB`. Same `ws.valid` lifecycle
discipline as Backend A/C+'s own workspaces.
"""
function build_lfix_base_cache_KB!(ws::LFixKBPlusWorkspace, x_free0::AbstractVector, ctx, base::BaseDualState; validate_dense::Bool = false)
    ws.valid = false
    obj = ctx.obj
    D = ctx.D; W = size(obj.U, 1); oci = obj.outer_constr_index
    μ = base.θ_full0[1]; σ = ctx.σ; bi = ctx.bi
    γo = ctx.γ
    gammafac = spgamma(μ * (1 - σ) + 1)
    ws.SW .= @view γo.SamplingWeights[1:W]
    for d in 1:D
        ws.denom[d] = γo.wHat[d] * γo.L[d]
    end
    λstar = base.λstar

    ref = build_winner_ref_KB!(ws, x_free0, ctx)
    ws.USigmaPow .= @view(γo.Uσ[1:W, :]) .^ (-μ)

    constCons0, _, _ = constCons_matrix(base.θ_full0, ctx)
    constConsσ0 = constCons0 .^ (1 - σ)

    CONST_d = ws.CONST_d
    for d in 1:D
        s = 0.0
        for o in 1:D
            d1 = d + (o - 1) * D
            s += λstar[d1] * (-γo.P[d1] * ws.denom[d])
        end
        CONST_d[d] = s
    end

    contrib0 = ws.contrib0
    @inbounds for d in 1:D, ω in 1:W
        wo = ref.winner[ω, d]
        d1w = d + (wo - 1) * D
        pTσ_wo = pTσ_from_ratio(constConsσ0[wo, d], ws.USigmaPow[ω, wo])
        contrib0[ω, d] = (ws.SW[ω] / gammafac) * (CONST_d[d] + λstar[d1w] * pTσ_wo)
    end

    wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
    wPrime_bi = wPrime[bi]
    τPrime_bi = γo.τPrime[bi, bi]
    LPrime_bi = γo.LPrime[bi]
    ws.Uσ_bi .= @view(γo.Uσ[:, bi]) .^ (-μ)
    d1_cf = D^2 + 1
    λ_cf = oci - 1 >= d1_cf ? λstar[d1_cf] : 0.0

    AodPow_bibi0 = aod_pow_cell(base.θ_full0, ctx, bi, bi)
    γ_prime_bi0 = base.θ_full0[3+D]
    constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi0 * τPrime_bi)^(1 - σ)
    denom_cf0 = γ_prime_bi0^σ * wPrime_bi_gdp(wPrime_bi, LPrime_bi)
    ws.cf_contrib0 .= λ_cf .* ((constConsσ_bibi ./ ws.Uσ_bi .- denom_cf0) ./ gammafac .* ws.SW)

    q0 = ws.q0
    @inbounds for s in 1:W
        q0[s] = -base.ζstar - sum(@view(contrib0[s, :])) - ws.cf_contrib0[s]
    end

    if validate_dense
        K = zeros(W); Gfull = zeros(W, obj.d)
        obj.moments!(K, Gfull, base.θ_full0, obj.U, obj)
        maxerr = 0.0
        @inbounds for s in 1:W
            q0_true_s = -base.ζstar - dot(λstar, @view(Gfull[s, 1:oci-1]))
            maxerr = max(maxerr, abs(q0_true_s - q0[s]))
        end
        maxerr < 1e-8 || error("build_lfix_base_cache_KB!: self-validation FAILED, max|q0_true-q0_cache|=$maxerr")
    end

    ws.valid = true
    return LFixBaseCacheKB(D, oci, W, μ, σ, bi, gammafac, ws.SW, ws.denom, ws.CONST_d, ref, ws.USigmaPow, contrib0,
        λstar, base.ζstar, q0, wPrime_bi, τPrime_bi, LPrime_bi, ws.Uσ_bi, λ_cf, ws.cf_contrib0)
end

"In-place variant of `dest_contrib_incremental_top3_KB`: writes into caller-supplied `contrib_buf`."
function dest_contrib_incremental_top3_KB!(contrib_buf::AbstractVector, cache::LFixBaseCacheKB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W; σ = cache.σ
    ref = cache.ref
    Cd = changed_origins
    constCons′, logCC′, _ = constCons_matrix(θ_full, ctx)
    constConsσ′ = constCons′ .^ (1 - σ)
    @inbounds for ω in 1:W
        r1 = ref.winner[ω, d]; r2 = ref.runnerup[ω, d]; r3 = ref.third[ω, d]
        best_o = 0; best_s = Inf
        if !(r1 in Cd)
            best_o = r1; best_s = ref.sw[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_s = ref.sr[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_s = ref.st3[ω, d]
        end
        bo = best_o; bs = best_s
        for o in Cd
            v = logCC′[o, d] + ref.mulU[ω, o]
            # Remediation task Part E (finding F5): canonical exact-tie convention --
            # lowest origin index wins (matches every generic-rescan tier's own natural
            # behavior, which scans o=1..D with strict `<`). Previously `if v < bs` alone,
            # which on an exact tie kept whichever candidate was considered FIRST (the
            # cached top-3 survivor, regardless of its index vs the tying changed origin's)
            # -- disagreed with the generic-rescan tiers whenever the survivor's index
            # exceeded the tying origin's. Measure-zero in exact arithmetic; see
            # test_winner_forced_tie.jl.
            if v < bs || (v == bs && o < bo)
                bs = v; bo = o
            end
        end
        if bo == 0
            bo = 1; bs = logCC′[1, d] + ref.mulU[ω, 1]
            for o in 2:D
                v = logCC′[o, d] + ref.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
        end
        pTσ_wo = pTσ_from_ratio(constConsσ′[bo, d], cache.USigmaPow[ω, bo])
        d1w = d + (bo - 1) * D
        contrib_buf[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib_buf
end

"In-place variant of `cf_contrib_at` for `LFixBaseCacheKB` (identical formula to C+'s own `cf_contrib_at_C!`, retyped)."
function cf_contrib_at_KB!(buf::AbstractVector, cache::LFixBaseCacheKB, θ_full::AbstractVector, ctx)
    bi = cache.baseIndex; σ = cache.σ
    AodPow_bibi = aod_pow_cell(θ_full, ctx, bi, bi)
    γ_prime_bi = θ_full[3+ctx.D]
    constConsσ_bibi = cache.wPrime_bi^(1 - σ) * (AodPow_bibi * cache.τPrime_bi)^(1 - σ)
    denom_cf = γ_prime_bi^σ * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi)
    buf .= cache.λ_cf .* ((constConsσ_bibi ./ cache.Uσ_bi .- denom_cf) ./ cache.gammafac .* cache.SW)
    return buf
end

"""
    lfix_incremental_at_KBplus!(ws::GradWorkspace, cache, ctx, pe, w0, coord_idx, new_val) -> Float64

Buffer-aware twin of `lfix_kbplus.jl::lfix_incremental_at_KB`, using `ws.q`/`ws.psi`/
`ws.contrib`/`ws.cf` (the SAME `GradWorkspace` slots C+'s own pooled path uses).
"""
function lfix_incremental_at_KBplus!(ws::GradWorkspace, cache::LFixBaseCacheKB, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64)
    w = copy(w0); w[coord_idx] = new_val   # O(D^2), negligible
    z = pivot_expand(w[2:end], pe)
    Aod_theta = exp.(z)
    x_free = vcat(w[1], vec(Aod_theta))
    θ_full = CS.reconstruct_full(x_free, ctx.m)

    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache.baseIndex && d == cache.baseIndex, cells)

    copyto!(ws.q, cache.q0)
    for d in affected_dests
        old_contrib = @view cache.contrib0[:, d]
        origins_here = [o for (o, dd) in cells if dd == d]
        dest_contrib_incremental_top3_KB!(ws.contrib, cache, ctx, θ_full, d, origins_here)
        ws.q .-= ws.contrib .- old_contrib
    end
    if cf_touched
        cf_contrib_at_KB!(ws.cf, cache, θ_full, ctx)
        ws.q .-= ws.cf .- cache.cf_contrib0
    end

    CS.Psi!(ws.psi, ws.q)
    return -(sum(ws.psi) / length(ws.q) + cache.ζstar)
end

"Buffer-aware twin of `lfix_kbplus.jl::a_block_fd_component_KB`, including the AUD-12 nonfinite-both-sides retry discipline."
function a_block_fd_component_KBplus!(ws::GradWorkspace, cache::LFixBaseCacheKB, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64; max_h_shrinks::Int = 4)
    h_try = h
    for attempt in 1:(max_h_shrinks + 1)
        Lp = lfix_incremental_at_KBplus!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h_try)
        Lm = lfix_incremental_at_KBplus!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h_try)
        if isfinite(Lp) && isfinite(Lm)
            return (Lp - Lm) / (2h_try)
        elseif isfinite(Lp) || isfinite(Lm)
            L0 = lfix_incremental_at_KBplus!(ws, cache, ctx, pe, w0, coord_idx, w0[coord_idx])
            isfinite(L0) || break
            return isfinite(Lp) ? (Lp - L0) / h_try : (L0 - Lm) / h_try
        end
        h_try /= 4
    end
    return NaN
end

"""
    composite_gradient_at_KBplus(x_free0, ctx, pe, grad_pool::GradWorkspacePool, ws::LFixKBPlusWorkspace;
                                  base=nothing, threaded=false, h_mode=:cached, bandwidth_cache=nothing, multi_method=:top3) -> (g, meta)

"Backend :kbplus": ratio-based factorized base cache (persistent via `ws`) + the SAME
per-coordinate `GradWorkspacePool` production already uses. Structurally identical to
`composite_gradient_at_Cplus` (same threading discipline, same static scheduling), substituting
the ratio-based dest_contrib/bandwidth functions from this file for C+'s log-exp ones.
"""
function composite_gradient_at_KBplus(x_free0::AbstractVector, ctx, pe, pool::GradWorkspacePool, ws::LFixKBPlusWorkspace;
        base::Union{Nothing,BaseDualState} = nothing, threaded::Bool = false,
        h_mode::Symbol = :cached, h0::Float64 = 0.01,
        bandwidth_cache::Union{Nothing,Dict{Int,Float64}} = nothing,
        multi_method::Symbol = :top3)
    h_mode in (:fixed, :cached) || error("composite_gradient_at_KBplus: h_mode must be :fixed|:cached, got $h_mode")
    h_mode == :cached && bandwidth_cache === nothing && error("composite_gradient_at_KBplus: h_mode=:cached requires a bandwidth_cache Dict")

    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache_KB!(ws, x_free0, ctx, base; validate_dense = false)
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
        else
            local h, is_hit
            lock(bandwidth_cache_lock) do
                is_hit = haskey(bandwidth_cache, k)
                h = is_hit ? bandwidth_cache[k] : NaN
            end
            if is_hit
                cache_hits[k] = true
            else
                h, _, _ = select_bandwidth_KB(cache, ctx, pe, w0, k)
                lock(bandwidth_cache_lock) do
                    bandwidth_cache[k] = h
                end
                cache_hits[k] = false
            end
            h_used[k] = h
        end
        g[k] = a_block_fd_component_KBplus!(tws, cache, ctx, pe, w0, k, h_used[k])
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
