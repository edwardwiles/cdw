# ============================================================================
# Experiment v2 (2026-07-16): SAME question as experiment_dual_warmstart.jl
# (does warm-starting recover_lfd within seq_gravcol's own k-loop help, and
# does persisting that warm start ACROSS theta points help further) but with
# a controlled, predictable design instead of letting an outer KNITRO loop
# wander unpredictably in search of new theta points (the previous attempt
# spent 14 minutes on 11 theta-evals, only 2 of them feasible, because the
# cheap :pointwise_ad outer gradient explored genuinely hard territory).
#
# Here: 3 FIXED theta points (Astar, rand1, rand2 -- the SAME 3 A_od starts
# already used and validated by the official head-to-head LC/GC comparison,
# loaded from shared_starts.jld2, not invented), gamma'_focal held at theta_r0's
# own value throughout (only A_od varies), delta=1.0 (T2), tol=1e-5 (tight
# enough to force >=1 augmented re-solve per point based on the very first,
# tol=5e-4 probe -- see experiment_dual_warmstart.jl's own header). Calls
# seq_gravcol_ws directly, in the SAME order for all 3 modes, no outer loop.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra, JLD2

# ---- dual-cache warm-start helper: handles the D+2 <-> D+3 dimension mismatch ----
function warmstart_for(dual_cache::Dict{Symbol,Any}, target_oci::Int)
    xa = dual_cache[:aug]
    if xa !== nothing
        length(xa) == target_oci && return copy(xa)
        length(xa) >  target_oci && return xa[1:target_oci]
    end
    xb = dual_cache[:blind]
    if xb !== nothing
        length(xb) == target_oci && return copy(xb)
        length(xb) <  target_oci && return vcat(xb, zeros(target_oci - length(xb)))
    end
    return nothing
end

function recover_lfd_ws(θ, moments_fn, d; x_init::Union{Nothing,Vector{Float64}} = nothing)
    oci = d + 1
    use_warm = x_init !== nothing
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = JacW, lower_limit = -5000, use_cached_x = use_warm,
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    use_warm && (obj.x .= x_init)
    iters0 = CS.INNER_ITERS_TOTAL[]
    val, x, nStatus = inner_loop(obj, θ)
    n_iters = CS.INNER_ITERS_TOTAL[] - iters0
    ok_status = nStatus ∈ (0, -100, -101, -103)
    if !ok_status || !all(isfinite, x)
        return fill(1.0 / W, W), false, nothing, nStatus, n_iters, use_warm
    end
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    if !(isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD))
        return fill(1.0 / W, W), false, nothing, nStatus, n_iters, use_warm
    end
    return LFD ./ s, true, x, nStatus, n_iters, use_warm
end

mutable struct WSStats
    n_calls::Int
    n_warm_calls::Int
    total_iters::Int
    total_wall::Float64
    log::Vector{NamedTuple}
end
WSStats() = WSStats(0, 0, 0, 0.0, NamedTuple[])

function seq_gravcol_ws(θ; δ::Real, maxit = 20, tol = 5e-4, warm = nothing,
        warm_p::Union{Nothing,AbstractVector} = nothing, verbose = false,
        dual_cache::Dict{Symbol,Any}, mode::Symbol, stats::WSStats)
    mode === :reset_per_theta && (dual_cache[:blind] = nothing; dual_cache[:aug] = nothing)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    (all(isfinite, uf) && all(isfinite, log_x)) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    function invert_all(p_arg, u_init_fn::Function)
        um = zeros(D, D); um[:, focal] .= uf
        stats_ = Vector{Any}(undef, D)
        all_converged = Threads.Atomic{Bool}(true)
        if PARALLEL_INVERSION
            Threads.@threads for i in eachindex(omitted)
                d = omitted[i]
                inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = DEST_INV_TOL,
                                         maxit = 150, ls_iters = 50, u_init = u_init_fn(d))
                um[:, d] .= inv.u_full
                stats_[d] = inv.stats
                inv.converged || (all_converged[] = false)
            end
        else
            for d in omitted
                inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = DEST_INV_TOL,
                                         maxit = 150, ls_iters = 50, u_init = u_init_fn(d))
                um[:, d] .= inv.u_full
                stats_[d] = inv.stats
                inv.converged || (all_converged[] = false)
            end
        end
        um, stats_, all_converged[]
    end
    invert_omitted(p_arg; warm = nothing) = invert_all(p_arg, d -> warm === nothing ? nothing : warm[:, d])

    x_init_blind = mode === :cold ? nothing : warmstart_for(dual_cache, D + 2)
    t0 = time()
    p, ok, x_sol, nSt, n_iters, was_warm = recover_lfd_ws(θ, EK_moments_focal_norm_directgp!, D + 1; x_init = x_init_blind)
    wall = time() - t0
    stats.n_calls += 1; stats.total_iters += n_iters; stats.total_wall += wall; was_warm && (stats.n_warm_calls += 1)
    push!(stats.log, (kind = :blind, ok = ok, nStatus = nSt, n_iters = n_iters, warm = was_warm, wall = wall))
    ok && mode !== :cold && (dual_cache[:blind] = x_sol)
    if !ok
        verbose && println("    [seq] initial recover_lfd (focal-only) FAILED")
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    local umat, R, stats_cur
    try
        umat, stats_cur, all_ok_init = invert_omitted(p; warm = warm)
        if !all_ok_init
            verbose && println("    [seq] initial invert_omitted: at least one destination did NOT converge -- treating as infeasible")
            return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
        end
        R = gravity_residual(umat, logτ, logw, σ).R_mean
    catch e
        verbose && println("    [seq] initial invert_omitted threw: ", e)
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    isfinite(R) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    verbose && @printf("    [seq] init: R0=%.4e\n", R)
    col = zeros(W); Rcol = 0.0
    for k in 1:maxit
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ,
                                  scale = :R_beta, precomputed_stats = stats_cur)
        col = infl.ψ_bar .+ infl.R_beta
        Rcol = infl.R_beta
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
        end
        x_init_aug = mode === :cold ? nothing : warmstart_for(dual_cache, D + 3)
        t0 = time()
        p_cand, okc, x_sol_aug, nStc, n_iters_aug, was_warm_aug = recover_lfd_ws(θ, moments_aug!, D + 2; x_init = x_init_aug)
        wall = time() - t0
        stats.n_calls += 1; stats.total_iters += n_iters_aug; stats.total_wall += wall; was_warm_aug && (stats.n_warm_calls += 1)
        push!(stats.log, (kind = :aug, ok = okc, nStatus = nStc, n_iters = n_iters_aug, warm = was_warm_aug, wall = wall))
        okc && mode !== :cold && (dual_cache[:aug] = x_sol_aug)
        if !okc
            verbose && println("    [seq] iter $k: augmented recover_lfd FAILED (linearized moment likely unmatchable)")
            break
        end
        α = 1.0; acc = false
        um_endpoint1 = nothing; stats_endpoint1 = nothing
        for _ in 1:12
            p_try = (1 - α) .* p .+ α .* p_cand
            local um_try, R_try, stats_try, all_ok_try
            try
                if α == 1.0 || um_endpoint1 === nothing
                    um_try, stats_try, all_ok_try = invert_all(p_try, d -> umat[:, d])
                    if um_endpoint1 === nothing
                        um_endpoint1 = um_try; stats_endpoint1 = stats_try
                    end
                else
                    um_try, stats_try, all_ok_try = invert_all(p_try, d -> (1 - α) .* umat[:, d] .+ α .* um_endpoint1[:, d])
                end
                R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
            catch
                α *= 0.5; continue
            end
            if all_ok_try && isfinite(R_try) && abs(R_try) < abs(R)
                p = p_try; umat = um_try; R = R_try; stats_cur = stats_try; acc = true; break
            end
            α *= 0.5
        end
        verbose && @printf("    [seq] iter %d: R_mean -> %.4e  accepted=%s  α=%.4f\n", k, R, acc, acc ? α : 0.0)
        acc || break
    end
    div_p = divergence_of(p)
    gravity_ok = abs(R) <= tol
    δ_ok = div_p <= δ * (1 + 1e-6) + 1e-10
    return col, R, Rcol, umat, p, gravity_ok && δ_ok
end

# ============================================================================
# 3 fixed points: Astar, rand1, rand2 -- SAME A_od starts as the official
# head-to-head comparison (shared_starts.jld2), gamma'_focal held at theta_r0's
# own value (only A_od varies). No outer loop.
# ============================================================================
shared = JLD2.load(joinpath(@__DIR__, "shared_starts.jld2"))
mkθ(Aod) = vcat(θr0[1:3], Float64.(Aod))
POINTS = [("Astar", mkθ(shared["Acol_star"])), ("rand1", mkθ(shared["rand1"])), ("rand2", mkθ(shared["rand2"]))]
const DELTA_T2 = 1.0
const STRESS_TOL = 1e-5

results = Dict{Symbol,Any}()
for mode in (:cold, :reset_per_theta, :persist)
    println("\n" * "="^90)
    @printf(">>> MODE = %s  (3 fixed points: Astar/rand1/rand2, delta=%.2f, tol=%.1e, no outer loop)\n", mode, DELTA_T2, STRESS_TOL)
    println("="^90); flush(stdout)
    stats = WSStats()
    dual_cache = Dict{Symbol,Any}(:blind => nothing, :aug => nothing)
    for (name, θ) in POINTS
        t0 = time()
        col, R, Rcol, umat, p, ok = seq_gravcol_ws(θ; δ = DELTA_T2, tol = STRESS_TOL,
            dual_cache = dual_cache, mode = mode, stats = stats)
        wall = time() - t0
        @printf("  [%s] R_mean=%.3e ok=%s wall=%.2fs\n", name, R, ok, wall); flush(stdout)
    end
    @printf("MODE=%s DONE: n_calls=%d n_warm_calls=%d total_KNITRO_iters=%d total_inner_wall=%.2fs\n",
            mode, stats.n_calls, stats.n_warm_calls, stats.total_iters, stats.total_wall)
    for (i, row) in enumerate(stats.log)
        @printf("    call %2d: kind=%-6s ok=%-5s nStatus=%4d n_iters=%3d warm=%-5s wall=%.2fs\n",
                i, row.kind, row.ok, row.nStatus, row.n_iters, row.warm, row.wall)
    end
    results[mode] = (n_calls = stats.n_calls, n_warm_calls = stats.n_warm_calls,
                      total_iters = stats.total_iters, total_inner_wall = stats.total_wall)
    flush(stdout)
end

println("\n" * "="^90); println(">>> DUAL WARM-START COMPARISON SUMMARY (direct, fixed points)"); println("="^90)
@printf("%-16s %10s %10s %14s %12s\n", "mode", "dual-calls", "warm-calls", "total-iters", "inner-wall")
for mode in (:cold, :reset_per_theta, :persist)
    r = results[mode]
    @printf("%-16s %10d %10d %14d %12.2f\n", mode, r.n_calls, r.n_warm_calls, r.total_iters, r.total_inner_wall)
end
println("\nEXPERIMENT_DUAL_WARMSTART_DIRECT DONE")
