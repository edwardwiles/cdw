# ============================================================================
# Experiment (2026-07-16): does warm-starting the inner CC dual solve
# (recover_lfd's KNITRO call) across seq_gravcol's own within-theta k-loop
# help, and does ALSO persisting that warm start across separate outer-loop
# theta trial points help further (vs resetting to cold at the start of every
# new theta)?
#
# Per HANDOFF_2026-07-16_recover_lfd_bug.md's warm-start investigation: today
# recover_lfd is COLD on every single call, at every level (within-theta
# k-iterations AND across theta). This experiment adds warm-starting at two
# levels and compares 3 configurations on ONE point: LC upper bound
# (find_smallest=true), delta=1.0 (T2 budget), theta_init=Astar (theta_r0),
# outer KNITRO loop capped at ~10 major iterations
# (ek_outer_loop_options_cap10.opt, a copy of the production outer .opt file
# with maxit 0->10).
#
#   :cold          -- baseline, matches current production exactly (no warm
#                      start at any level)
#   :reset_per_theta -- warm-start ACROSS the k=1..maxit augmented-recover_lfd
#                      re-solves within one theta (the "obviously worth doing"
#                      case per the user), but reset to cold at the start of
#                      every NEW theta the outer KNITRO loop visits
#   :persist       -- same within-theta warm start, but the dual cache is
#                      NEVER reset across theta either -- a new theta's first
#                      (blind, D+1-moment) recover_lfd call warm-starts from
#                      the PREVIOUS theta's last converged dual. Only ever
#                      cached on convergence (nStatus acceptable) -- a failed
#                      solve never poisons the cache.
#
# Dimension mismatch handled as the user suggested: the blind solve has
# outer_constr_index=D+2 (moments 1..D+1), the augmented solve (once gravity
# is linearized in) has outer_constr_index=D+3 (moments 1..D+2, one extra
# lambda for the new gravity moment). `warmstart_for` below truncates a
# longer cached dual (drop the trailing/newest lambda) or zero-pads a shorter
# one (cold-start only the genuinely-new coordinate), and otherwise reuses
# the rest of the vector as-is.
#
# Simplifications vs full production (kept deliberately, to isolate the dual
# warm-start effect from unrelated machinery):
#   - gradient_method=:pointwise_ad (not :fixed_dual_fd_full) -- the corrected
#     gradient method's finite-difference outer-constraint gradient calls its
#     OWN separate battery of fixed-dual inner solves (build_fixed_dual_bundle
#     et al), which would confound the iteration-count comparison with a
#     second, unrelated source of extra inner solves.
#   - use_var_scaling=false -- irrelevant to warm-starting, drops one probe
#     inner solve per config for a slightly cleaner comparison.
#   - GRAVITY_SEED left at its production default (false / blind D+1 initial
#     solve every theta).
# Neither simplification affects the DUAL warm-start mechanics being tested.
#
# Metric: primarily KNITRO's own reported iteration count (_kn_num_iters,
# already instrumented globally as INNER_ITERS_TOTAL by inner_loop_KNITRO),
# NOT wall-clock alone -- this machine is shared/loaded, and iteration count
# is the apples-to-apples measure of whether warm-starting actually reduces
# the KNITRO solver's own work (see feedback-verify-before-causal-claims:
# don't infer causal timing effects from wall-clock alone).
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

const CAP_OPT_FILE = joinpath(@__DIR__, "ek_outer_loop_options_cap5.opt")
@assert isfile(CAP_OPT_FILE)

# ---- dual-cache warm-start helper: handles the D+2 <-> D+3 dimension mismatch ----
function warmstart_for(dual_cache::Dict{Symbol,Any}, target_oci::Int)
    xa = dual_cache[:aug]
    if xa !== nothing
        length(xa) == target_oci && return copy(xa)
        length(xa) >  target_oci && return xa[1:target_oci]          # truncate: drop newest lambda(s)
    end
    xb = dual_cache[:blind]
    if xb !== nothing
        length(xb) == target_oci && return copy(xb)
        length(xb) <  target_oci && return vcat(xb, zeros(target_oci - length(xb)))  # pad: cold-start only the new coord
    end
    return nothing
end

# ---- warm-startable recover_lfd: identical logic to production's own (see
# run_profiled_production.jl:147, the file the recover_lfd nStatus fix lives
# in), plus an optional x_init warm start and returning the solved x (so the
# caller can cache it) and the KNITRO iteration count actually used. ----
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

# ---- warm-startable seq_gravcol: line-for-line copy of production's own
# (run_profiled_production.jl:192-356) except for the 2 recover_lfd call
# sites (now recover_lfd_ws, threaded through dual_cache/mode) and stats
# instrumentation. Destination-inversion warm-starting (invert_all,
# damping/line-search interpolation) is UNCHANGED -- not what's being tested
# here. ----
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

# ---- warm-startable make_stateful_moments: copy of production's own
# (run_profiled_production.jl:396-453), calling seq_gravcol_ws instead ----
function make_stateful_moments_ws(; find_smallest::Bool, δ::Real, mode::Symbol, stats::WSStats)
    lastθ = Ref(fill(NaN, length(θr0)))
    gcol  = Ref(zeros(W))
    lastRmean = Ref(NaN); lastRcol = Ref(NaN); lastok = Ref(true)
    dRdθ  = Ref(zeros(length(θr0)))
    neval = Ref(0); nfeas = Ref(0)
    warm  = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    warm_p = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_θ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_κ = Ref(find_smallest ? Inf : -Inf)
    best_warm = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    dual_cache = Dict{Symbol,Any}(:blind => nothing, :aug => nothing)
    Ktmp = zeros(1); Gtmp = zeros(1, D + 1)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rmean, Rcol, um, p, ok = seq_gravcol_ws(θf; δ = δ, warm = warm[], warm_p = warm_p[],
                                                            dual_cache = dual_cache, mode = mode, stats = stats,
                                                            tol = STRESS_TOL, maxit = MAXIT_KLOOP)
                lastRmean[] = Rmean; lastRcol[] = Rcol; lastok[] = ok
                if ok
                    gcol[] = c; warm[] = um; warm_p[] = p; nfeas[] += 1
                    dRdθ[] = grad_R_theta(θf, um, p)
                    EK_moments_focal_norm_directgp!(Ktmp, Gtmp, θf, view(U, 1:1, :), (γ = γ,))
                    κθ = Ktmp[1]
                    if (find_smallest && κθ < best_κ[]) || (!find_smallest && κθ > best_κ[])
                        best_κ[] = κθ; best_θ[] = copy(θf); best_warm[] = copy(um)
                    end
                else
                    gcol[] = INFCOL
                    dRdθ[] = zeros(length(θf))
                end
                neval[] += 1
                @printf("    [theta-eval %d, feasible %d] R_mean=%.2e ok=%s seq_time=%.2fs\n",
                        neval[], nfeas[], lastRmean[], ok, time()-t0); flush(stdout)
                lastθ[] = copy(θf)
            end
        end
        EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θ, Uarg, obj)
        nrow = size(G, 1)
        if eltype(θ) <: ForwardDiff.Dual && lastok[]
            Rlin = lastRcol[] + dot(dRdθ[], θ .- lastθ[])
            @inbounds @views @. G[:, D+2] = (gcol[][1:nrow] - lastRcol[]) + Rlin
        else
            @inbounds @views @. G[:, D+2] = gcol[][1:nrow]
        end
    end
    return m!, best_θ, best_κ, neval, nfeas
end

# ---- warm-startable outer_solve_nested_cached: simplified copy of
# production's own (run_profiled_production.jl:500-581) -- pointwise_ad
# gradient, no var scaling (see module docstring for why), capped outer
# maxit=10 via CAP_OPT_FILE ----
function outer_solve_nested_cached_ws(find_smallest, θinit; δ::Real, mode::Symbol, stats::WSStats)
    d = D + 2; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, best_θ, best_κ, neval, nfeas = make_stateful_moments_ws(; find_smallest = find_smallest, δ = δ, mode = mode, stats = stats)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,
        outer_loop_opt = CAP_OPT_FILE, inner_loop_opt = INNER_OPT_FILE)
    l_full = length(θinit)
    free_idx = vcat(3, collect(4:3+D)); fixed_idx = [1, 2]; fixed_vals = θinit[fixed_idx]
    fpmap = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    div_grad_fn! = make_seq_div_grad_fn!(obj, fpmap)
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0); g_free[1] = (-1.0)^find_smallest
    end
    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = CAP_OPT_FILE, var_scales = nothing)
    gp = r.θ_min_full[3]
    (gp = gp, θ_min_full = r.θ_min_full, nStatus = r.nStatus, best_θ = best_θ[], best_κ = best_κ[],
     wall = r.wall, outer_iters = r.outer_iters, outer_fc = r.outer_fc, neval = neval[], nfeas = nfeas[])
end

# ============================================================================
# Run the 3-way comparison, one config at a time (fresh dual_cache/stats each)
# ============================================================================
const DELTA_T2 = 1.0
# STRESS_TOL (tighter than production's default tol=5e-4): the first probe run (tol=5e-4) found
# gravity matched in ONE CC solve at every theta visited near Astar (R_mean already ~1e-4-2e-5
# right after the blind solve) -- so the within-theta k-loop (the scenario this experiment exists
# to test) never actually fired; dual-calls==theta-evals, 0 augmented re-solves. Tightening tol
# forces multiple augmented recover_lfd re-solves per theta regardless of starting point, directly
# exercising the mechanism being compared, without changing any economics (just how many
# refinement rounds are needed to declare convergence).
const STRESS_TOL = 1e-5
# Safety cap on within-theta augmented re-solve rounds (production default is 20) -- keeps any
# one theta-eval bounded even if STRESS_TOL is never reached, so the run stays "quick" as
# requested rather than repeatedly grinding to a 20-round ceiling like the previous (1e-6) attempt.
const MAXIT_KLOOP = 6

results = Dict{Symbol,Any}()
for mode in (:cold, :reset_per_theta, :persist)
    println("\n" * "="^90)
    @printf(">>> MODE = %s  (LC upper bound, delta=%.2f, theta_init=Astar, outer maxit capped=10)\n", mode, DELTA_T2)
    println("="^90); flush(stdout)
    stats = WSStats()
    t0 = time()
    r = outer_solve_nested_cached_ws(true, θr0; δ = DELTA_T2, mode = mode, stats = stats)
    wall_total = time() - t0
    κ = gp2kappa(r.gp)
    @printf("MODE=%s DONE: gp=%.6f kappa=%.6f nStatus=%d outer_iters=%d outer_fc=%d theta_evals=%d feasible_evals=%d\n",
            mode, r.gp, κ, r.nStatus, r.outer_iters, r.outer_fc, r.neval, r.nfeas)
    @printf("  dual-solve stats: n_calls=%d n_warm_calls=%d total_KNITRO_iters=%d total_inner_wall=%.2fs  outer_wall=%.2fs\n",
            stats.n_calls, stats.n_warm_calls, stats.total_iters, stats.total_wall, wall_total)
    for (i, row) in enumerate(stats.log)
        @printf("    call %2d: kind=%-6s ok=%-5s nStatus=%4d n_iters=%3d warm=%-5s wall=%.2fs\n",
                i, row.kind, row.ok, row.nStatus, row.n_iters, row.warm, row.wall)
    end
    results[mode] = (gp = r.gp, kappa = κ, nStatus = r.nStatus, outer_iters = r.outer_iters,
                      theta_evals = r.neval, feasible_evals = r.nfeas,
                      n_calls = stats.n_calls, n_warm_calls = stats.n_warm_calls,
                      total_iters = stats.total_iters, total_inner_wall = stats.total_wall,
                      outer_wall = wall_total, log = stats.log)
    flush(stdout)
end

println("\n" * "="^90); println(">>> DUAL WARM-START COMPARISON SUMMARY"); println("="^90)
@printf("%-16s %8s %8s %10s %10s %14s %12s %10s\n",
        "mode", "kappa", "θ-evals", "dual-calls", "warm-calls", "total-iters", "inner-wall", "outer-wall")
for mode in (:cold, :reset_per_theta, :persist)
    r = results[mode]
    @printf("%-16s %8.5f %8d %10d %10d %14d %12.2f %10.2f\n",
            mode, r.kappa, r.theta_evals, r.n_calls, r.n_warm_calls, r.total_iters, r.total_inner_wall, r.outer_wall)
end
println("\nEXPERIMENT_DUAL_WARMSTART DONE")
