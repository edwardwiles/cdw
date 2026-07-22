# ============================================================================
# Phase 1A: instrumented mirror of oracle.jl::evaluate_fullA. Additive only --
# oracle.jl is NOT modified. Reproduces its logic call-for-call (same
# subroutines: CS.reconstruct_full, CS.inner_loop_internal, obj.moments!,
# gravity_value, compute_winners) so it exercises the EXACT SAME validated
# code paths, wrapped in @prof blocks at each stage boundary. A dedicated
# equivalence test (test_oracle_profiled.jl) checks this produces
# bit-identical results to evaluate_fullA before it is trusted for timing
# conclusions.
#
# GRANULARITY NOTE: `CS.inner_loop_internal` itself is timed as ONE block
# (KNITRO setup + the actual optimize call together) -- KNITRO's own
# per-callback (objective/gradient/Hessian-vector) timing is not separately
# exposed by the installed KNITRO.jl API without patching cc_algo/ itself,
# which this investigation avoided touching. Where per-callback KNITRO detail
# matters, `results/fullA_d4/.../phaseB_hessian_matrix/*_knitro.log` (grepped
# for iteration/eval counts) remains the source for that specific granularity.
# `CS.INNER_SOLVE_COUNT[]` / `CS.INNER_ITERS_TOTAL[]` (pre-existing production
# counters, cc_algo/inner_loop_functions.jl) are read via before/after diffs
# to get an AUTHORITATIVE inner-solve count and total KNITRO iteration count
# per call -- this is what resolves the Phase D "n_inner_solves=34" mislabel
# (see docs/fullA_performance_profile.md).
# ============================================================================
include(joinpath(@__DIR__, "instrumentation.jl"))

"""
    evaluate_fullA_profiled(x_free, ctx; kwargs...) -> (result, prof_meta)

Same signature/semantics as `evaluate_fullA` (oracle.jl), plus a second
return value `prof_meta`: (n_inner_solves, n_inner_infeasible, n_inner_iters)
-- the AUTHORITATIVE counters read directly from `CS.INNER_SOLVE_COUNT[]` /
`CS.INNER_INFEAS_COUNT[]` / `CS.INNER_ITERS_TOTAL[]` via before/after diffs,
not inferred from FD-probe counting (which is what produced the Phase D
"n_inner_solves=34 even for methods that don't re-solve" mislabel -- that
metric was actually counting FUNCTION evaluations, not inner solves; Q_adj/
L_fix never call CS.inner_loop_internal at all, so their TRUE
n_inner_solves is 0, always, by construction).
"""
function evaluate_fullA_profiled(x_free::AbstractVector{Float64}, ctx;
        cache::Union{Nothing,Dict} = nothing, use_cache::Bool = true,
        mode::Symbol = :hard, warm::Bool = true, tag::String = "")

    mode == :hard || error("evaluate_fullA_profiled: mode=:$mode not implemented (matches oracle.jl)")

    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode, context_fingerprint(ctx))

    cache_hit = false
    if cache !== nothing && use_cache
        @prof "cache_lookup" begin
            cache_hit = haskey(cache, key)
        end
        if cache_hit
            hit = cache[key]
            return merge(hit, (cache_hit = true, tag = tag)), (n_inner_solves = 0, n_inner_infeasible = 0, n_inner_iters = 0)
        end
    end

    solves0 = CS.INNER_SOLVE_COUNT[]; infeas0 = CS.INNER_INFEAS_COUNT[]; iters0 = CS.INNER_ITERS_TOTAL[]

    t_total0 = time()
    if !warm
        @prof "warm_start_reset" begin
            obj.x .= NaN
        end
    end

    θ_full = @prof "reconstruct_full" CS.reconstruct_full(x_free, ctx.m)

    t_inner0 = time()
    K_hard, inner_x, nStatus = @prof "inner_solve" CS.inner_loop_internal(obj, θ_full)
    t_inner = time() - t_inner0

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        elapsed = (total = time() - t_total0, inner = t_inner, post = 0.0)
        result = (x_free = collect(x_free), θ_full = θ_full,
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, ctx.D),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, benchmark_unweighted_moment_mean = Float64[], max_abs_moment_resid = NaN,
                  zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
                  weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
                  winner_hash = UInt64(0), inner_status = nStatus, inner_iters = inner_iters,
                  primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
                  elapsed = elapsed, error_reason = "inner solve failed: nStatus=$nStatus")
        cache !== nothing && @prof("cache_materialize", cache[key] = result)
        prof_meta = (n_inner_solves = CS.INNER_SOLVE_COUNT[] - solves0,
                     n_inner_infeasible = CS.INNER_INFEAS_COUNT[] - infeas0,
                     n_inner_iters = CS.INNER_ITERS_TOTAL[] - iters0)
        return result, prof_meta
    end

    W = size(obj.U, 1); d = obj.d
    K = zeros(W); G = zeros(W, d)
    @prof "moments_recompute" obj.moments!(K, G, θ_full, obj.U, obj)

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = @prof "outer_constraint_callback" obj(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = @prof "primal_divergence_compute" primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    ζstar = inner_x[1]; λstar = inner_x[2:end]
    moment_kkt = @prof "kkt_residual_compute" [abs(sum(m_weights .* G[:, j]) / W) for j in 1:min(length(λstar), size(G,2))]
    max_abs_moment_kkt_resid = isempty(moment_kkt) ? NaN : maximum(moment_kkt)

    gravity_raw = obj.outer_constr_index <= d ? cbuf[2] : NaN
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D^2], ctx.D, ctx.D)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (ctx.D, ctx.D))'
    gravity_val, logA, R_sum, R_mean, R_beta = @prof "gravity_compute" begin
        Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
        AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
        gv = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)
        lA = -log.(AodPow)
        rs = sum(ctx.q_tilde .* lA)
        (gv, lA, rs, rs / ctx.D^2, rs / sum(ctx.q_tilde .^ 2))
    end

    benchmark_unweighted_moment_mean = vec(sum(G, dims=1)) ./ W
    max_abs_moment_resid = isempty(benchmark_unweighted_moment_mean) ? NaN : maximum(abs.(benchmark_unweighted_moment_mean))

    winner, price_, gap_ = @prof "winner_compute" compute_winners(θ_full, ctx)
    winner_hash = hash(winner)

    t_total = time() - t_total0
    elapsed = (total = t_total, inner = t_inner, post = t_total - t_inner)

    result = (x_free = collect(x_free), θ_full = θ_full,
              gamma_focal_prime = θ_full[3+ctx.D], logA = logA,
              K_hard = K_hard, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              Delta_minus_delta = Delta_dual - obj.δ,
              gravity_raw = gravity_raw, gravity_value = gravity_val,
              gravity_R_sum = R_sum, gravity_R_mean = R_mean, gravity_R_beta = R_beta,
              benchmark_unweighted_moment_mean = benchmark_unweighted_moment_mean, max_abs_moment_resid = max_abs_moment_resid,
              zeta = ζstar, lambda = collect(λstar),
              m_mean = sum(m_weights)/W, m_min = minimum(m_weights), m_max = maximum(m_weights),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              winner_hash = winner_hash, inner_status = nStatus, inner_iters = inner_iters,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              cache_hit = false, warm_started = warm, tag = tag,
              elapsed = elapsed, error_reason = nothing)

    cache !== nothing && @prof("cache_materialize", cache[key] = result)
    prof_meta = (n_inner_solves = CS.INNER_SOLVE_COUNT[] - solves0,
                 n_inner_infeasible = CS.INNER_INFEAS_COUNT[] - infeas0,
                 n_inner_iters = CS.INNER_ITERS_TOTAL[] - iters0)
    return result, prof_meta
end
