# ============================================================================
# Phase 1 (continuation 3): fast + correctly-decomposed additive mirror of
# oracle.jl::evaluate_fullA / oracle_profiled.jl::evaluate_fullA_profiled.
#
# Implements the three Phase 1 fixes:
#   1A. Reuses obj.H's own moments!-output columns (already computed once
#       inside inner_loop_internal) instead of oracle.jl's literal SECOND
#       obj.moments!(...) call in post-processing (Finding #1).
#   1B. Replaces winners.jl's per-column sort() with winners_v2.jl's
#       allocation-free two-pass min/second-min scan (Finding #2).
#   1C. Replaces the KKT-residual list-comprehension's repeated column
#       slicing with a single preallocated reduction loop.
#
# CORRECTED PROFILING GRANULARITY (per explicit user request after reviewing
# the Phase 1 (continuation 2) profile): the ORIGINAL "inner_solve" timer
# wrapped the ENTIRE inner_loop_internal call, which is NOT "the CC
# multiplier optimization" -- it is [ONE moments! build, O(D^2 W), building
# the FIXED G matrix at this theta] followed by [KNITRO's own inner iteration
# loop over the dual variables (zeta,lambda), which is CHEAP because it reuses
# the fixed H/G matrix via BLAS.gemv! and never rebuilds moments -- verified
# directly from cc_algo/PsiObjectiveBundle.jl:172-251's callable method: the
# `length(theta)==0` branch (exactly how callbackEvalFG_inner! calls it) only
# touches H via BLAS ops, no moments! call]. This file adds NESTED timers so
# the moment-build cost and the actual dual-optimization cost are no longer
# conflated:
#   inner_moment_build        -- the ONE obj.moments! call inner_loop_internal
#                                 makes (builds obj.H once per theta)
#   inner_knitro_dual_solve   -- inclusive wall time of inner_loop_KNITRO's
#                                 own KN_solve call (dual optimization proper)
#     inner_dual_fg_callback  -- (nested under the above) sum of time inside
#                                 callbackEvalFG_inner! -- FUSED objective+
#                                 gradient w.r.t. (zeta,lambda) in ONE KNITRO
#                                 callback (see cc_algo/inner_loop_functions.jl
#                                 :26-34 -- `obj(x, evalResult.objGrad)` is a
#                                 single call producing both; genuinely NOT
#                                 separable into "objective-only" vs
#                                 "gradient-only" time without patching
#                                 PsiObjectiveBundle.jl's own callable method,
#                                 which this investigation's additive-only
#                                 discipline avoids -- documented as a
#                                 granularity limit, not silently merged)
#     inner_dual_hessian_callback -- (nested) sum of time inside
#                                 callbackEvalH_inner! (registered because
#                                 ek_inner.opt sets hessopt=exact=1)
#   primal_weight_recovery     -- the obj(inner_x, constr=...) call that
#                                 populates obj.arg1 = m(s) (dPsi! re-evaluated
#                                 at the converged (zeta*,lambda*))
#   kkt_residual_compute       -- (1C fix: preallocated reduction, no column
#                                 slicing list comprehension)
#   gravity_compute, winner_compute (1B fix) -- as before
#
# `inner_knitro_dual_solve`'s EXCLUSIVE cost (its own inclusive time minus the
# nested fg/hessian callback totals) is KNITRO's own SQP/interior-point
# overhead (line search, subproblem factorization, convergence checks) --
# reported directly in the profile output, not left implicit.
#
# Equivalence-tested (test_oracle_fast.jl) against oracle.jl::evaluate_fullA
# before being trusted for either correctness or timing.
# ============================================================================
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
using KNITRO
using LinearAlgebra: mul!

# ============================================================================
# Continuation 10, Section 9 (finalize-architecture, Part A #2): BLAS-gemv
# replacements for the hand-rolled KKT-residual / moment-residual reductions.
# Validated (docs/fullA_D20_blas_audit_report.md, Section 4 of this
# continuation's BLAS audit) at ~2.1-2.2x over the nested-loop form on a
# synthetic W=80,000 x d=402 matrix; used identically by oracle_fast.jl's own
# dense tail, compressed_live.jl's compressed tail, and
# infeasibility_screen.jl's screened-compressed tail (the SAME pattern was
# independently present, hand-rolled, in all three). `transpose(view(...))`
# is a lazy wrapper -- `transpose(G)*v` dispatches to BLAS gemv (no copy) for
# dense Float64 arrays, which is exactly the swap the audit benchmarked.
# ============================================================================

"BLAS-gemv max-abs KKT residual: max_j |sum_ω m_weights[ω]*G[ω,j]| / W for j in 1:nkkt. Equivalent to the hand-rolled nested-loop reduction it replaces (Continuation 10 BLAS audit, ~2.1-2.2x)."
function kkt_residual_blas(G::AbstractMatrix, m_weights::AbstractVector, nkkt::Int, W::Int)
    nkkt == 0 && return 0.0
    s = Vector{Float64}(undef, nkkt)
    mul!(s, transpose(@view(G[:, 1:nkkt])), m_weights)
    return maximum(abs, s) / W
end

"BLAS-gemv moment residual: column-mean of G[:,1:d] (Continuation 10 BLAS audit, ~2.1x). Uses gemv against a vector of ones rather than sum(dims=1) to avoid a full-size temporary."
function moment_resid_blas(G::AbstractMatrix, d::Int, W::Int)
    d == 0 && return Float64[]
    mr = Vector{Float64}(undef, d)
    mul!(mr, transpose(@view(G[:, 1:d])), ones(W))
    mr ./= W
    return mr
end

# ---- per-inner-solve callback counters (reset at the start of each inner_loop_KNITRO_profiled call) ----
mutable struct InnerCallCounters
    n_fg_calls::Int
    n_hess_calls::Int
end
const _INNER_CALL_COUNTERS = Ref(InnerCallCounters(0, 0))

function _callbackEvalFG_inner_profiled!(kc, cb, evalRequest, evalResult, userParams)
    obj = userParams
    x = evalRequest.x
    @prof "inner_dual_fg_callback" begin
        evalResult.obj[1] = obj(x, evalResult.objGrad)
    end
    _INNER_CALL_COUNTERS[].n_fg_calls += 1
    return 0
end

function _callbackEvalH_inner_profiled!(kc, cb, evalRequest, evalResult, userParams)
    obj = userParams
    x = evalRequest.x
    @prof "inner_dual_hessian_callback" begin
        obj(x, h = evalResult.hess)
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

"""
    inner_loop_KNITRO_profiled(obj) -> (nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)

Faithful additive mirror of `cc_algo/inner_loop_functions.jl::inner_loop_KNITRO`
(same variable/bound/init-value setup, same option file, same Hessian/
complementarity-constraint wiring), registering PROFILED callback wrappers
instead of the production ones. `KN_solve` itself is timed as
`inner_knitro_dual_solve` (inclusive of the nested fg/hessian callback time).
"""
function inner_loop_KNITRO_profiled(obj)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)

    # Ported from diag/fullA-inner-blas-threading (parallelism_guards.jl). This is the ACTUAL
    # production inner solve (evaluate_fullA_screened_ranged's compressed/fast path calls into
    # this, not cc_algo/inner_loop_functions.jl::inner_loop_KNITRO directly), so it gets the
    # same guard as that one.
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_profiled!)
        KNITRO.KN_set_cb_user_params(kc, cb, obj)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_profiled!)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        @prof "inner_knitro_dual_solve" begin
            KNITRO.KN_solve(kc)
        end
        nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        KNITRO.KN_free(kc)

        return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
    finally
        CS.guard_exit_inner_solve!()
    end
end

"""
    inner_loop_internal_profiled(obj::PsiObjectiveBundleImplicit, θ) -> (K_hard, x, nStatus, n_fg_calls, n_hess_calls)

Faithful additive mirror of `cc_algo/inner_loop_functions.jl::inner_loop_internal`
for `PsiObjectiveBundleImplicit` (the only concrete type this investigation
uses), splitting the ONE `obj.moments!` call (`inner_moment_build`) from the
actual dual optimization (`inner_knitro_dual_solve`, timed inside
`inner_loop_KNITRO_profiled`).
"""
function inner_loop_internal_profiled(obj, θ)
    @prof "inner_moment_build" begin
        obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    end
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_profiled(obj)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess
    end
end

"""
    evaluate_fullA_fast(x_free, ctx; kwargs...) -> (result, prof_meta)

Same signature/semantics as `evaluate_fullA` (oracle.jl), with the Phase 1A/B/C
fixes applied and nested profiling. `prof_meta` additionally reports
`n_fg_calls`/`n_hess_calls` (the TRUE inner-KNITRO callback counts for this
one evaluation, not a cumulative counter diff).
"""
function evaluate_fullA_fast(x_free::AbstractVector{Float64}, ctx;
        cache = nothing, use_cache::Bool = true,
        mode::Symbol = :hard, warm::Bool = true, tag::String = "",
        moment_representation::Symbol = :dense,
        # Final-architecture-closure task (2026-07-27), Goal 9: passthrough to
        # evaluate_fullA_fast_compressed's own dense_reference_diagnostics kwarg (see that
        # function's docstring) -- only meaningful when moment_representation=:compressed; ignored
        # (never read) on the :dense path below, which always computes these fields in full.
        dense_reference_diagnostics::Bool = false)

    # ADDITIVE (continuation 8, workstream 2): opt-in compressed winner-form
    # inner-dual evaluation, see compressed_live.jl. Default :dense preserves
    # this function's ENTIRE body below unchanged for every existing caller
    # that does not pass this kwarg -- dense stays the trusted reference.
    # :compressed dispatches to a SEPARATE function (evaluate_fullA_fast_compressed,
    # compressed_live.jl) that itself falls back to calling THIS function with
    # moment_representation=:dense on any TiedWinnerError -- never silently.
    if moment_representation === :compressed
        return evaluate_fullA_fast_compressed(x_free, ctx; cache = cache, use_cache = use_cache,
                                               mode = mode, warm = warm, tag = tag,
                                               dense_reference_diagnostics = dense_reference_diagnostics)
    elseif moment_representation !== :dense
        error("evaluate_fullA_fast: moment_representation=:$moment_representation not implemented (only :dense, :compressed)")
    end

    mode == :hard || error("evaluate_fullA_fast: mode=:$mode not implemented (matches oracle.jl)")

    # Part A (2026-07-23): see oracle.jl::evaluate_fullA for the same fallback rationale.
    D_dest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode, context_fingerprint(ctx))

    cache_hit = false
    hit = nothing
    if cache !== nothing && use_cache
        @prof "cache_lookup" begin
            hit = _cache_lookup(cache, key)
        end
        cache_hit = hit !== nothing
        if cache_hit
            return merge(hit, (cache_hit = true, tag = tag)), (n_inner_solves = 0, n_inner_infeasible = 0, n_inner_iters = 0, n_fg_calls = 0, n_hess_calls = 0)
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
    K_hard, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_profiled(obj, θ_full)
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
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, D_dest),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, benchmark_unweighted_moment_mean = Float64[], max_abs_moment_resid = NaN,
                  zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
                  weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
                  winner_hash = UInt64(0), inner_status = nStatus, inner_iters = inner_iters,
                  primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
                  elapsed = elapsed, error_reason = "inner solve failed: nStatus=$nStatus")
        cache !== nothing && is_cacheable_result(result) && @prof("cache_materialize", _cache_store!(cache, key, result))
        prof_meta = (n_inner_solves = CS.INNER_SOLVE_COUNT[] - solves0,
                     n_inner_infeasible = CS.INNER_INFEAS_COUNT[] - infeas0,
                     n_inner_iters = CS.INNER_ITERS_TOTAL[] - iters0, n_fg_calls = n_fg, n_hess_calls = n_hess)
        return result, prof_meta
    end

    # ---- Phase 1A fix: REUSE obj.H's own moments!-output (already built inside
    #      inner_loop_internal_profiled above, at this SAME theta_full -- byte-
    #      identical by construction, not merely "should agree") instead of a
    #      second obj.moments! call. Copy into fresh buffers (not a live view)
    #      so nothing downstream can be corrupted by a LATER call reusing obj.H. ----
    W = size(obj.U, 1); d = obj.d
    K, G = @prof "moments_reuse" begin
        (copy(@view(obj.H[:, 1])), copy(CS.select_G_from_H(obj, obj.H)))
    end

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = @prof "primal_weight_recovery" begin
        obj(inner_x, constr = @view(cbuf[1:ncon]))
    end
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = @prof "primal_divergence_compute" primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    ζstar = inner_x[1]; λstar = inner_x[2:end]
    # Phase 1C fix: preallocated reduction instead of a column-slicing list comprehension.
    nkkt = min(length(λstar), size(G, 2))
    # Continuation 10 Section 9: BLAS-gemv swap (kkt_residual_blas, oracle_fast.jl), was a
    # hand-rolled nested loop -- see docs/fullA_D20_blas_audit_report.md, ~2.1-2.2x.
    max_abs_moment_kkt_resid = @prof "kkt_residual_compute" kkt_residual_blas(G, m_weights, nkkt, W)

    gravity_raw = obj.outer_constr_index <= d ? cbuf[2] : NaN
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*D_dest], ctx.D, D_dest)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (D_dest, ctx.D))'
    gravity_val, logA, R_sum, R_mean, R_beta = @prof "gravity_compute" begin
        Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
        AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
        gv = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs; exclude_diagonal=get(ctx, :exclude_diagonal_gravity, false), exclude_cells=get(ctx, :gravity_exclude_cells, Tuple{Int,Int}[]))
        lA = -log.(AodPow)
        rs = sum(ctx.q_tilde .* lA)
        (gv, lA, rs, rs / (ctx.D * D_dest), rs / sum(ctx.q_tilde .^ 2))
    end

    # Phase 1C fix: preallocated reduction instead of sum(G,dims=1) (a full temp-array allocation).
    # Continuation 10 Section 9: BLAS-gemv swap (moment_resid_blas, oracle_fast.jl) --
    # see docs/fullA_D20_blas_audit_report.md, ~2.1x.
    benchmark_unweighted_moment_mean = @prof "moment_resid_compute" moment_resid_blas(G, d, W)
    max_abs_moment_resid = isempty(benchmark_unweighted_moment_mean) ? NaN : maximum(abs.(benchmark_unweighted_moment_mean))

    # Phase 1B fix: allocation-free two-pass min/second-min instead of per-column sort().
    winner, price_, gap_ = @prof "winner_compute" compute_winners_fast(θ_full, ctx)
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

    cache !== nothing && is_cacheable_result(result) && @prof("cache_materialize", _cache_store!(cache, key, result))
    prof_meta = (n_inner_solves = CS.INNER_SOLVE_COUNT[] - solves0,
                 n_inner_infeasible = CS.INNER_INFEAS_COUNT[] - infeas0,
                 n_inner_iters = CS.INNER_ITERS_TOTAL[] - iters0, n_fg_calls = n_fg, n_hess_calls = n_hess)
    return result, prof_meta
end
