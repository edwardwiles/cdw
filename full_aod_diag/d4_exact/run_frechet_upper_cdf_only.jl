# CDF-only analogue of run_frechet_upper.jl (which is :cdf_power-only, unchanged). Structural twin:
# same objective/constraint/search-space/state-reuse discipline (task brief §8), swapping in the
# threaded/syrk CDF-only value function (cm_frechet_verified_state_threaded,
# cm_frechet_production_bundle_threaded.jl) and the CDF-only lookup-based outer gradient
# (cm_frechet_production_gradient_cdf_only, cm_frechet_cdf_only_gradient.jl) instead of the
# :cdf_power-only cm_frechet_verified_state / cm_frechet_production_gradient.
# Task: FIXED_FRECHET_INNER_SOLVER production-feasibility 2026-07-24 (CDF-only addendum).
using KNITRO

x_free_from_w_frechet(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

"""
    run_frechet_upper_cdf_only(fpcx, ctx, pe, w0, tls::ThreadLocalBinScratch; delta=1.0,
                                maxtime_real=1800.0, opt_file="csw_outer_wallclock_sr1.opt",
                                z_halfwidth=30.0, gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi,
                                verbose=true) -> NamedTuple

CDF-only (`fpcx.cfg.frechet_feature_set === :cdf_only`) constrained-upper-bound outer KNITRO solve,
using the threaded/syrk structured Hessian for every inner solve. `tls` from
`build_thread_local_scratch(fpcx.fctx.cctx)`, built once by the caller (reused across every inner
solve in this outer run -- the whole point of avoiding a rebuild per callback).
"""
function run_frechet_upper_cdf_only(fpcx, ctx, pe, w0::Vector{Float64}, tls::ThreadLocalBinScratch;
        delta::Float64 = 1.0, maxtime_real::Float64 = 1800.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        verbose::Bool = true)
    fpcx.cfg.frechet_feature_set === :cdf_only || error("run_frechet_upper_cdf_only: fpcx must be :cdf_only")
    D2 = length(w0)
    w_lo = vcat(gp_lo, w0[2:end] .- z_halfwidth)
    w_hi = vcat(gp_hi, w0[2:end] .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(nothing)
    n_eval = Ref(0); n_grad = Ref(0); trace = NamedTuple[]
    n_base_reused_at_gradient = Ref(0); n_new_point_solve = Ref(0)
    n_time_limit_no_certificate = Ref(0); n_infeasible_certificate = Ref(0)
    bandwidth_cache = Dict{Int,Float64}()
    t_start = time()

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w_frechet(w, pe)
        local base, verify
        try
            base, verify = cm_frechet_verified_state_threaded(xf, fpcx, tls)
            n_new_point_solve[] += 1
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            msg = sprint(showerror, e)
            if occursin("time_limit_no_certificate", msg) || occursin("limit_infeasible", msg)
                n_time_limit_no_certificate[] += 1
            else
                n_infeasible_certificate[] += 1
            end
            evalResult.obj[1] = w[1]; evalResult.c[1] = 1.0e10
            n_eval[] += 1
            push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = NaN, feasible = false, verified = false, reason = msg[1:min(120,end)]))
            return 0
        end
        Δ = verify.Delta_dual
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        verified = is_verified_success(verify)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = time() - t_start, base = base)
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible, verified = verified, reason = ""))
        if verbose && (n_eval[] <= 3 || n_eval[] % 5 == 0)
            println("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible, " verified=", verified)
            flush(stdout)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w_frechet(w, pe)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        base !== nothing && (n_base_reused_at_gradient[] += 1)
        gfull, meta = cm_frechet_production_gradient_cdf_only(xf, fpcx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= gfull
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))
    return (knitro_status = nStatus, wall = wall, n_eval = n_eval[], n_grad = n_grad[],
            n_new_point_solve = n_new_point_solve[], n_base_reused_at_gradient = n_base_reused_at_gradient[],
            n_time_limit_no_certificate = n_time_limit_no_certificate[],
            n_infeasible_certificate = n_infeasible_certificate[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace)
end
