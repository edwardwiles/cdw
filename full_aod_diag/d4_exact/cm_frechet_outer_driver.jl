# ============================================================================
# Fixed-Fréchet-marginals outer KNITRO driver (2026-07-23), for the task
# brief §10.3 shakedown: direct joint constrained search, C+ gradient
# backend. Structural twin of `run_cm_upper` (cm_outer_driver.jl), swapping
# in the fixed-Fréchet production context/gradient (`build_cm_frechet_production_context`,
# `cm_frechet_production_gradient_cplus`) and `archC_frechet_verified_state`'s
# typed verified-success gate. Purely additive: does not modify
# cm_outer_driver.jl. Same objective/constraint convention:
#     minimize    w[1]  (gamma'_focal, find_smallest=true -> UPPER bound on kappa)
#     subject to  Delta_dual(w) <= delta
# ============================================================================
using KNITRO

"CM-aware `cm_production_value_verified` analog for fixed Fréchet marginals."
function cm_frechet_production_value_verified(x_free0::AbstractVector, fpcx)
    base, verify = archC_frechet_verified_state(x_free0, fpcx.ctx_cm, fpcx.fctx)
    K = fpcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    run_frechet_upper_cplus(fpcx, ctx, pe, w0; delta=1.0, maxtime_real=180.0,
        opt_file="csw_outer_wallclock_sr1.opt", z_halfwidth=30.0,
        gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi) -> NamedTuple

One constrained-upper-bound outer KNITRO solve against the fixed-Fréchet
production bundle `fpcx` (`build_cm_frechet_production_context`), C+
gradient backend (`cm_frechet_production_gradient_cplus`). Same
best-exact-feasible-incumbent tracking and typed verified-success gate
`run_cm_upper` establishes, reused via the same `is_verified_success`/
`reject_point` (oracle.jl, unchanged).
"""
function run_frechet_upper_cplus(fpcx, ctx, pe, w0::Vector{Float64};
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        verbose::Bool = true)
    D2 = length(w0)
    W = size(ctx.U, 1); D = ctx.D
    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, W)
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
    bandwidth_cache = Dict{Int,Float64}()
    t_start = time()

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        local base, verify
        try
            _, base, verify = cm_frechet_production_value_verified(xf, fpcx)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_frechet_upper_cplus: infeasible/failed inner solve at this point")
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
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = time() - t_start)
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if verbose && (n_eval[] <= 3 || n_eval[] % 10 == 0)
            println("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible, " verified=", verified)
            flush(stdout)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        gfull, meta = cm_frechet_production_gradient_cplus(xf, fpcx, ctx, pe, pool, ws; base = base, threaded = true,
            h_mode = :cached, bandwidth_cache = bandwidth_cache)
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
            best = b, kappa = κ, xsol = collect(xsol), trace = trace)
end
