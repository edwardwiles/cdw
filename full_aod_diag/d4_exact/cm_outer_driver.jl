# ============================================================================
# Continuation 13, Section 5 (outer wiring) + Section 7/9 driver.
#
# Constrained-upper-bound KNITRO outer loop for the CM production bundle:
#     minimize    w[1]              (gamma'_focal -- find_smallest=true convention
#                                     maximizes kappa, i.e. an UPPER bound; matches
#                                     c10_d20_production_driver.jl's own
#                                     run_polish_checkpointed sign convention exactly,
#                                     re-derived and confirmed here, not assumed)
#     subject to  Delta_dual(w) <= delta
# over w = [gamma'_focal; zfree] (pivot-eliminated reduced A_od coordinates,
# gravity satisfied by construction -- matches the production D20 driver's own
# search space per docs/fullA_common_marginals_handoff.md Section 1's correction).
#
# Objective gradient is trivial (e[1]); the constraint gradient is the CM-aware
# Lfix gradient (cm_production_gradient, Section 4+3A+5), reusing the SAME base
# dual state between cb_F! and cb_G! at a shared point (mirrors
# c10_d20_production_driver.jl's last_F_state pattern) to avoid a redundant
# inner solve on every KNITRO gradient call.
# ============================================================================

using KNITRO

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

"""
    run_cm_upper(pcx, ctx, pe, w0; delta=1.0, maxtime_real=180.0, opt_file="csw_outer_wallclock_sr1.opt",
                  z_halfwidth=30.0, gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi) -> NamedTuple

One constrained-upper-bound outer KNITRO solve against the CM production
bundle `pcx` (`cm_production_bundle.jl::build_cm_production_context`).
Returns the best exact-feasible incumbent found (not merely the terminal
iterate -- KNITRO's own `Delta<=delta+1e-6` bookkeeping tracks this exactly
as `c10_d20_production_driver.jl` already does), plus the full evaluation
trace and KNITRO's own terminal status for diagnostic purposes.
"""
function run_cm_upper(pcx, ctx, pe, w0::Vector{Float64};
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        verbose::Bool = true)
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
    bandwidth_cache = Dict{Int,Float64}()
    t_start = time()

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        local base
        try
            _, base = cm_production_value(xf, pcx)
        catch e
            throw(DomainError(w[1], "run_cm_upper: infeasible/failed inner solve at this point"))
        end
        # Remediation fix (task Part A, finding F1): `-base.ζstar` silently omits mean(Psi(q*))
        # and overstates the divergence at tail-active points (m* > e) -- see cm_checkpoint.jl's
        # identical fix and cm_production_bundle.jl's delta_dual_from_base docstring. This plain
        # (un-checkpointed) driver has no `verify` tuple available cheaply from
        # cm_production_value, so recompute the canonical Delta_dual directly (one extra
        # obj(...) call, same recompute pattern archC_verified_state uses).
        Δ = delta_dual_from_base(pcx.ctx_cm.obj, base)
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        is_new_best = feasible && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = time() - t_start)
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible))
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            println("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        gfull, meta = cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
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
