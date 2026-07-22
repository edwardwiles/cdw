# Production-gate addendum, Section 13 (minimal version): a real, short, multi-iteration outer
# KN_solve trajectory at real D=20/W=80000, comparing the Reference gradient
# (composite_gradient_at_fast_buffered) against Backend C+ (composite_gradient_at_Cplus).
#
# Deliberately NOT a modification of c10_d20_production_driver.jl (that file's own recent
# .opt-related change caused a real regression this session -- see
# docs/fullA_nested_knitro_solve_hang_rootcause.md -- so it is not touched again here). This is
# a standalone, additive script that builds its own minimal outer KN_solve, reusing
# screened_eval/ScreenCounters/direction_gamma_bounds from c10_d20_production_driver.jl (proven,
# unmodified) for the objective/constraint side, and composite_gradient_at_fast_buffered /
# composite_gradient_at_Cplus for the gradient side -- everything needed for a genuine trajectory
# comparison, none of the checkpoint/dual-bank/negative-cache/organic-capture machinery a full
# production run also has (out of scope for this comparison).
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
using Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))

"""
    run_short_trajectory(label, gradient_kind::Symbol, g0, zfree0, ctx, pe, rsc;
                          maxtime_real=60.0) -> NamedTuple

`gradient_kind` in (:reference, :cplus). Mirrors run_polish_checkpointed's essential structure
(direction-box validation, screened objective, KNITRO outer KN_solve) without checkpointing/dual
banks/negative cache -- a real, but minimal, trajectory driver for this comparison only.
"""
function run_short_trajectory(label::String, gradient_kind::Symbol, g0::Float64, zfree0::Vector{Float64},
        ctx, pe, rsc::RangedScreenContext; maxtime_real::Float64 = 60.0, find_smallest::Bool = true)
    D = ctx.D; D2 = D^2
    x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    sc = ScreenCounters()
    n_eval = Ref(0)

    w0 = vcat(g0, zfree0)
    r0, _ = screened_eval(x_free_from_w(w0, pe), ctx, rsc, sc, n_eval; warm = false)
    lp("[", label, "] start point: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)
    r0.inner_status in FEASIBLE_CODES || error("run_short_trajectory($label): start point not inner-feasible")

    z_halfwidth = 30.0
    gp_dir_lo, gp_dir_hi = direction_gamma_bounds(ctx, find_smallest)
    w_lo = vcat(gp_dir_lo, zfree0 .- z_halfwidth)
    w_hi = vcat(gp_dir_hi, zfree0 .+ z_halfwidth)

    grad_pool = build_grad_workspace_pool(size(ctx.obj.U, 1))
    ws_c = gradient_kind == :cplus ? build_lfix_factorized_workspace(D, size(ctx.obj.U, 1)) : nothing
    bandwidth_cache = Dict{Int,Float64}()

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    n_grad_calls = Ref(0); n_eval_calls = Ref(0)
    trace = NamedTuple[]
    last_base = Ref{Union{Nothing,BaseDualState}}(nothing)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        n_eval_calls[] += 1
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true)
        if !(r.inner_status in FEASIBLE_CODES)
            r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
        end
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = r.Delta_dual
        if r.inner_status in FEASIBLE_CODES
            last_base[] = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        end
        push!(trace, (n_eval = n_eval_calls[], w1 = w[1], Delta = r.Delta_dual, inner_status = r.inner_status))
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        base = last_base[] !== nothing && last_base[].x_free0 == xf ? last_base[] : solve_base_state(xf, ctx)
        if gradient_kind == :reference
            gfull, _ = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        else
            gfull, _ = composite_gradient_at_Cplus(xf, ctx, pe, grad_pool, ws_c; base = base, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        end
        n_grad_calls[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        evalResult.jac .= gfull
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    t0 = time()
    KNITRO.KN_solve(kc)
    wall = time() - t0
    nStatus, objSol, xsol, _ = KNITRO.KN_get_solution(kc)
    κ = 1 - xsol[1]^(ctx.σ / (ctx.σ - 1))
    lp("[", label, "] DONE: status=", nStatus, " wall=", round(wall, digits = 2), "s n_eval=", n_eval_calls[],
       " n_grad_calls=", n_grad_calls[], " gp_final=", xsol[1], " kappa=", κ)
    KNITRO.KN_free(kc)

    # cold re-verify the final point via the trusted exact-hard value path
    r_final, _ = screened_eval(x_free_from_w(xsol, pe), ctx, rsc, ScreenCounters(), Ref(0); warm = false)
    lp("[", label, "] cold re-verify: inner_status=", r_final.inner_status, " Delta=", r_final.Delta_dual)

    return (label = label, gradient_kind = gradient_kind, knitro_status = nStatus, wall = wall,
        n_eval = n_eval_calls[], n_grad_calls = n_grad_calls[], gp_final = xsol[1], kappa = κ,
        Delta_final_cold = r_final.Delta_dual, inner_status_final_cold = r_final.inner_status, trace = trace)
end

lp("=== c18_short_trajectory_comparison === ", Dates.now())
for δ in (1.0, 2.0)
    lp(""); lp("="^100); lp("δ = ", δ); lp("="^100)
    t0 = time()
    ctx = d20_real_setup(W = 80000, δ = δ, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    rsc = build_ranged_screen_context(ctx)
    D = ctx.D; D2 = D^2
    lp(@sprintf(">>> ctx built in %.1fs", time() - t0))

    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
    gp0 = ctx.θ0_up[3+D]

    res_ref = run_short_trajectory("delta$(δ)_reference", :reference, gp0, zfree0, ctx, pe, rsc; maxtime_real = 60.0)
    res_cplus = run_short_trajectory("delta$(δ)_cplus", :cplus, gp0, zfree0, ctx, pe, rsc; maxtime_real = 60.0)

    lp(@sprintf("SUMMARY δ=%.1f: ref kappa=%.6f (status=%d, n_eval=%d, wall=%.1fs)  C+ kappa=%.6f (status=%d, n_eval=%d, wall=%.1fs)",
        δ, res_ref.kappa, res_ref.knitro_status, res_ref.n_eval, res_ref.wall,
        res_cplus.kappa, res_cplus.knitro_status, res_cplus.n_eval, res_cplus.wall))
    lp(@sprintf("  cold-reverified Delta:  ref=%.6e (status=%d)   C+=%.6e (status=%d)",
        res_ref.Delta_final_cold, res_ref.inner_status_final_cold, res_cplus.Delta_final_cold, res_cplus.inner_status_final_cold))
end
lp(">>> done ", Dates.now())
