# ============================================================================
# OZC-CROSS outer-loop smoke-test driver (2026-08-09 task, phase 3). Mirrors `run_cm_upper`
# (cm_outer_driver.jl) EXACTLY -- same KNITRO NLP structure (minimize w[1]=gp subject to
# Delta_dual(w)<=delta), same verified-success gating (is_verified_success, oracle.jl),
# same CMExpectedSolveFailure->reject_point rejection pattern, same last_F_state cb_F!/cb_G!
# sharing idiom -- adapted ONLY for origin-ZC-cross's wider coordinate space
# `w = [gp; zfree; eta]` (economic pivot-space PLUS the K_pair^2-grid eta/nu vector, absent from
# CM's w=[gp;zfree] since CM's own nu is handled differently) via the ALREADY-WRITTEN
# `cm_originzc_cross_production_gradient` (cm_originzc_cross_production.jl) for cb_G! and
# `archOZ_verified_state` (cm_originzc_production.jl, UNCHANGED) for cb_F!.
#
# This is deliberately NOT `run_originzc_upper_checkpointed` (cm_originzc_checkpoint.jl) --
# that 500+-line production driver has no support for OriginByPowerCrossLayout at all
# (OriginZCConfig/make_target_layout hard-validate power_target_layout in
# (:shared_by_power,:origin_by_power) and build_originzc_augmented_obj asserts
# layout isa OriginByPowerLayout||SharedByPowerLayout -- wiring OZC-CROSS all the way through
# its checkpointing/dual-bank/exact-cache/backend-switching machinery is real future production-
# integration work, out of scope for TODAY's smoke test, whose only job is confirming the outer
# KNITRO loop genuinely explores multiple points using the already-verified inner-solve/gradient
# machinery). No checkpoint/resume support -- single-shot, matching run_cm_upper's own documented
# scope for exactly this kind of short diagnostic call.
# ============================================================================

isdefined(Main, :x_free_from_w) || include(joinpath(@__DIR__, "cm_outer_driver.jl"))
isdefined(Main, :CMCheckpointV4) || include(joinpath(@__DIR__, "cm_checkpoint.jl"))   # cm_originzc_checkpoint.jl's own load-order assumption -- it references CMCheckpointV4 without including it itself
isdefined(Main, :originzc_profiled_nu_value) || include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))   # 2026-08-09 Variant D: originzc_profiled_nu_value
using KNITRO

"""
    run_originzc_cross_upper(pcx, ctx, pe, w0; delta=1.0, maxtime_real=60.0,
                              opt_file="csw_outer_wallclock_sr1.opt", z_halfwidth=30.0,
                              eta_halfwidth=4.0, gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi,
                              aml=nothing, verbose=true) -> NamedTuple

`w0 = vcat(gp0, zfree0, eta0)`. When `aml === nothing`, `eta0 = log.(nu0)` (dense, length
`n_eta(pcx.aug.layout)`), unchanged. When `aml::ActiveMeanLayout` is supplied (Variant D, 2026-08-09
-- mirrors `run_originzc_upper_checkpointed`'s own aml wiring, `cm_originzc_checkpoint.jl` lines
859-935, EXACTLY, adapted to this driver's simpler single-shot structure), `eta0` is the ACTIVE
vector (length `aml.n_eta_active`, one shorter than dense -- the focal `(aml.focal_origin,
aml.kstar)` coordinate has no eta at all, its value derived every callback via
`originzc_profiled_nu_value`/`scatter_nu_eff`). Returns the same-shaped result NamedTuple as
`run_cm_upper` (`knitro_status, wall, n_eval, n_grad, best, kappa, xsol, trace`).
"""
function run_originzc_cross_upper(pcx, ctx, pe, w0::Vector{Float64};
        delta::Float64 = 1.0, maxtime_real::Float64 = 60.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0, eta_halfwidth::Float64 = 4.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        aml::Union{Nothing,ActiveMeanLayout} = nothing,
        verbose::Bool = true)
    layout = pcx.aug.layout
    aml === nothing || aml.base === layout ||
        error("run_originzc_cross_upper: aml.base must be === pcx.aug.layout (got a different layout object)")
    n_eta_total = aml === nothing ? n_eta(layout) : aml.n_eta_active
    D2econ = length(w0) - n_eta_total   # 1 (gp) + length(zfree)
    D2 = length(w0)
    w_lo = vcat(gp_lo, w0[2:D2econ] .- z_halfwidth, w0[D2econ+1:end] .- eta_halfwidth)
    w_hi = vcat(gp_hi, w0[2:D2econ] .+ z_halfwidth, w0[D2econ+1:end] .+ eta_halfwidth)

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
    t_start = time()

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:D2econ], pe)
        νvec_active = exp.(w[D2econ+1:end])
        νfull = aml === nothing ? νvec_active : scatter_nu_eff(aml, νvec_active, originzc_profiled_nu_value(xf, ctx))
        local base, verify
        try
            base, verify = archOZ_verified_state(xf, νfull, pcx.ctx_cm)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_originzc_cross_upper: infeasible/failed inner solve at this point")
        end
        Δ = verify.Delta_dual
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base, verify = verify)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        verified = is_verified_success(verify)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = time() - t_start)
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if verbose
            println("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible, " verified=", verified, " inner_status=", verify.inner_status)
            flush(stdout)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:D2econ], pe)
        νvec_active = exp.(w[D2econ+1:end])
        νfull = aml === nothing ? νvec_active : scatter_nu_eff(aml, νvec_active, originzc_profiled_nu_value(xf, ctx))
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        verify = (shared !== nothing && shared.w == w) ? shared.verify : nothing
        gfull, meta = cm_originzc_cross_production_gradient(xf, νfull, pcx, ctx, pe; base = base, verify = verify, threaded = true)
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
