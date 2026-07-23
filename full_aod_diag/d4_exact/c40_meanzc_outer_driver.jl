# ============================================================================
# Section 5 (outer-loop integration) + Section 7 (D=4 trial) driver for the
# two extended arms (cm_plus_mean, cm_plus_mean_zero_covariance). Exact
# structural analog of cm_outer_driver.jl's `run_cm_upper`, extended by
# exactly ONE outer KNITRO variable, appended (not inserted) at the end:
#     w = [gamma'_focal; zfree; eta_nu]
# so `w[1:end-1]` is byte-identical in meaning to the pre-existing
# `w = [gamma'_focal; zfree]` convention -- cm_extension=:cm_only never
# reaches this file at all (that arm's driver is the pre-existing,
# byte-unmodified cm_outer_driver.jl/run_cm_upper).
#
# Objective is still w[1] (gamma'_focal) alone: the economic objective kappa
# does not depend directly on nu (math note Section 5 / task brief Section
# 5), so `evalResult.objGrad[end] = 0.0` by construction (never even set),
# consistent with the objective gradient being e[1] exactly as in
# cm_outer_driver.jl.
#
# nu_ref (aug.nu_ref) is mutated exactly once per cb_F! call, immediately
# before the inner solve, and NEVER inside cb_G! (cb_G! only ever READS
# nu_ref[] via the shared last_F_state, matching cb_G!'s existing "reuse the
# base from cb_F! at the same point" pattern) -- this is the sequencing this
# file's own docstring promises in mean_zero_cov_moments.jl.
# ============================================================================

using KNITRO

"""
    run_meanzc_upper(pcx, ctx, pe, w0; delta=1.0, maxtime_real=180.0, opt_file="csw_outer_wallclock_sr1.opt",
                       z_halfwidth=30.0, gp_lo=ctx.bounds.γp_lo, gp_hi=ctx.bounds.γp_hi,
                       eta_nu_lo, eta_nu_hi, verbose=true) -> NamedTuple

`pcx` must be a `build_cm_meanzc_production_context(...)` result (its `pcx.aug.cm_extension`
determines whether the pair block is active). `w0 = vcat(gp0, zfree0, eta_nu0)` -- the caller is
responsible for choosing `eta_nu0` (e.g. via a local profile at the starting economic point, per
Section 7/9's "initialize nu by an arm-specific local profile" requirement -- this function does
NOT auto-profile, it takes whatever `w0[end]` it is given as the KNITRO initial value).

Same single-shot (no checkpoint/resume) scope as `run_cm_upper` -- appropriate for the D=4 trial
(Section 7); the D=20 long-running trial (Section 8/9) uses a checkpointed analog
(`c40_meanzc_outer_driver_checkpointed.jl`).
"""
function run_meanzc_upper(pcx, ctx, pe, w0::Vector{Float64};
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        eta_nu_lo::Float64, eta_nu_hi::Float64,
        verbose::Bool = true)
    D2 = length(w0)   # 1 (gp) + n_free_econ (zfree) + 1 (eta_nu)
    w_lo = vcat(gp_lo, w0[2:end-1] .- z_halfwidth, eta_nu_lo)
    w_hi = vcat(gp_hi, w0[2:end-1] .+ z_halfwidth, eta_nu_hi)
    aug = pcx.aug

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
        xf = x_free_from_w(w[1:end-1], pe)
        η_ν = w[end]
        aug.nu_ref[] = exp(η_ν)
        local base, verify
        try
            _, base, verify = cm_meanzc_production_value_verified(xf, pcx)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_meanzc_upper: infeasible/failed inner solve at this point (eta_nu=$η_ν)")
        end
        @assert aug.nu_ref[] == exp(η_ν) "nu_ref was mutated during the inner solve -- sequencing violation"
        Δ = verify.Delta_dual
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base, verify = verify)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        verified = is_verified_success(verify)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, nu = exp(η_ν), n_eval = n_eval[], t = time() - t_start)
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], eta_nu = η_ν, nu = exp(η_ν),
                       Delta = Δ, feasible = feasible, verified = verified))
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            println("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1],
                    " nu=", exp(η_ν), " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:end-1], pe)
        shared = last_F_state[]
        base_verify_match = (shared !== nothing && shared.w == w)
        base = base_verify_match ? shared.base : nothing
        verify = base_verify_match ? shared.verify : nothing
        if base === nothing || verify === nothing
            η_ν = w[end]
            aug.nu_ref[] = exp(η_ν)
            _, base, verify = cm_meanzc_production_value_verified(xf, pcx)
        end
        # (g, A_od) block: existing CM-aware Lfix machinery, mean/pair columns held FIXED at
        # aug.nu_ref[]'s current value throughout (composite_gradient_at_fast_cm_meanzc never
        # mutates nu_ref -- see cm_meanzc_lfix_aware.jl's docstring)
        gfull, meta = cm_meanzc_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
            h_mode = :cached, bandwidth_cache = bandwidth_cache)
        n_grad[] += 1
        # eta_nu block: analytic envelope derivative (Section 6), needs only base.lambdastar +
        # verify.m_mean -- no per-coordinate work, no W-scale reconstruction
        d_eta = d_delta_dual_d_eta_nu(base.λstar, aug, aug.nu_ref[]; mean_m = verify.m_mean)
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0   # kappa's objective grad: e[1], zero at eta_nu (Section 5)
        evalResult.jac[1:end-1] .= gfull
        evalResult.jac[end] = d_eta
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
