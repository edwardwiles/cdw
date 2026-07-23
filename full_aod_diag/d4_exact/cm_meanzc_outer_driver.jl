# ============================================================================
# Constrained-upper-bound KNITRO outer loop for the CM+moments(+ZC) production
# bundle, generalizing cm_outer_driver.jl's run_cm_upper. Single-shot (no
# checkpoint/resume) -- exactly like run_cm_upper itself, this is the
# diagnostic/gate-testing entry point, NOT the production entry point for a
# real/reportable D=20 run (checkpoint/resume integration is a separate,
# later increment, mirroring cm_checkpoint.jl's relationship to
# cm_outer_driver.jl).
#
# w_ext = [gamma'_focal; zfree; eta_nu_1; ...; eta_nu_{K_mean}] when
# cfg.cm_extension != :cm_only (K_mean extra trailing coordinates, log-nu
# parameterized); w_ext = w = [gamma'_focal; zfree] UNCHANGED when
# cfg.cm_extension == :cm_only, which DELEGATES BYTE-FOR-BYTE to the existing
# run_cm_upper (cm_outer_driver.jl) -- the extension code in this file is
# never reached at all on that path, matching the task brief's "a CM-only run
# with the extension code present but disabled must use the same current
# production call path" requirement.
# ============================================================================

using KNITRO

"x_free_from_w_meanzc(w_ext, pe, K_mean) -> x_free. Strips the trailing K_mean eta_nu coordinates and delegates to the UNCHANGED x_free_from_w (cm_outer_driver.jl)."
x_free_from_w_meanzc(w_ext::AbstractVector, pe, K_mean::Int) = x_free_from_w(@view(w_ext[1:end-K_mean]), pe)

"nu_from_w_meanzc(w_ext, K_mean) -> Vector{Float64}. nu_k = exp(eta_nu_k), the trailing K_mean coordinates."
nu_from_w_meanzc(w_ext::AbstractVector, K_mean::Int) = exp.(w_ext[end-K_mean+1:end])

"""
    run_cm_meanzc_upper(cfg::CMMeanZCConfig, ctx, pe, w0_ext; delta=1.0, maxtime_real=180.0,
        opt_file="csw_outer_wallclock_sr1.opt", z_halfwidth=30.0, gp_lo=ctx.bounds.γp_lo,
        gp_hi=ctx.bounds.γp_hi, verbose=true) -> NamedTuple

`cfg.cm_extension === :cm_only`: builds the plain CM production context
(`build_cm_production_context`) and calls `run_cm_upper` UNCHANGED --
`w0_ext` must be a plain `w0` in that case (no trailing eta_nu coordinates).

Otherwise: one constrained-upper-bound outer KNITRO solve against the
CM+moments(+ZC) production bundle (`cm_meanzc_production.jl`). `w0_ext` must
be `[gp; zfree; eta_nu_1;...;eta_nu_{K_mean}]`. ν_k's box comes from
`cfg.meanzc_nu_bounds` if set, else `meanzc_default_nu_bounds` (a
deliberately wide box derived from the actual draws, task brief Section 6).
Returns the best exact-feasible incumbent found, the full evaluation trace,
KNITRO's terminal status, and (new vs. `run_cm_upper`) `K_mean`/`K_pair`/
`nu_bounds`/`nu_box_interior` (per-level "did the solved η_ν,k land away from
its box edge" diagnostic, `meanzc_verify_box_not_binding`).
"""
function run_cm_meanzc_upper(cfg::CMMeanZCConfig, ctx, pe, w0_ext::Vector{Float64};
        delta::Float64 = 1.0, maxtime_real::Float64 = 180.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        gp_lo::Float64 = ctx.bounds.γp_lo, gp_hi::Float64 = ctx.bounds.γp_hi,
        verbose::Bool = true)
    _meanzc_validate(cfg)
    K_mean, K_pair = meanzc_resolve_K(cfg)

    if cfg.cm_extension === :cm_only
        pcx0 = build_cm_production_context(ctx, CS; L = cfg.cm.cm_grid_size, contrasts = cfg.cm.contrasts)
        return run_cm_upper(pcx0, ctx, pe, w0_ext; delta = delta, maxtime_real = maxtime_real,
            opt_file = opt_file, z_halfwidth = z_halfwidth, gp_lo = gp_lo, gp_hi = gp_hi, verbose = verbose)
    end

    pcx = build_cm_meanzc_production_context(ctx, CS; L = cfg.cm.cm_grid_size, K_mean = K_mean, K_pair = K_pair,
        contrasts = cfg.cm.contrasts, meanzc_basis = cfg.meanzc_basis)
    nu_bounds = cfg.meanzc_nu_bounds !== nothing ? cfg.meanzc_nu_bounds : meanzc_default_nu_bounds(ctx, K_mean)

    D2ext = length(w0_ext)
    D2 = D2ext - K_mean
    length(w0_ext) == D2 + K_mean || error("run_cm_meanzc_upper: length(w0_ext)=$(length(w0_ext)) inconsistent with K_mean=$K_mean")
    w_lo = vcat(gp_lo, w0_ext[2:D2] .- z_halfwidth, first.(nu_bounds))
    w_hi = vcat(gp_hi, w0_ext[2:D2] .+ z_halfwidth, last.(nu_bounds))

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2ext)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0_ext)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(nothing)
    n_eval = Ref(0); n_grad = Ref(0); trace = NamedTuple[]
    t_start = time()

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w_meanzc(w, pe, K_mean)
        νvec = nu_from_w_meanzc(w, K_mean)
        local base, verify
        try
            _, base, verify = cm_meanzc_production_value_verified(xf, νvec, pcx)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_cm_meanzc_upper: infeasible/failed inner solve at this point")
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
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            println("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " nu=", νvec,
                     " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w_meanzc(w, pe, K_mean)
        νvec = nu_from_w_meanzc(w, K_mean)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        gext, meta = cm_meanzc_production_gradient(xf, νvec, pcx, ctx, pe; base = base)
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= gext
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2ext), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))
    box_ok = b === nothing ? nothing : meanzc_verify_box_not_binding(log.(nu_from_w_meanzc(b.w, K_mean)), nu_bounds)
    return (knitro_status = nStatus, wall = wall, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace,
            K_mean = K_mean, K_pair = K_pair, nu_bounds = nu_bounds, nu_box_interior = box_ok)
end
