# Continuation session (2026-07-24), Stage 2 (main prompt Sections 8-10): the profiled
# nuisance-minimization alternative to the constrained finite-delta search, at the real
# D=20/W=80,000/seed=1 calibrated reference.
#
# Section 8: A-only / f-only / full nuisance minimizations of Delta at the FIXED g_fixed
# (the Phase 6 fixed-A/f verified boundary point, g_fixed=-0.49783321) -- does A/f
# flexibility create local divergence slack at the restricted boundary?
#
# Sections 9-10: if nuisance minimization reduces Delta below 1 at g_fixed, continue in g
# (small increments, warm-started nuisance + dual from the preceding point) until the
# profiled minimum brackets Delta_profile(g)=1, then solve for the boundary by bisection.
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_profile_continuation_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm
using Roots: find_zero, Bisection

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT_PROFILE", joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt"))
const OUTER_OPT_PROFILE = get(ENV, "MELITZ_OUTER_OPT_PROFILE", joinpath(dirname(@__DIR__), "melitz_outer_nuisance_profile.opt"))
const G_FIXED_REFERENCE = -0.49783321
const RADIUS = parse(Float64, get(ENV, "MELITZ_NUISANCE_RADIUS", "0.5"))

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
end

kappa_of_g(g::Real, calib) = (calib.w_prime / calib.w[calib.target_country]) * exp(g)^(1 / (calib.sigma - 1))

# ============================================================================
# Section 8: A-only / f-only / full nuisance minimizations at g_fixed.
# ============================================================================
function section8_matched_problems(ctx, obj_inner, theta_fixed_af, calib)
    println("="^100)
    println("SECTION 8: A-only, f/q-only, full nuisance minimizations at g_fixed")
    println("="^100)
    r0 = evaluate_melitz_delta(theta_fixed_af, ctx, obj_inner; cold=true, store_G=false)
    @printf("Starting point (fixed A/f, calibrated): g_fixed=%.8f  Delta=%.6e  nStatus=%d\n",
        theta_fixed_af[1], r0.Delta, r0.nStatus)
    flush(stdout)

    results = Dict{Symbol,Any}()
    for (label, block) in ((:A_only, :A_only), (:f_only, :f_only), (:full, :full))
        mask = melitz_nuisance_free_mask(ctx; block=block)
        println("\n--- $label (n_free=$(count(mask))) ---")
        flush(stdout)
        # `r0` (cold=true) leaves obj_inner.x set to its own converged dual but
        # obj_inner.use_cached_x FALSE (evaluate_melitz_delta's cold path clears the flag
        # and nothing turns it back on after a successful solve, cc_algo/inner_loop_functions.jl's
        # own inner_loop_internal only updates .x, never .use_cached_x) -- so without an
        # explicit warm_start_x here, EVERY block's first inner solve at theta_fixed_af
        # itself (an UNPERTURBED repeat of r0's own problem) would start from an all-zeros
        # cold dual at the single most fragile point in this whole experiment (the actual
        # Delta~1 boundary) instead of trivially reusing r0's own already-converged answer.
        # Found live this session (a >6-minute stall on the very first inner solve before
        # this fix).
        t0 = time()
        res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_fixed_af; free_mask=mask,
            radius=RADIUS, gradient_backend=:B_direct_argument_parallel, h=1e-4,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_PROFILE, warm_start_x=r0.dual_x)
        wall = time() - t0
        A_start, f_start, _, _ = melitz_expand_theta(theta_fixed_af, ctx)
        A_final, f_final, _, _ = melitz_expand_theta(res.theta_final, ctx)
        state_final = melitz_outer_state(res.theta_final, ctx)
        @printf("[%s] nStatus=%-6d Delta_min(KNITRO)=%.6e Delta_min(cold-verified)=%.6e wall=%.2fs n_fc=%-4d n_ga=%-4d verified=%s\n",
            label, res.nStatus, res.Delta_min, res.r_final.Delta, wall, res.n_fc_calls, res.n_ga_calls, res.r_final.verified)
        @printf("    ||dA||=%.4f  ||df||=%.4f  min_slack=%.5f\n",
            norm(A_final .- A_start), norm(f_final .- f_start), state_final.min_slack)
        flush(stdout)
        results[label] = (result=res, wall=wall, dA=norm(A_final .- A_start), df=norm(f_final .- f_start),
            min_slack=state_final.min_slack)
    end
    return r0, results
end

# ============================================================================
# Sections 9-10: continuation in g (full nuisance block, warm-started), then bisection
# for the profiled boundary Delta_profile(g)=1.
# ============================================================================
function delta_profile_at(ctx, obj_inner, theta_g; mask, radius, warm_start_x=nothing)
    res = solve_melitz_nuisance_min_delta(ctx, obj_inner, theta_g; free_mask=mask, radius=radius,
        gradient_backend=:B_direct_argument_parallel, h=1e-4,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_PROFILE, warm_start_x=warm_start_x)
    return res
end

function section9_10_continuation(ctx, obj_inner, theta_fixed_af, calib; max_steps::Int=8,
                                   g_increment_init::Float64=0.0025)
    println("\n" * "="^100)
    println("SECTIONS 9-10: continuation in g (full nuisance block, warm-started)")
    println("="^100)
    mask = melitz_nuisance_free_mask(ctx; block=:full)

    # Seed the FIRST continuation step's warm start too (same rationale as Section 8's own
    # fix above -- otherwise the first step cold-starts at the fragile g_fixed boundary).
    r0 = evaluate_melitz_delta(theta_fixed_af, ctx, obj_inner; cold=true, store_G=false)

    theta_cur = copy(theta_fixed_af)
    dual_x = copy(r0.dual_x)
    trajectory = NamedTuple[]
    g_increment = g_increment_init
    prev_Delta = nothing

    step = 0
    while step < max_steps
        step += 1
        res = delta_profile_at(ctx, obj_inner, theta_cur; mask=mask, radius=RADIUS, warm_start_x=dual_x)
        kappa = kappa_of_g(theta_cur[1], calib)
        @printf("[continuation step %2d] g=%.6f  kappa=%.6f  Delta_profile(KNITRO)=%.6e  Delta_profile(cold)=%.6e  nStatus=%d  wall=%.2fs\n",
            step, theta_cur[1], kappa, res.Delta_min, res.r_final.Delta, res.nStatus, res.wall)
        flush(stdout)
        push!(trajectory, (step=step, g=theta_cur[1], kappa=kappa, Delta=res.r_final.Delta, nStatus=res.nStatus))

        # bracket check: once we cross Delta=1, stop -- have both sides.
        if prev_Delta !== nothing && ((prev_Delta <= 1.0) != (res.r_final.Delta <= 1.0))
            println("  -> bracket found (Delta_profile crossed 1 between consecutive continuation points)")
            return trajectory, mask
        end
        if res.r_final.Delta > 1.0
            println("  -> Delta_profile already > 1 at this g -- no further continuation needed toward larger GT")
            return trajectory, mask
        end

        # warm-start the NEXT point from this point's own converged full theta/dual.
        dual_x = copy(res.r_final.dual_x)
        theta_cur = copy(res.theta_final)
        # adapt increment: if Delta moved a lot, shrink; if it barely moved, grow (bounded).
        if prev_Delta !== nothing
            dDelta = abs(res.r_final.Delta - prev_Delta)
            if dDelta > 0.15
                g_increment = max(g_increment / 2, 0.0005)
            elseif dDelta < 0.02
                g_increment = min(g_increment * 1.5, 0.01)
            end
        end
        prev_Delta = res.r_final.Delta
        theta_cur[1] -= g_increment   # move toward the upper-GT-bound direction (g falling)
    end
    println("  -> max_steps reached without a clean bracket")
    return trajectory, mask
end

function section10_boundary_solve(ctx, obj_inner, trajectory, mask, calib)
    println("\n" * "="^100)
    println("SECTION 10: profiled boundary solve (Delta_profile(g)=1)")
    println("="^100)
    below1 = filter(t -> t.Delta <= 1.0, trajectory)
    above1 = filter(t -> t.Delta > 1.0, trajectory)
    if isempty(below1) || isempty(above1)
        println("  No bracket available from the continuation trajectory -- cannot bisect.")
        return nothing
    end
    g_lo = minimum(t.g for t in below1)   # most extreme g still <=1
    g_hi = maximum(t.g for t in above1 if t.g < g_lo; init=g_lo + 0.01)
    @printf("  bracket: g_hi=%.6f (Delta>1 side)  g_lo=%.6f (Delta<=1 side)\n", g_hi, g_lo)

    root_fn(g) = begin
        theta_g = copy(_shared_theta_calib[])
        theta_g[1] = g
        res = delta_profile_at(ctx, obj_inner, theta_g; mask=mask, radius=RADIUS)
        res.r_final.Delta - 1.0
    end
    g_star = find_zero(root_fn, (g_hi, g_lo), Bisection(); xatol=1e-3)
    theta_star = copy(_shared_theta_calib[]); theta_star[1] = g_star
    res_star = delta_profile_at(ctx, obj_inner, theta_star; mask=mask, radius=RADIUS)
    kappa_star = kappa_of_g(g_star, calib)
    @printf("\n  g_profile_star=%.8f  kappa_profile=%.8f  GT_profile=%.8f  Delta=%.6e\n",
        g_star, kappa_star, 1 - kappa_star, res_star.r_final.Delta)
    return (g_star=g_star, kappa_star=kappa_star, res_star=res_star)
end

const _shared_theta_calib = Ref{Vector{Float64}}()

function main()
    calib = load_calibration()
    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    _shared_theta_calib[] = copy(theta_calib)

    theta_fixed_af = copy(theta_calib); theta_fixed_af[1] = G_FIXED_REFERENCE

    r0, section8_results = section8_matched_problems(ctx, obj_inner, theta_fixed_af, calib)

    trajectory, mask = section9_10_continuation(ctx, obj_inner, theta_fixed_af, calib)

    boundary = section10_boundary_solve(ctx, obj_inner, trajectory, mask, calib)

    println("\n" * "="^100)
    println("STAGE 2 SUMMARY")
    println("="^100)
    @printf("Fixed A/f Delta at g_fixed: %.6e\n", r0.Delta)
    for label in (:A_only, :f_only, :full)
        r = section8_results[label]
        @printf("  %-8s Delta_min=%.6e  ||dA||=%.4f  ||df||=%.4f  wall=%.2fs\n",
            label, r.result.r_final.Delta, r.dA, r.df, r.wall)
    end
    if boundary !== nothing
        @printf("Profiled boundary: g=%.8f  kappa=%.8f  GT=%.8f\n",
            boundary.g_star, boundary.kappa_star, 1 - boundary.kappa_star)
    else
        println("Profiled boundary: NOT FOUND (no bracket)")
    end

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return r0, section8_results, trajectory, boundary, calib, ctx, obj_inner
end

main()
