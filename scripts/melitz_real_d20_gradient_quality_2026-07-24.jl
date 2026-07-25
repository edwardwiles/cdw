# Continuation session (2026-07-24), Phase 5: gradient-quality diagnostics at the real
# D=20/W=80,000/seed=1 calibrated reference point (the ONLY seed found to converge cleanly
# at W=80,000 in the Phase 3 seed sweep -- see docs). Compares the direct fixed-dual
# secant (:B_direct_argument_serial) against a genuinely REOPTIMIZED (independently
# re-solved) central-difference secant of Delta(theta), for a representative direction set.
#
# Scoping decision (documented, following this repo's established practice of not running
# a full grid once a representative subset is informative -- e.g. the closure-audit
# session's Phase C3): h=1e-4 (the production default bandwidth) for all 8 directions,
# plus h=1e-3 as a single-order-of-magnitude robustness cross-check on 2 representative
# directions, rather than the full {3e-5,1e-4,3e-4,1e-3} x 8 grid (32 directions x
# bandwidths x 2 signs = 64 real cold KNITRO solves) -- each REOPTIMIZED solve is a real,
# potentially-slow cold KNITRO problem (Phase 3's seed sweep shows this fixture is not
# uniformly well-conditioned away from the exact calibrated point), so the full grid would
# cost tens of minutes to hours for marginal additional information at fixed h=1e-4.
#
# Usage: julia --project=. scripts/melitz_real_d20_gradient_quality_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm
using Random

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")

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

function build_directions(ctx, theta_free, D, j)
    n = length(theta_free)
    nA = D^2 - 1
    dirs = Dict{String,Vector{Float64}}()

    # 1. gamma direction
    v = zeros(n); v[1] = 1.0; dirs["gamma"] = v

    # 2/4. A-pivot-sensitive / f-pivot-sensitive: coordinate with largest |c[other[k]]|
    # (largest effect on the eliminated pivot cell under pivot_expand's affine map).
    A_other_c = abs.(ctx.A_pivot.c[ctx.A_pivot.other])
    idx_Apivot_sens = argmax(A_other_c)          # index INTO A_free block (1..nA)
    idx_Aordinary = argmin(A_other_c)             # smallest pivot sensitivity: "ordinary" A direction

    v = zeros(n); v[1+idx_Apivot_sens] = 1.0; dirs["A_pivot_sensitive"] = v
    v = zeros(n); v[1+idx_Aordinary] = 1.0; dirs["A_ordinary"] = v

    # f block starts at index 2+nA. Rebuild the f-pivot the SAME way expand_free_theta does
    # (its offset depends on f[j,j] at theta_free's own point, but the SENSITIVITY ranking
    # -- which free f coordinate moves the f-pivot cell most -- depends only on c, not on g0).
    avoid_f = f_gravity_pivot_avoid_indices(D, ctx.f_free_lin, ctx.A_pivot.pivot)
    f_pivot = build_gravity_pivot(ctx.c_full[ctx.f_free_lin], 0.0; avoid=avoid_f)
    f_other_c = abs.(f_pivot.c[f_pivot.other])
    idx_fpivot_sens = argmax(f_other_c)
    idx_fordinary = argmin(f_other_c)
    v = zeros(n); v[1+nA+idx_fpivot_sens] = 1.0; dirs["f_pivot_sensitive"] = v
    v = zeros(n); v[1+nA+idx_fordinary] = 1.0; dirs["f_ordinary"] = v

    # focal-origin direction: an A_free coordinate whose physical cell has origin==j.
    focal_A_free_idx = findfirst(k -> lin2od(ctx.A_pivot.other[k], D)[1] == j, 1:length(ctx.A_pivot.other))
    v = zeros(n); v[1+focal_A_free_idx] = 1.0; dirs["focal_origin"] = v

    # two random normalized block directions (mix of A_free and f_free_free coordinates).
    rng = MersenneTwister(20260724)
    v = randn(rng, n); v[1] = 0.0; v ./= norm(v); dirs["random_block_1"] = v
    v = randn(rng, n); v[1] = 0.0; v ./= norm(v); dirs["random_block_2"] = v

    return dirs
end

function main(; W::Int=80_000, seed::Int=1)
    calib = load_calibration()
    BLAS.set_num_threads(16)
    obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=seed, inner_loop_opt=INNER_OPT)
    ctx = obj.γ
    D, j = ctx.D, ctx.target_country

    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true)
    @printf("Reference point: Delta=%.6e  nStatus=%d  verified=%s\n\n", r0.Delta, r0.nStatus, r0.verified)
    x = r0.dual_x

    dirs = build_directions(ctx, theta0, D, j)
    grad_serial! = make_melitz_gradient_delta_direct_serial(1e-4)
    n = length(theta0)
    g_direct = zeros(n)
    grad_serial!(g_direct, theta0, ctx, obj, x)   # d(1e10*Delta)/dtheta

    cache = MelitzDeltaEvalCache(64)
    println("="^120)
    @printf("%-22s %-8s %14s %14s %14s %14s %8s\n", "direction", "h", "direct_deriv", "reopt_secant", "sign_match", "rel_err", "switches")
    println("="^120)

    results = NamedTuple[]
    for (name, v) in dirs
        for h in (1e-4,)
            direct_deriv = dot(g_direct, v) / 1e10   # predicted d(Delta)/dtheta . v

            theta_p = theta0 .+ h .* v
            theta_m = theta0 .- h .* v
            rp = evaluate_melitz_delta(theta_p, ctx, obj; cold=true, cache=cache)
            rm = evaluate_melitz_delta(theta_m, ctx, obj; cold=true, cache=cache)
            reopt_secant = (rp.Delta - rm.Delta) / (2h)

            # Genuine participation-switch count: for every (o,d) cell, count reference
            # draws z_draws[w,o] that lie strictly between the two displaced points' cutoffs
            # (theta_p vs theta_m) -- exactly the draws whose active/inactive status flips
            # because of this h-sized move, summed over all D^2 cells.
            state_p = melitz_outer_state(theta_p, ctx)
            state_m = melitz_outer_state(theta_m, ctx)
            n_switch = 0
            for o in 1:D, d in 1:D
                lo, hi = minmax(state_p.cutoff[o, d], state_m.cutoff[o, d])
                n_switch += count(z -> lo < z <= hi, @view obj.U[:, o])
            end

            sign_match = sign(direct_deriv) == sign(reopt_secant) || abs(reopt_secant) < 1e-10
            rel_err = abs(direct_deriv - reopt_secant) / max(abs(reopt_secant), 1e-10)

            @printf("%-22s %-8.0e %14.6e %14.6e %14s %14.3e %8d\n",
                name, h, direct_deriv, reopt_secant, sign_match, rel_err, n_switch)
            push!(results, (name=name, h=h, direct=direct_deriv, reopt=reopt_secant,
                sign_match=sign_match, rel_err=rel_err, n_switch=n_switch,
                rp_verified=rp.verified, rm_verified=rm.verified, rp_nStatus=rp.nStatus, rm_nStatus=rm.nStatus))
            flush(stdout)
        end
    end

    println("\n" * "="^120)
    println("h=1e-3 robustness cross-check on 2 representative directions (gamma, A_pivot_sensitive)")
    println("="^120)
    for name in ("gamma", "A_pivot_sensitive")
        v = dirs[name]
        h = 1e-3
        direct_deriv = dot(g_direct, v) / 1e10
        rp = evaluate_melitz_delta(theta0 .+ h .* v, ctx, obj; cold=true, cache=cache)
        rm = evaluate_melitz_delta(theta0 .- h .* v, ctx, obj; cold=true, cache=cache)
        reopt_secant = (rp.Delta - rm.Delta) / (2h)
        @printf("%-22s %-8.0e %14.6e %14.6e %14s\n", name, h, direct_deriv, reopt_secant,
            sign(direct_deriv) == sign(reopt_secant))
        flush(stdout)
    end

    println("\n" * "="^120)
    println("Final registered divergence-constraint Jacobian vs. FD of the FINAL registered constraint")
    println("="^120)
    # c_delta(theta) = Delta(theta)/delta; registered jac = grad(Delta)/delta. Verify with
    # delta=1 (Phase 6/7's own budget) directly against the reoptimized secants above.
    delta_budget = 1.0
    for r in results
        registered_jac_component = r.direct / delta_budget
        fd_of_registered = r.reopt / delta_budget
        @printf("  %-22s registered=%.6e  FD-of-registered=%.6e  rel_err=%.3e\n",
            r.name, registered_jac_component, fd_of_registered,
            abs(registered_jac_component - fd_of_registered) / max(abs(fd_of_registered), 1e-10))
    end

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return results
end

main()
