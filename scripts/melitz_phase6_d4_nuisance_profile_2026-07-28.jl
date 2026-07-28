# Governing prompt (2026-07-28 outer-search gamma-profile session), Phase 6: D4 scaled
# nuisance-profile curve -- min_{A,f} DeltaStar(g,A,f) at each g on the SAME predetermined
# fraction grid Phase 2 used (subset, for wall-clock reasons), full A/f free (free_mask
# excludes only g itself), continuation-warm-started from the preceding solved g's own
# nuisance coordinates + inner dual, fixed-A/f (theta0 with only g moved) always retained as
# a verified incumbent -- a flexible point may never be reported worse than it.
#
# Usage: julia --project=. scripts/melitz_phase6_d4_nuisance_profile_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
const CAP = 10.0
# Subset of Phase 2's own grid: dense enough near the interesting (finite, non-cap) region
# Phase 2 already located (frac<=0.5 all FiniteSolved for D4), sparser beyond it.
const FRACTIONS = [0.00, 0.10, 0.20, 0.35, 0.50]

kappa_of_g(g::Real, wratio::Real, sigma::Real) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa::Real, wratio::Real, sigma::Real) = log((kappa / wratio)^(sigma - 1))

function main()
    BLAS.set_num_threads(1)
    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free)
    ctx = obj.γ
    n = length(theta0)
    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
    @assert r0.verified

    j = ctx.target_country
    state0 = melitz_outer_state(theta0, ctx)
    lambda_jj = state0.equilibrium.trade_flow[j, j] / state0.equilibrium.expenditure[j]
    wratio = 1.0 / ctx.w[j]
    sigma = ctx.sigma
    kappa_pareto = kappa_of_g(theta0[1], wratio, sigma)
    kappa_min = lambda_jj^(1 / (sigma - 1))

    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    inner_cfg = MelitzInnerSolveConfig(:diagnostic; delta_evaluation_cap=CAP)
    free_mask = trues(n); free_mask[1] = false

    rows = NamedTuple[]
    warm_x = nothing
    warm_theta_nuisance = copy(theta0)
    for frac in FRACTIONS
        kappa_f = kappa_pareto + frac * (kappa_min - kappa_pareto)
        g_f = frac == 0.0 ? theta0[1] : g_of_kappa(kappa_f, wratio, sigma)
        theta_start = copy(warm_theta_nuisance); theta_start[1] = g_f
        # Fixed-A/f reference at this g (verified incumbent floor)
        theta_fixed = copy(theta0); theta_fixed[1] = g_f
        r_fixed = evaluate_melitz_delta(theta_fixed, ctx, obj; cold=true, store_G=false)

        println("="^80); @printf("g=%.6f (frac=%.2f)  fixed-A/f Delta=%s\n", g_f, frac,
            r_fixed.verified ? @sprintf("%.4e", r_fixed.Delta) : "UNVERIFIED nStatus=$(r_fixed.nStatus)")
        local res
        wall = @elapsed begin
            res = solve_melitz_nuisance_min_delta(ctx, obj, theta_start; free_mask=free_mask,
                radius=0.3, gradient_backend=:B_direct_argument_parallel, h=1e-4,
                inner_loop_opt=inner_opt, warm_start_x=warm_x, inner_solve_config=inner_cfg,
                forbid_dense_fallback=true)
        end
        ok = res.nStatus in (0, -101, -102, -103)
        flex_delta = res.r_final.Delta
        improved = r_fixed.verified && res.r_final.verified && flex_delta <= r_fixed.Delta
        @printf("  nuisance-min: nStatus=%d wall=%.2fs n_fc=%d n_ga=%d Delta_min=%.4e ok=%s improved_on_fixed=%s\n",
            res.nStatus, wall, res.n_fc_calls, res.n_ga_calls, flex_delta, res.r_final.verified, improved)
        flush(stdout)

        # report the BETTER of the two (never worse than fixed-A/f, per governing prompt)
        report_delta = (res.r_final.verified && (!r_fixed.verified || flex_delta <= r_fixed.Delta)) ?
            flex_delta : (r_fixed.verified ? r_fixed.Delta : NaN)
        push!(rows, (fixture="D4_seed29_W20000", gamma_fraction=frac, g=g_f,
            Delta_fixed_af=r_fixed.verified ? r_fixed.Delta : NaN,
            Delta_nuisance_min=res.r_final.verified ? flex_delta : NaN,
            Delta_reported=report_delta, nuisance_ok=ok, nStatus=res.nStatus,
            wall_s=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls, improved=improved))

        if res.r_final.verified
            warm_x = res.r_final.dual_x
            warm_theta_nuisance = copy(res.theta_final)
        end
    end

    outfile = joinpath(OUTDIR, "melitz_phase6_d4_nuisance_profile_2026-07-28.csv")
    open(outfile, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
    println("\nDONE. CSV written to ", outfile)
    return rows
end

main()
