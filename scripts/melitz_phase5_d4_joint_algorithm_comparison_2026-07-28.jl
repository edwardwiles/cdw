# Governing prompt (2026-07-28 outer-search gamma-profile session), Phase 5: D4 scaled
# joint algorithm comparison -- full 798->30-coordinate joint search (g + log A + log f),
# with the Phase 1 objective_scale fix ALWAYS applied (objective_scale=s_g), Phase 7
# (2026-07-27)'s own block scales (s_g=1e-4, s_A=1e-5, s_f=1e-5), native :linear cutoffs,
# evaluation cap=10, :logA/:logf production parameterization.
#
# Algorithms: Active Set, SQP, Interior/CG (Interior/Direct excluded -- 2026-07-27 Phase 8
# already found it runs away catastrophically with this scale set at both D=4 and real D=20,
# unrelated to the objective-scale question this session is diagnosing).
# deltas: 1e-2 and 1e-3, both directions (delta=1 upper as a secondary diagnostic, run last
# if time allows).
#
# Usage: julia --project=. -t 20 scripts/melitz_phase5_d4_joint_algorithm_comparison_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
const S_G, S_A, S_F = 1e-4, 1e-5, 1e-5

function main()
    BLAS.set_num_threads(1)
    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free)
    ctx = obj.γ
    n = length(theta0); D = ctx.D; nA = D^2 - 1
    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
    @assert r0.verified
    @printf("theta0 Delta0=%.6e  g0=%.6f\n", r0.Delta, theta0[1])

    var_scale = ones(n); var_scale[1] = S_G; var_scale[2:1+nA] .= S_A; var_scale[2+nA:end] .= S_F
    var_center = collect(Float64.(theta0))
    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    algs = [(:active_set, "melitz_outer_finite_delta_alg_active_2026-07-27.opt"),
            (:sqp, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt"),
            (:interior_cg, "melitz_outer_finite_delta_alg_cg_2026-07-27.opt")]

    rows = NamedTuple[]
    for (algname, optfile) in algs
        outer_opt = joinpath(REPO, optfile)
        for delta in (1e-2, 1e-3)
            for direction in (:upper, :lower)
                println("\n=== algorithm=$algname delta=$delta direction=$direction (objective_scale=$S_G) ===")
                local res
                wall = @elapsed begin
                    res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
                        delta_evaluation_cap=10.0, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
                        theta_box=2.0, cutoff_constraint_backend=:linear,
                        inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
                        var_scale=var_scale, var_center=var_center,
                        backend=:matrix_free, forbid_dense_fallback=true,
                        objective_scale=S_G)
                end
                cv = res.cold_verified_incumbent
                best_g = cv === nothing ? theta0[1] : cv.eval.theta_free[1]
                dg = best_g - theta0[1]
                dlogA = cv === nothing ? NaN : norm(cv.eval.theta_free[2:1+nA] .- theta0[2:1+nA])
                dlogf = cv === nothing ? NaN : norm(cv.eval.theta_free[2+nA:end] .- theta0[2+nA:end])
                push!(rows, (algorithm=algname, delta=delta, direction=direction, wall=wall,
                    nStatus=res.nStatus, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
                    n_inner_solved=res.n_inner_solved, n_infinite_delta_reject=res.n_infinite_delta_reject,
                    n_above_cap_reject=res.n_above_cap_reject,
                    n_numerical_failure_reject=res.n_numerical_failure_reject,
                    g0=theta0[1], best_g=best_g, dg=dg, norm_dlogA=dlogA, norm_dlogf=dlogf,
                    best_Delta=cv === nothing ? NaN : cv.eval.Delta,
                    best_GT=cv === nothing ? NaN : NaN))
                @printf("  nStatus=%d wall=%.2fs n_fc=%d n_ga=%d n_above_cap=%d dg=%.6e norm_dlogA=%.4e norm_dlogf=%.4e\n",
                    res.nStatus, wall, res.n_fc_calls, res.n_ga_calls, res.n_above_cap_reject, dg,
                    dlogA === NaN ? NaN : dlogA, dlogf === NaN ? NaN : dlogf)
                flush(stdout)
            end
        end
    end

    outfile = joinpath(OUTDIR, "melitz_phase5_d4_joint_algorithm_comparison_2026-07-28.csv")
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
