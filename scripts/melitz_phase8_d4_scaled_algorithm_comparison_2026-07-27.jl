# Governing prompt (outer-search session), Phase 8: scaled joint KNITRO comparison at D=4.
#
# Scale set chosen from this session's own Phase 7 measurements
# (docs/key_results/melitz_phase7_d4_scale_selection_2026-07-27.csv): switches/DeltaStar
# movement become non-trivial around |dg|~1e-4 (g block), aggregate norm~1e-5 (technology),
# aggregate norm~1e-5 (participation) -- so a UNIT scaled step should map to roughly that
# raw movement: s_g=1e-4, s_A=1e-5, s_f=1e-5 (matching, not coincidentally, the prior
# 2026-07-25 real-D20 session's own independently-derived candidate at a different scale/
# fixture -- same qualitative economic sensitivity ordering).
#
# D=4, W=20,000, seed=29, matrix-free/forbid_dense_fallback=true, native :linear cutoff
# constraints, sorted outer gradient backend, evaluation cap=10.0, dimensionless divergence
# constraint. Compares KNITRO's 4 named algorithms (Interior/Direct, Interior/CG, Active Set,
# SQP) at delta=1e-2 (both directions) with this ONE scale set -- per the governing prompt's
# own "first use one reasonable scale set" instruction, not a scale grid.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free)
ctx = obj.γ
n = length(theta0)
D = ctx.D
nA = D^2 - 1

r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
@assert r0.verified
println("theta0 Delta0=", r0.Delta)

# Scale vector: coordinate 1 = gamma (g block), 2:1+nA = technology (A) block, rest = participation.
s_g, s_A, s_f = 1e-4, 1e-5, 1e-5
var_scale = ones(n)
var_scale[1] = s_g
var_scale[2:1+nA] .= s_A
var_scale[2+nA:end] .= s_f
var_center = collect(Float64.(theta0))

inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
algs = [(:interior_direct, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"),
        (:interior_cg, "melitz_outer_finite_delta_alg_cg_2026-07-27.opt"),
        (:active_set, "melitz_outer_finite_delta_alg_active_2026-07-27.opt"),
        (:sqp, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt")]

rows = NamedTuple[]

for (algname, optfile) in algs
    outer_opt = joinpath(REPO, optfile)
    for direction in (:upper, :lower)
        delta = 1e-2
        println("\n=== algorithm=$algname delta=$delta direction=$direction ===")
        local res
        local wall = @elapsed begin
            res = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=delta, direction=direction,
                delta_evaluation_cap=10.0, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
                theta_box=2.0, cutoff_constraint_backend=:linear,
                inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
                var_scale=var_scale, var_center=var_center,
                backend=:matrix_free, forbid_dense_fallback=true)
        end
        cv = res.cold_verified_incumbent
        push!(rows, (algorithm=algname, delta=delta, direction=direction, wall=wall,
            nStatus=res.nStatus, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
            n_inner_solved=res.n_inner_solved, n_infinite_delta_reject=res.n_infinite_delta_reject,
            n_above_cap_reject=res.n_above_cap_reject,
            n_numerical_failure_reject=res.n_numerical_failure_reject,
            best_g=cv === nothing ? NaN : cv.eval.theta_free[1],
            best_Delta=cv === nothing ? NaN : cv.eval.Delta,
            terminal_g=res.terminal_eval.theta_free[1]))
        println(@sprintf("  nStatus=%d wall=%.2fs n_fc=%d n_ga=%d n_inner_solved=%d n_above_cap=%d best_g=%s",
            res.nStatus, wall, res.n_fc_calls, res.n_ga_calls, res.n_inner_solved, res.n_above_cap_reject,
            cv === nothing ? "NA" : @sprintf("%.6f", cv.eval.theta_free[1])))
    end
end

outfile = joinpath(OUTDIR, "melitz_phase8_d4_scaled_algorithm_comparison_2026-07-27.csv")
open(outfile, "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join([r[c] for c in cols], ","))
    end
end
println("\nWrote ", outfile)
