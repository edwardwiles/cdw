# ============================================================================
# Claude Code task 2026-08-01, follow-up: validate the O(1)-incremental
# profiled gradient (profiled_lfix_incremental_2026-08-01.jl) against the
# already-gated full-rebuild version (profiled_outer_gradient_fd_2026-08-01.jl)
# at MACHINE PRECISION, not cosine similarity against expensive re-solved
# ground truth -- both are the exact same central-FD formula of the exact
# same functional (Delta_dual, fixed dual), just computed two different ways,
# so they must agree to floating-point tolerance if the incremental
# mechanism is correct. This is the direct, cheap test of whether the
# earlier (abandoned) incremental draft's failure was really just the
# separately-diagnosed gp-formula bug.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "profiled_lfix_incremental_2026-08-01.jl"))
using LinearAlgebra, Printf, CSV, DataFrames

const IS_D20 = "D20" in ARGS

if IS_D20
    include(joinpath(@__DIR__, "context_real_d20.jl"))
    println("Building real D=20 :exclude_row context at W=80000 ..."); flush(stdout)
    ctx = d20_real_setup(W = 80_000, destination_sample = :exclude_row)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
    tag = "D20_W80000"
else
    ctx0 = d4_exact_setup()
    ctx = build_unrestricted_operator_ctx(ctx0)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
    tag = "D4"
end
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("tag=$tag  n_total=$n_total"); flush(stdout)

function compare_at(label::String, w::Vector{Float64}, rows)
    println("\n" * "="^90); println("POINT: $label"); println("="^90); flush(stdout)
    ev = evaluate_profiled_point(w, ctx, spec, pe)
    println("base Delta_dual=$(ev.result.Delta_dual)  inner_status=$(ev.result.inner_status)"); flush(stdout)

    t1 = time()
    g_full, _ = profiled_composite_gradient_at(w, ctx, spec, pe, ev)
    t_full = time() - t1
    println("full-rebuild gradient computed in $(t_full)s"); flush(stdout)

    t2 = time()
    g_inc, _ = profiled_composite_gradient_at_incremental(w, ctx, spec, pe, ev)
    t_inc = time() - t2
    println("incremental gradient computed in $(t_inc)s (speedup=$(round(t_full/t_inc,digits=1))x)"); flush(stdout)

    diff = g_full .- g_inc
    max_abs_err = maximum(abs.(diff))
    max_rel_err = maximum(abs.(diff) ./ max.(abs.(g_full), 1e-8))
    cos_sim = dot(g_full, g_inc) / (norm(g_full) * norm(g_inc) + 1e-300)
    worst_k = argmax(abs.(diff))
    println(@sprintf("max_abs_err=%.4e  max_rel_err=%.4e  cos_sim=%.10f  worst_k=%d (full=%.6e inc=%.6e)",
        max_abs_err, max_rel_err, cos_sim, worst_k, g_full[worst_k], g_inc[worst_k]))
    println(@sprintf("gp: full=%.10e  inc=%.10e  diff=%.3e", g_full[1], g_inc[1], abs(g_full[1]-g_inc[1])))
    flush(stdout)

    push!(rows, (label = label, t_full = t_full, t_inc = t_inc, speedup = t_full / t_inc,
        max_abs_err = max_abs_err, max_rel_err = max_rel_err, cos_sim = cos_sim,
        gp_full = g_full[1], gp_inc = g_inc[1]))
    return rows
end

rows = []
compare_at("calibration", w_calib, rows)

using Random
Random.seed!(42)
n_free = n_total - 1
w_pert = copy(w_calib); w_pert[2:end] .+= 0.01 .* randn(n_free)
compare_at("small_perturbation", w_pert, rows)

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_INCREMENTAL_VS_FULLREBUILD_2026-08-01_$tag.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)

all_pass = all(r.max_rel_err < 1e-6 for r in rows)
println("\nINCREMENTAL VS FULL-REBUILD (machine precision) ($tag): ", all_pass ? "PASS" : "NEEDS REVIEW")
