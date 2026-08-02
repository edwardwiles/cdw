# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §11
# "Unrestricted regression": require the refactored shared engine
# (profiled_shared_economic_gradient_engine_2026-08-01.jl, via
# UnrestrictedFamilyCtx) to reproduce the pre-refactor
# `profiled_composite_gradient_at_incremental` gradient EXACTLY -- same
# bandwidths, same winner-update counts, same objective probes, machine
# precision. Real D4 KNITRO solve (no mocks).
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
include(joinpath(@__DIR__, "profiled_outer_gradient_layout_contract_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_shared_economic_gradient_engine_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_family_adapters_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl"))
using LinearAlgebra, Printf, CSV, DataFrames, Random

const IS_D20 = "D20" in ARGS

if IS_D20
    include(joinpath(@__DIR__, "context_real_d20.jl"))
    D20_W = something(tryparse(Int, get(ENV, "D20_W", "")), 20_000)
    println("Building real D=20 :exclude_row context at W=$D20_W ..."); flush(stdout)
    ctx = d20_real_setup(W = D20_W, destination_sample = :exclude_row)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
    tag = "D20_W$(D20_W)"
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
    g_pre, meta_pre = profiled_composite_gradient_at_incremental(w, ctx, spec, pe, ev)
    t_pre = time() - t1

    ufctx = build_unrestricted_family_ctx(ctx, spec, pe, ev)
    t2 = time()
    g_shared, meta_shared = shared_family_outer_gradient(w, ctx, ufctx, ev)
    t_shared = time() - t2

    diff = g_pre .- g_shared
    max_abs_err = maximum(abs.(diff))
    max_rel_err = maximum(abs.(diff) ./ max.(abs.(g_pre), 1e-8))
    cos_sim = dot(g_pre, g_shared) / (norm(g_pre) * norm(g_shared) + 1e-300)
    h_match = meta_pre.h_used == meta_shared.h_used
    switch_mass_match = meta_pre.switch_mass == meta_shared.switch_mass
    q0_match = meta_pre.cache.q0 == meta_shared.cache.q0
    println(@sprintf("pre-refactor: %.5fs   shared-engine: %.5fs", t_pre, t_shared))
    println(@sprintf("max_abs_err=%.4e  max_rel_err=%.4e  cos_sim=%.15f", max_abs_err, max_rel_err, cos_sim))
    println("h_used bit-identical: $h_match   switch_mass bit-identical: $switch_mass_match   cache.q0 bit-identical: $q0_match")
    flush(stdout)

    push!(rows, (label = label, t_pre = t_pre, t_shared = t_shared,
        max_abs_err = max_abs_err, max_rel_err = max_rel_err, cos_sim = cos_sim,
        h_used_bit_identical = h_match, switch_mass_bit_identical = switch_mass_match,
        q0_bit_identical = q0_match))
    return rows
end

rows = []
compare_at("calibration", w_calib, rows)

Random.seed!(42)
n_free = n_total - 1
w_pert = copy(w_calib); w_pert[2:end] .+= 0.01 .* randn(n_free)
compare_at("small_perturbation", w_pert, rows)

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_UNRESTRICTED_SHARED_ENGINE_REGRESSION_2026-08-01_$tag.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)

all_pass = all(r.max_abs_err == 0.0 for r in rows)
println("\nUNRESTRICTED REGRESSION ($tag): ", all_pass ? "PASS (bit-identical)" : "NEEDS REVIEW")
all_pass || error("unrestricted regression: shared engine diverged from pre-refactor gradient")
