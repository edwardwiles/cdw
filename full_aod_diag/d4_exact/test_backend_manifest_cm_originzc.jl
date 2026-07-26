# ============================================================================
# Public entry-point backend manifest assertions, CM / CM+mean-ZC / origin-ZC families
# (allocation/Hessian port task §3). Companion: test_backend_manifest_unrestricted.jl.
#
# "Prove that the public driver actually reaches the intended implementation" -- calls the REAL
# public checkpointed drivers (run_cm_upper_checkpointed x2 for :cm_only/:cm_plus_equal_means,
# run_originzc_upper_checkpointed) with a short maxtime_real budget and asserts on the actual
# "[backend-manifest] ..." lines their own stdout produces, not on resolve_*_manifest called in
# isolation.
#
# Include list is EXACTLY d20_originzc_shakedown.jl's own (the one proven-working script in this
# tree that already loads the full CM + origin-ZC chain together) -- deliberately not layered on
# top of c10_d20_production_driver.jl's own partial overlapping chain, to avoid duplicate
# top-level struct/const definitions across the two chains.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/test_backend_manifest_cm_originzc.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_checkpoint.jl", "cm_originzc_target_layout.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl"]
    include(joinpath(_D4E, f))
end
using Dates, Serialization, LinearAlgebra, Statistics

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

"Runs `f()`, capturing everything it prints to stdout, and returns (result, captured_text)."
function capture_stdout(f)
    old = stdout
    rd, wr = redirect_stdout()
    result = try
        f()
    finally
        redirect_stdout(old)
        close(wr)
    end
    text = read(rd, String)
    close(rd)
    print(text)   # still show it in this test's own log
    return result, text
end

ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)
w_calib = vcat(g0, zfree0)

OUTDIR = mktempdir()

println("="^78); println("1. run_cm_upper_checkpointed (flexible CM, :cm_only)"); println("="^78)
snaps = nested_grid_sequence([10, 20, 50]); probs10 = snaps[10]
_, txt = capture_stdout() do
    run_cm_upper_checkpointed(w_calib; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
        draw_seed = 20260719, L = 10, contrasts = :orthonormal, probs = probs10,
        cm_hessian_backend = :structured, maxtime_real = 5.0, ckpt_dir = joinpath(OUTDIR, "cm"),
        label = "assert_cm", checkpoint_interval_s = 9999.0, destination_sample = :exclude_row)
end
check("prints [backend-manifest] family=flexible_cm", occursin("[backend-manifest] family=flexible_cm", txt))
# 2026-07-25 continuation: hessian_backend now carries the "_with_winner_pair_core" suffix (this
# session found + fixed the bug that silently kept core_hessian_backend reporting
# dense_reference_fallback_this_point at startup -- see production_backend_manifest.jl's
# resolve_flexible_cm_manifest docstring). core_hessian_backend is the field that actually matters.
check("cm.hessian_backend=threaded_architecture_c_with_winner_pair_core (production default)", occursin("hessian_backend=threaded_architecture_c_with_winner_pair_core", txt))
check("cm.core_hessian_backend=exact_winner_pair_parallel", occursin("core_hessian_backend=exact_winner_pair_parallel", txt))
# final-gate continuation 2026-07-25 (task §3): dynamic policy, not a hard-coded 10 -- see
# test_backend_manifest_unrestricted.jl's identical fix for the rationale.
check("cm.core_hessian_workers=$(resolve_core_hessian_workers_default())",
      occursin("core_hessian_workers=$(resolve_core_hessian_workers_default())", txt))
check("cm.threaded_bins=true", occursin("threaded_bins=true", txt))
check("cm.checkpoint_schema=6", occursin("checkpoint_schema=6", txt))
check("cm.cm_restriction_basis=cumulative", occursin("cm_restriction_basis=cumulative", txt))
check("cm.core_moment_representation=compressed_winner_form", occursin("core_moment_representation=compressed_winner_form", txt))

println("="^78); println("2. run_cm_upper_checkpointed (CM+mean/ZC, :cm_plus_equal_means, K_mean=1)"); println("="^78)
w_calib_meanzc = vcat(w_calib, 0.0)   # append eta_nu for K_mean=1
_, txt = capture_stdout() do
    run_cm_upper_checkpointed(w_calib_meanzc; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
        draw_seed = 20260719, L = 10, contrasts = :orthonormal, probs = probs10,
        cm_hessian_backend = :structured, maxtime_real = 5.0, ckpt_dir = joinpath(OUTDIR, "cm_meanzc"),
        label = "assert_cm_meanzc", checkpoint_interval_s = 9999.0, destination_sample = :exclude_row,
        cm_extension = :cm_plus_equal_means, meanzc_K_mean = 1)
end
check("prints [backend-manifest] family=cm_meanzc", occursin("[backend-manifest] family=cm_meanzc", txt))
check("cm_meanzc.hessian_backend=threaded_architecture_c_with_winner_pair_core", occursin("hessian_backend=threaded_architecture_c_with_winner_pair_core", txt))
check("cm_meanzc.core_hessian_backend=exact_winner_pair_parallel", occursin("core_hessian_backend=exact_winner_pair_parallel", txt))
check("cm_meanzc.K_mean=1", occursin("K_mean=1", txt))

println("="^78); println("3. run_originzc_upper_checkpointed (origin-specific ZC, K_mean=1)"); println("="^78)
# Real feasible eta0 = log(mean(U^k)) per origin -- see d20_originzc_shakedown.jl's identical
# construction; eta0=0.0 (nu0=1.0) is NOT a feasible origin-specific-power start point.
layout1 = OriginByPowerLayout(ctx0.D, 1, 1)
nu0_1 = Vector{Float64}(undef, n_eta(layout1))
for o in 1:ctx0.D
    nu0_1[target_index(layout1, o, 1)] = mean(@view ctx0.U[:, o])
end
w_calib_originzc = vcat(w_calib, log.(nu0_1))
_, txt = capture_stdout() do
    run_originzc_upper_checkpointed(w_calib_originzc; W = 80_000, delta = 1.0,
        draw_design = :pseudorandom, draw_seed = 20260719, maxtime_real = 5.0,
        ckpt_dir = joinpath(OUTDIR, "originzc"), label = "assert_originzc",
        checkpoint_interval_s = 9999.0, distribution_restriction = :origin_specific_moments,
        K_mean = 1, destination_sample = :exclude_row)
end
check("prints [backend-manifest] family=origin_zc", occursin("[backend-manifest] family=origin_zc", txt))
check("origin_zc.hessian_backend=partitioned_winner_pair_core_dense_restriction", occursin("hessian_backend=partitioned_winner_pair_core_dense_restriction", txt))
check("origin_zc.core_hessian_backend=exact_winner_pair_parallel", occursin("core_hessian_backend=exact_winner_pair_parallel", txt))
check("origin_zc.cross_hessian_backend=dense_exact (H_ER retained dense)", occursin("cross_hessian_backend=dense_exact", txt))
check("origin_zc.checkpoint_schema=7", occursin("checkpoint_schema=7", txt))

println("="^78)
println(isempty(FAILURES) ? "ALL CM/CM+MEANZC/ORIGIN-ZC BACKEND ASSERTIONS PASSED" :
        "FAILURES ($(length(FAILURES))): " * join(FAILURES, "; "))
println("="^78)
isempty(FAILURES) || error("test_backend_manifest_cm_originzc.jl: $(length(FAILURES)) assertion(s) failed")
