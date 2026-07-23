# Step 9.1: checkpoint/supervisor integration gates for the CM+moments(+ZC) extension, tested at
# D=4 speed via a lightweight ctx_builder adapter (run_cm_meanzc_upper_checkpointed itself is
# generic over ctx_builder -- production callers pass d20_real_setup_design; this test passes a
# D=4 adapter so the SCHEMA/resume/mismatch MECHANICS are covered without needing a real D=20
# solve for every gate). Covers: clean completion, wall-budget exhaustion + resume, schema
# mismatch refusal, context-fingerprint mismatch refusal, cold verification of the stored best
# vector.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_meanzc_checkpoint.jl"))
using Test, Printf, Random, Serialization

"D=4 ctx_builder adapter: run_cm_meanzc_upper_checkpointed is generic over this signature; the D=4 test context is a fixed dataset so W/draw_design/draw_seed are accepted and ignored."
ctx_builder_d4(; W, δ, find_smallest, draw_design, draw_seed) = d4_exact_setup(δ = δ, find_smallest = find_smallest, needs_outer_moment_jacobian = false)

ctx0 = ctx_builder_d4(; W = 0, δ = 1.0, find_smallest = true, draw_design = :na, draw_seed = 0)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0 = vcat(x_free_calib[1], pivot_reduce(z0, pe0))
w0_ext = vcat(w0, log(1.0))   # K_mean=1

ckpt_dir = mktempdir(prefix = "cm_meanzc_ckpt_test_")
println(">>> ckpt_dir = ", ckpt_dir)

println("="^100)
println("9.1a: fresh run -> checkpoint written -> load back -> fields match")
println("="^100)
@testset "fresh run + checkpoint round trip" begin
    res1 = run_cm_meanzc_upper_checkpointed(ctx_builder_d4, w0_ext;
        W = 1, delta = 1.0, draw_design = :na, draw_seed = 0,
        L = 10, contrasts = :anchored, probs = cm_equal_grid_probs(10),
        K_mean = 1, K_pair = 1, meanzc_basis = :direct,
        maxtime_real = 6.0, ckpt_dir = ckpt_dir, run_id = "test_run_1", label = "stageA",
        checkpoint_interval_s = 0.5, verbose = false)

    @test res1.n_eval > 0
    @test isfile(res1.ckpt_path)

    ckpt = load_cm_meanzc_checkpoint(res1.ckpt_path)
    @test ckpt.schema == CM_MEANZC_CHECKPOINT_SCHEMA
    @test ckpt.K_mean == 1 && ckpt.K_pair == 1
    @test ckpt.meanzc_basis == :direct
    @test length(ckpt.eta_nu) == 1
    @test ckpt.run_id == "test_run_1"
    @test ckpt.n_eval == res1.n_eval
    @printf "  n_eval=%d  checkpoint_reason=%s  best=%s\n" res1.n_eval string(ckpt.checkpoint_reason) (res1.best === nothing ? "nothing" : "gp=$(res1.best.gp) Delta=$(res1.best.Delta)")
end
println()

println("="^100)
println("9.1b: resume from a checkpoint continues (does not restart n_eval from zero)")
println("="^100)
@testset "resume continues n_eval/wall_elapsed" begin
    res1 = run_cm_meanzc_upper_checkpointed(ctx_builder_d4, w0_ext;
        W = 1, delta = 1.0, draw_design = :na, draw_seed = 0,
        L = 10, contrasts = :anchored, probs = cm_equal_grid_probs(10),
        K_mean = 1, K_pair = 0, meanzc_basis = :direct,
        maxtime_real = 4.0, ckpt_dir = ckpt_dir, run_id = "test_run_2", label = "stageB",
        checkpoint_interval_s = 0.3, verbose = false)
    ckpt1 = load_cm_meanzc_checkpoint(res1.ckpt_path)
    @test ckpt1.n_eval > 0

    res2 = run_cm_meanzc_upper_checkpointed(ctx_builder_d4, nothing;
        L = 10, maxtime_real = 4.0, ckpt_dir = ckpt_dir, label = "stageB",
        checkpoint_interval_s = 0.3, resume_from = res1.ckpt_path, verbose = false)
    @test res2.n_eval >= ckpt1.n_eval    # resumed counter picks up where it left off, never resets
    @printf "  pre-resume n_eval=%d  post-resume n_eval=%d\n" ckpt1.n_eval res2.n_eval
end
println()

println("="^100)
println("9.1c: schema mismatch is refused, never silently accepted")
println("="^100)
@testset "schema mismatch refusal" begin
    good_path = joinpath(ckpt_dir, "stageA_latest.jls")
    good = deserialize(good_path)::CMMeanZCCheckpoint
    bad = CMMeanZCCheckpoint(99, good.run_id, good.label, good.branch, good.find_smallest, good.delta,
        good.W, good.draw_seed, good.draw_design, good.ctx_fingerprint,
        good.cm_L, good.cm_probs, good.cm_contrasts, good.cm_grid_rule, good.cm_basis, good.cm_hessian_backend,
        good.K_mean, good.K_pair, good.meanzc_basis, good.nu_bounds,
        good.g, good.zfree, good.eta_nu, good.logA_full, good.dual_warm_start, good.bandwidth_cache,
        good.best_feasible, good.n_eval, good.n_grad, good.wall_elapsed, good.wall_budget_remaining,
        good.checkpoint_reason, good.knitro_version)
    bad_path = joinpath(ckpt_dir, "schema_mismatch.jls")
    save_cm_meanzc_checkpoint(bad_path, bad)
    @test_throws ErrorException load_cm_meanzc_checkpoint(bad_path)
    @test_throws ErrorException run_cm_meanzc_upper_checkpointed(ctx_builder_d4, nothing;
        L = 10, maxtime_real = 2.0, ckpt_dir = ckpt_dir, label = "stageA", resume_from = bad_path, verbose = false)
end
println()

println("="^100)
println("9.1d: context-fingerprint mismatch is refused, never silently accepted")
println("="^100)
@testset "context fingerprint mismatch refusal" begin
    good_path = joinpath(ckpt_dir, "stageA_latest.jls")
    good = deserialize(good_path)::CMMeanZCCheckpoint
    bad = CMMeanZCCheckpoint(good.schema, good.run_id, good.label, good.branch, good.find_smallest, good.delta,
        good.W, good.draw_seed, good.draw_design, "deliberately-wrong-fingerprint",
        good.cm_L, good.cm_probs, good.cm_contrasts, good.cm_grid_rule, good.cm_basis, good.cm_hessian_backend,
        good.K_mean, good.K_pair, good.meanzc_basis, good.nu_bounds,
        good.g, good.zfree, good.eta_nu, good.logA_full, good.dual_warm_start, good.bandwidth_cache,
        good.best_feasible, good.n_eval, good.n_grad, good.wall_elapsed, good.wall_budget_remaining,
        good.checkpoint_reason, good.knitro_version)
    bad_path = joinpath(ckpt_dir, "fingerprint_mismatch.jls")
    save_cm_meanzc_checkpoint(bad_path, bad)
    @test_throws ErrorException run_cm_meanzc_upper_checkpointed(ctx_builder_d4, nothing;
        L = 10, maxtime_real = 2.0, ckpt_dir = ckpt_dir, label = "stageA", resume_from = bad_path, verbose = false)
end
println()

println("="^100)
println("9.1e: cold verification of the exact stored best vector")
println("="^100)
@testset "cold-verify stored best_feasible.w reproduces stored Delta" begin
    good_path = joinpath(ckpt_dir, "stageA_latest.jls")
    ckpt = load_cm_meanzc_checkpoint(good_path)
    ckpt.best_feasible === nothing && error("test setup: stageA found no verified-feasible incumbent to cold-verify -- widen maxtime_real")

    ctx_cold = ctx_builder_d4(; W = 1, δ = ckpt.delta, find_smallest = ckpt.find_smallest, draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed)
    @test context_fingerprint(ctx_cold) == ckpt.ctx_fingerprint
    pe_cold = build_pivot_elimination(ctx_cold)
    pcx_cold = build_cm_meanzc_production_context(ctx_cold, CS; L = ckpt.cm_L, K_mean = ckpt.K_mean, K_pair = ckpt.K_pair,
        contrasts = ckpt.cm_contrasts, meanzc_basis = ckpt.meanzc_basis, probs = ckpt.cm_probs)

    w = ckpt.best_feasible.w
    xf = x_free_from_w_meanzc(w, pe_cold, ckpt.K_mean)
    νvec = nu_from_w_meanzc(w, ckpt.K_mean)
    _, _, verify_cold = cm_meanzc_production_value_verified(xf, νvec, pcx_cold)
    @printf "  stored Delta=%.10f  cold-verified Delta_dual=%.10f  |diff|=%.3e\n" ckpt.best_feasible.Delta verify_cold.Delta_dual abs(ckpt.best_feasible.Delta - verify_cold.Delta_dual)
    @test verify_cold.Delta_dual ≈ ckpt.best_feasible.Delta atol = 1e-9
    @test is_verified_success(verify_cold)
end

println()
println("All checkpoint/supervisor gate tests passed.")
rm(ckpt_dir; recursive = true, force = true)
