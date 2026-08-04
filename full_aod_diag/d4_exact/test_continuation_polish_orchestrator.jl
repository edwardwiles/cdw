# test_continuation_polish_orchestrator.jl -- pure-logic tests for continuation_polish_orchestrator.jl.
#
# Deliberately does NOT include the KNITRO driver chain -- every function under test here operates
# on plain NamedTuples/Vectors/CSV text, injected via a mock `run_fn`, so these tests run in
# seconds without a real inner/outer solve. `load_checkpoint_w` (the one function that needs
# `load_checkpoint_unified`/`load_cm_checkpoint` from the real driver chain) is verified separately
# against real files in CHECKPOINT_CONTENTS_VERIFICATION_2026-08-03.md, not re-tested here.
#
# Run: julia --project=<repo root> full_aod_diag/d4_exact/test_continuation_polish_orchestrator.jl

using Test

include(joinpath(@__DIR__, "continuation_polish_orchestrator.jl"))

@testset "better / strictly_better" begin
    @test better(:upper, 0.08, 0.07)
    @test better(:upper, 0.07, 0.07)
    @test !better(:upper, 0.06, 0.07)
    @test better(:lower, 0.06, 0.07)
    @test !better(:lower, 0.08, 0.07)
    @test strictly_better(:upper, 0.08, 0.07)
    @test !strictly_better(:upper, 0.07, 0.07)
    @test strictly_better(:lower, 0.06, 0.07)
end

@testset "MonotoneEnvelope: register! and envelope_at (upper/lower comparison + inheritance)" begin
    env = MonotoneEnvelope()
    prov(x) = (source_delta = x, source_start = 1, outer_vector_path = "p$x", outer_vector_sha256 = "h$x")
    wv(x) = [x]

    # Upper: larger GT is better. Register out of order, confirm max-so-far semantics.
    register!(env, "fam", :upper, 0.5, 0.06, wv(0.5), 0.5, prov(0.5))
    register!(env, "fam", :upper, 1.0, 0.07, wv(1.0), 1.0, prov(1.0))
    register!(env, "fam", :upper, 2.0, 0.05, wv(2.0), 2.0, prov(2.0))  # worse than delta=1.0's own point

    # Envelope AT delta=2.0 must inherit delta=1.0's better point (0.07), not delta=2.0's own (0.05).
    e2 = envelope_at(env, "fam", :upper, 2.0)
    @test e2.GT == 0.07
    @test e2.source_delta == 1.0

    # Envelope at delta=0.5 only sees delta<=0.5.
    e05 = envelope_at(env, "fam", :upper, 0.5)
    @test e05.GT == 0.06

    # No registrations yet for this delta/family/direction combo -> nothing.
    @test envelope_at(env, "fam", :upper, 0.01) === nothing
    @test envelope_at(env, "other_fam", :upper, 2.0) === nothing

    # Lower: smaller GT is better. A worse (larger) GT at a later delta must not overwrite the
    # earlier better one when re-registering at the SAME delta.
    register!(env, "fam", :lower, 1.0, 0.02, wv(1.0), 1.0, prov(1.0))
    register!(env, "fam", :lower, 1.0, 0.05, wv(1.0), 1.0, prov(1.0))  # worse, same delta -- must not overwrite
    @test envelope_at(env, "fam", :lower, 1.0).GT == 0.02
    register!(env, "fam", :lower, 1.0, 0.01, wv(1.0), 1.0, prov(1.0))  # better, same delta -- must overwrite
    @test envelope_at(env, "fam", :lower, 1.0).GT == 0.01
end

@testset "dedup_seeds: distance + objective dedup, first occurrence kept" begin
    mkseed(role, w, gt) = Seed("fam", :upper, role, 1.0, 1, gt, 1.0, w, "hash_$role", "path_$role")
    w1 = [1.0, 2.0, 3.0]
    w1_near = w1 .+ 1e-12          # effectively identical point, effectively identical GT
    w2 = [10.0, 20.0, 30.0]        # genuinely different point

    seeds = [mkseed("A_envelope", w1, 0.07), mkseed("B_near_dup_of_A", w1_near, 0.07 + 1e-12),
             mkseed("C_distinct", w2, 0.05)]
    kept = dedup_seeds(seeds)
    @test length(kept) == 2
    @test kept[1].role == "A_envelope"   # first occurrence wins, not privileged by any other rule
    @test kept[2].role == "C_distinct"

    # A seed at the same point but a MEANINGFULLY different GT (e.g. re-verified) is NOT a duplicate.
    seeds2 = [mkseed("A", w1, 0.07), mkseed("D_same_point_diff_gt", w1, 0.03)]
    @test length(dedup_seeds(seeds2)) == 2
end

@testset "algo_kwargs: family-specific algorithm wiring, explicit error not silent fallback" begin
    @test algo_kwargs("unrestricted", EXPLORE_DIRECT_SR1) == (outer_direct_hessopt = :sr1,)
    @test algo_kwargs("flexible_cm", EXPLORE_DIRECT_SR1) == (outer_direct_hessopt = :sr1,)

    @test algo_kwargs("flexible_cm", POLISH_SQP) == (opt_file = "csw_outer_phaseB_sqp_bfgs_maxit15.opt",)
    @test algo_kwargs("origin_zc", POLISH_SQP) == (opt_file = "csw_outer_phaseB_sqp_bfgs_maxit15.opt",)

    # unrestricted has no opt_file kwarg on its driver -- POLISH_SQP must error, not silently
    # substitute something else (matches this repo's own "no silent default substitution" rule).
    @test_throws ErrorException algo_kwargs("unrestricted", POLISH_SQP)
    @test algo_kwargs("unrestricted", POLISH_DIRECT_BFGS) == (outer_direct_hessopt = :bfgs,)
    @test algo_kwargs("flexible_cm", POLISH_DIRECT_BFGS) == (opt_file = "csw_outer_phaseB_direct_bfgs_maxit15.opt",)
end

@testset "normalize_result: field-name normalization across the two driver return shapes" begin
    unified_raw = (kappa = 0.075, best_feasible = (w = [1.0, 2.0], Delta = 0.9999), knitro_status = 0,
                   wall_ext = 123.4, n_eval = 50, final_checkpoint = "/tmp/ckpt.jls")
    r1 = normalize_result("unrestricted", unified_raw)
    @test r1.GT == 0.075 && r1.w == [1.0, 2.0] && r1.Delta_star == 0.9999 && r1.wall_s == 123.4

    cm_raw = (kappa = 0.065, best = (w = [3.0, 4.0], Delta = 0.5001), knitro_status = -101,
              wall = 456.7, n_eval = 30, ckpt_path = "/tmp/ckpt2.jls")
    r2 = normalize_result("flexible_cm", cm_raw)
    @test r2.GT == 0.065 && r2.w == [3.0, 4.0] && r2.knitro_status == -101

    # No verified point found (best_feasible/best === nothing) -> w=nothing, GT=NaN, not a crash.
    unified_none = (kappa = NaN, best_feasible = nothing, knitro_status = -411, wall_ext = 999.0,
                    n_eval = 10, final_checkpoint = "")
    r3 = normalize_result("unrestricted", unified_none)
    @test r3.w === nothing && isnan(r3.GT)
end

@testset "apply_never_regress: the core monotonicity guarantee (Section 4/12 of the task spec)" begin
    inherited_upper = (GT = 0.07, w = [1.0, 1.0], Delta_star = 1.0)
    better_cand = CellResult(0.08, [2.0, 2.0], 1.99, 0, 100.0, 5, "/tmp/a.jls")
    worse_cand  = CellResult(0.05, [3.0, 3.0], 1.5, -401, 7200.0, 3, "/tmp/b.jls")
    no_point    = CellResult(NaN, nothing, NaN, -411, 7200.0, 8, "/tmp/c.jls")

    gt, w, d, src = apply_never_regress(:upper, inherited_upper, better_cand)
    @test (gt, w, d, src) == (0.08, [2.0, 2.0], 1.99, "solved")

    # Candidate solved a real point but it's WORSE than the inherited incumbent -- must export the
    # inherited point instead, with explicit provenance, per the task spec's exact wording.
    gt2, w2, d2, src2 = apply_never_regress(:upper, inherited_upper, worse_cand)
    @test (gt2, w2, d2, src2) == (0.07, [1.0, 1.0], 1.0, "inherited_incumbent")

    # Solver found nothing at all -- must still export the inherited point, not crash/NaN out.
    gt3, w3, d3, src3 = apply_never_regress(:upper, inherited_upper, no_point)
    @test (gt3, w3, d3, src3) == (0.07, [1.0, 1.0], 1.0, "inherited_incumbent")

    # Lower direction: smaller GT is better -- confirm the comparison direction actually flips.
    inherited_lower = (GT = 0.02, w = [1.0], Delta_star = 1.0)
    better_lower_cand = CellResult(0.01, [2.0], 1.0, 0, 50.0, 4, "/tmp/d.jls")  # smaller = better
    gt4, _, _, src4 = apply_never_regress(:lower, inherited_lower, better_lower_cand)
    @test gt4 == 0.01 && src4 == "solved"
    worse_lower_cand = CellResult(0.03, [3.0], 1.0, 0, 50.0, 4, "/tmp/e.jls")  # larger = worse
    gt5, _, _, src5 = apply_never_regress(:lower, inherited_lower, worse_lower_cand)
    @test gt5 == 0.02 && src5 == "inherited_incumbent"

    # No inherited incumbent AND no verified candidate -> must error loudly, not fabricate a result.
    @test_throws ErrorException apply_never_regress(:upper, nothing, no_point)
    # No inherited incumbent but a real candidate exists -> use the candidate.
    gt6, w6, _, src6 = apply_never_regress(:upper, nothing, better_cand)
    @test gt6 == 0.08 && src6 == "solved"
end

@testset "assert_manifest_compatible: field-by-field mismatch rejection" begin
    target = (W = 100000, sigma = 3.0, draw_seed = 20260719, destination_sample = :exclude_row)
    good = (W = 100000, sigma = 3.0, draw_seed = 20260719, destination_sample = :exclude_row, extra_field_ok = true)
    @test assert_manifest_compatible(target, good) === true

    bad_sigma = (W = 100000, sigma = 2.5, draw_seed = 20260719, destination_sample = :exclude_row)
    e = try assert_manifest_compatible(target, bad_sigma); nothing catch e; e end
    @test e isa ManifestMismatchError
    @test e.field == "sigma" && e.expected == 3.0 && e.actual == 2.5

    # Mismatch must be reported for the FIRST differing field encountered, not silently swallowed.
    bad_W = (W = 80000, sigma = 3.0, draw_seed = 20260719, destination_sample = :exclude_row)
    @test_throws ManifestMismatchError assert_manifest_compatible(target, bad_W)
end

@testset "OrchestratorRunState checkpoint round trip" begin
    mktempdir() do dir
        report = FinalCellReport("flexible_cm", :upper, 2.0, "POLISH_SQP", "A_envelope_incumbent_delta_1.0",
                                  0.0724, 0.0730, 0.0730, [1.0, 2.0, 3.0], 1.9998, "solved", 0, 4321.0, 88,
                                  joinpath(dir, "ckpt.jls"))
        state = OrchestratorRunState("flexible_cm", :upper, 2.0, POLISH_SQP, "A_envelope_incumbent_delta_1.0",
                                      "2026-08-03T22:00:00", report)
        path = joinpath(dir, "run_state.jls")
        save_run_state(path, state)
        @test isfile(path)
        loaded = load_run_state(path)
        @test loaded.family == "flexible_cm" && loaded.direction == :upper && loaded.target_delta == 2.0
        @test loaded.stage == POLISH_SQP
        @test loaded.report.final_GT == 0.0730
        @test loaded.report.final_w == [1.0, 2.0, 3.0]
        @test loaded.report.result_source == "solved"

        # Round trip through a genuinely no-op "no candidate improved" state too.
        report2 = FinalCellReport("origin_zc", :lower, 1.0, "EXPLORE_DIRECT_SR1", "inherited_only",
                                   0.02, nothing, 0.02, [9.0], 0.5, "inherited_incumbent", -411, 7200.0, 200,
                                   joinpath(dir, "ckpt2.jls"))
        state2 = OrchestratorRunState("origin_zc", :lower, 1.0, EXPLORE_DIRECT_SR1, "inherited_only",
                                       "2026-08-03T22:10:00", report2)
        path2 = joinpath(dir, "run_state2.jls")
        save_run_state(path2, state2)
        loaded2 = load_run_state(path2)
        @test loaded2.report.new_GT === nothing
        @test loaded2.report.result_source == "inherited_incumbent"
    end
end

@testset "run_target_cell!: end-to-end orchestration with a mock run_fn (algorithm-stage transition)" begin
    # Mock run_fn records which (w0, stage) it was called with, and returns a deterministic,
    # improving-then-plateauing sequence so we can confirm explore->polish transition + never-regress.
    calls = NamedTuple[]
    function mock_run_fn(family, w0, stage, budget_s, ckpt_dir; find_smallest)
        push!(calls, (family = family, w0 = copy(w0), stage = stage, budget_s = budget_s, find_smallest = find_smallest))
        if stage == EXPLORE_DIRECT_SR1
            # Exploration improves on the seed.
            return (kappa = 0.078, best_feasible = (w = w0 .+ 1.0, Delta = 1.999), knitro_status = 0,
                    wall_ext = budget_s, n_eval = 20, final_checkpoint = joinpath(ckpt_dir, "explore.jls"))
        else
            # Polish finds a slightly better point still.
            return (kappa = 0.080, best_feasible = (w = w0 .+ 0.1, Delta = 1.9995), knitro_status = 0,
                    wall_ext = budget_s, n_eval = 5, final_checkpoint = joinpath(ckpt_dir, "polish.jls"))
        end
    end

    env = MonotoneEnvelope()
    register!(env, "unrestricted", :upper, 1.0, 0.073, [1.0, 1.0, 1.0], 1.0,
              (source_delta = 1.0, source_start = 1, outer_vector_path = "p", outer_vector_sha256 = "h"))

    seed = Seed("unrestricted", :upper, "A_envelope_incumbent_delta_1.0", 1.0, 1, 0.073, 1.0,
                [1.0, 1.0, 1.0], "hash", "path")

    mktempdir() do dir
        report = run_target_cell!(env, "unrestricted", :upper, 2.0, [seed],
                                   EXPLORE_DIRECT_SR1, POLISH_DIRECT_BFGS, mock_run_fn, dir;
                                   explore_budget_s = 1800.0, polish_budget_s = 900.0)

        # Both stages were actually invoked, in the right order, with the right budgets.
        @test length(calls) == 2
        @test calls[1].stage == EXPLORE_DIRECT_SR1 && calls[1].budget_s == 1800.0
        @test calls[1].w0 == [1.0, 1.0, 1.0]           # explore starts from the seed
        @test calls[2].stage == POLISH_DIRECT_BFGS && calls[2].budget_s == 900.0
        @test calls[2].w0 == [2.0, 2.0, 2.0]           # polish starts from explore's own output

        # Final result is polish's (best) point, correctly registered, beats the δ=1.0 inherited incumbent.
        @test report.result_source == "solved"
        @test report.final_GT == 0.080
        @test report.final_w == [2.1, 2.1, 2.1]
        @test report.inherited_GT == 0.073

        # And the envelope now reflects the improvement at δ=2.0 for future inheritance.
        @test envelope_at(env, "unrestricted", :upper, 2.0).GT == 0.080
        # ...while δ=1.0's own entry is untouched.
        @test envelope_at(env, "unrestricted", :upper, 1.0).GT == 0.073
    end

    # Second scenario: mock run_fn that NEVER improves -- confirm the never-regress rule actually
    # exports the inherited incumbent, not a worse solved point.
    function mock_run_fn_worse(family, w0, stage, budget_s, ckpt_dir; find_smallest)
        return (kappa = 0.01, best_feasible = (w = w0, Delta = 1.9), knitro_status = -401,
                wall_ext = budget_s, n_eval = 1, final_checkpoint = joinpath(ckpt_dir, "x.jls"))
    end
    env2 = MonotoneEnvelope()
    register!(env2, "unrestricted", :upper, 1.0, 0.073, [1.0, 1.0, 1.0], 1.0,
              (source_delta = 1.0, source_start = 1, outer_vector_path = "p", outer_vector_sha256 = "h"))
    mktempdir() do dir
        report2 = run_target_cell!(env2, "unrestricted", :upper, 2.0, [seed],
                                    EXPLORE_DIRECT_SR1, POLISH_DIRECT_BFGS, mock_run_fn_worse, dir;
                                    explore_budget_s = 60.0, polish_budget_s = 60.0)
        @test report2.result_source == "inherited_incumbent"
        @test report2.final_GT == 0.073
        @test report2.new_GT == 0.01   # the worse solved value IS recorded, just not exported as final
    end
end

println("All continuation_polish_orchestrator.jl tests passed.")
