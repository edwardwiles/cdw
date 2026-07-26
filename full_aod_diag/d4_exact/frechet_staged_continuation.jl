# Phase 0 gate 3 (2026-07-26, production-audit task): full staged continuation chain
# delta=0.01 -> 0.1 -> 0.5 -> 1.0, from the genuine calibrated point, for BOTH
# marginal_restriction=:common_frechet and the matched :common_flexible (flexible-CM) control --
# closing the disclosed gap in COMMON_FRECHET_CONTINUATION_OUTER_GATE_2026-07-25.md ("this session
# ran the explicitly-allowed simpler alternative [direct delta=1] instead... the full 4-stage
# chain was NOT run").
#
# Continuation mechanism: run_cm_upper_checkpointed (cm_checkpoint.jl) has no in-process
# dual-bank/exact-cache passthrough between separate calls (each call builds its own DualBank/
# SafeExactCache internally) -- the outer-loop-level continuation mechanism this driver's public
# API actually supports is carrying the previous stage's best cold-verified incumbent's outer
# coordinate vector (result.best.w, format [gp; zfree...], see cm_checkpoint.jl:893's
# best_feasible[] construction) forward as the NEXT stage's w0. That is what this script does.
# Disclosed, not a silent scope reduction: this carries forward the outer point, not the inner
# dual state itself (the inner solve warm-starts from its own fresh DualBank each stage).
#
# Per-stage wall-clock budget: 300s (not the 15-30 min/stage the branch's own prior doc
# mentioned), a disclosed budget choice given the overall audit's scope -- 4 stages x 2 families x
# 300s = 2400s (40 min) total, run SEQUENTIALLY (this project's standing rule: never concurrent
# KNITRO). Sequential, not concurrent.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)
using Printf

W = 80000
L = 50
probs = cm_equal_grid_probs(L)
ctx0 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D0 = ctx0.D; Ddest0 = ctx0.D_dest
x_free_calib0 = ctx0.θ0_up[ctx0.free_idx]
w0_calib = vcat(x_free_calib0[1], pivot_reduce(log.(reshape(x_free_calib0[2:end], D0, Ddest0)), pe0))

DELTAS = [0.01, 0.1, 0.5, 1.0]
STAGE_BUDGET_S = 300.0

function run_chain(marginal_restriction::Symbol, label_prefix::String)
    println("="^90)
    println("STAGED CONTINUATION: marginal_restriction=", marginal_restriction)
    println("="^90)
    w0 = copy(w0_calib)
    stage_results = NamedTuple[]
    for (i, delta) in enumerate(DELTAS)
        CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "$(label_prefix)_continuation_stage$(i)")
        rm(CKPT_DIR; force = true, recursive = true); mkpath(CKPT_DIR)
        println("-- stage $i/$(length(DELTAS)): delta=$delta, w0 carried from previous stage --")
        flush(stdout)
        t0 = time()
        result = run_cm_upper_checkpointed(w0; W = W, delta = delta, draw_design = :pseudorandom, draw_seed = 20260719,
            L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
            marginal_restriction = marginal_restriction, cm_gradient_backend = :cplus,
            ckpt_dir = CKPT_DIR, run_id = "$(label_prefix)_stage$(i)", label = "$(label_prefix)_stage$(i)",
            checkpoint_interval_s = 60.0, maxtime_real = STAGE_BUDGET_S, verbose = true)
        wall = time() - t0
        best_gp = result.best === nothing ? NaN : result.best.gp
        best_Delta = result.best === nothing ? NaN : result.best.Delta
        @printf ">>> stage %d (delta=%.2f): wall=%.1fs n_eval=%d n_grad=%d best_gp=%.6f best_Delta=%.6f knitro_status=%d\n" i delta wall result.n_eval result.n_grad best_gp best_Delta result.knitro_status
        push!(stage_results, (delta = delta, wall = wall, n_eval = result.n_eval, n_grad = result.n_grad,
                               best_gp = best_gp, best_Delta = best_Delta, knitro_status = result.knitro_status,
                               fallback_unexplained = result.screen_summary))
        if result.best !== nothing
            w0 = copy(result.best.w)   # carry forward the best cold-verified incumbent's outer point
        else
            println(">>> WARNING: stage $i produced no feasible incumbent -- carrying forward the raw terminal xsol instead (NOT a verified incumbent)")
            w0 = copy(result.xsol)
        end
    end
    return stage_results
end

println(">>> calibration gp = ", w0_calib[1])
flush(stdout)

results_frechet = run_chain(:common_frechet, "phase0_frechet")
results_flexcm = run_chain(:common_flexible, "phase0_flexcm")

println()
println("="^90)
println("SUMMARY: staged continuation delta=0.01->0.1->0.5->1.0")
println("="^90)
println("-- common_frechet --")
for r in results_frechet
    @printf "  delta=%.2f  n_eval=%d n_grad=%d best_gp=%.6f best_Delta=%.6f knitro_status=%d wall=%.1fs\n" r.delta r.n_eval r.n_grad r.best_gp r.best_Delta r.knitro_status r.wall
end
println("-- common_flexible (control) --")
for r in results_flexcm
    @printf "  delta=%.2f  n_eval=%d n_grad=%d best_gp=%.6f best_Delta=%.6f knitro_status=%d wall=%.1fs\n" r.delta r.n_eval r.n_grad r.best_gp r.best_Delta r.knitro_status r.wall
end

gp_calib = w0_calib[1]
frechet_d1 = results_frechet[end]
flexcm_d1 = results_flexcm[end]
improved_frechet = !isnan(frechet_d1.best_gp) && frechet_d1.best_gp < gp_calib
improved_flexcm = !isnan(flexcm_d1.best_gp) && flexcm_d1.best_gp < gp_calib
println()
println(">>> STAGED-CHAIN GATE: improved cold-verified incumbent by delta=1 -- frechet=", improved_frechet, "  flexcm=", improved_flexcm)
println(improved_frechet && improved_flexcm ? "STAGED CONTINUATION GATE: PASS" : "STAGED CONTINUATION GATE: FAIL (see above)")
