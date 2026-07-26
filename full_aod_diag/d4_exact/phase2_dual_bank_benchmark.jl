# Five-family finish task, Phase 2 (2026-07-26): real-trajectory benchmark of the restricted
# dual bank (distance-only RestrictedDualBank, cm_dual_bank_production.jl) for all four restricted
# families, through the ACTUAL public driver entry points (run_cm_upper_checkpointed /
# run_originzc_upper_checkpointed) -- not an isolated point-eval microbenchmark. Answers whether
# use_dual_bank=true should ever become a default (task's own KEEP_OPT_IN policy pending exactly
# this evidence).
#
# For each family: TWO real short outer-loop campaigns from the SAME calibration start, SAME
# maxtime_real budget, bank=off vs bank=on. Reports: warm/cold solve counts, KNITRO status mix,
# n_eval, wall time, best kappa/Delta achieved in budget (verified outer progress), selected
# distances, warm_start_failures ("harmful warm starts").
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))
const W = 80_000
const DELTA = 1.0
const BUDGET = 90.0   # real seconds per arm -- short-trajectory evidence, not a final profile
const OUT = joinpath(_D4E, "..", "..", "results", "phase2_dual_bank_2026-07-26")
mkpath(OUT)

# Real calibration w0, same construction cm_production_stage_runner.jl/originzc_production_stage_runner.jl
# use for a fresh (non-resumed) run -- run_cm_upper_checkpointed/run_originzc_upper_checkpointed
# both REQUIRE an explicit w0 for a fresh run (no implicit "nothing means calibration" convenience
# the unrestricted family's own stage runner provides).
lp("Building D=20 real context + calibration w0 ..."); flush(stdout)
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0))
const W0_PLAIN = w_calib
const W0_MEANZC = vcat(w_calib, log.(Float64.(factorial.(1:1))))   # eta_nu0 for K_mean=1
const NU0_ORIGINZC_LOG = begin
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1
        Uk = ctx0.U .^ k
        for o in 1:D
            nu0[target_index(layout0, o, k)] = mean(@view Uk[:, o])
        end
    end
    log.(nu0)
end
const W0_ORIGINZC = vcat(w_calib, NU0_ORIGINZC_LOG)
const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[50]
lp("w0 built: ||W0_PLAIN||=", round(norm(W0_PLAIN), digits=4))

"Runs one (family,bank) arm; returns comparable summary NamedTuple. `run_fn(bank_on::Bool)` must
call the real driver (passing use_dual_bank=bank_on itself) and return its own result NamedTuple
(with knitro_status/n_eval/kappa/best_feasible fields)."
function run_arm(label::AbstractString, bank_on::Bool, run_fn::Function)
    reset_restricted_dual_bank_counters!()
    t0 = time()
    res = run_fn(bank_on)
    wall = time() - t0
    c = deepcopy(RESTRICTED_DUAL_BANK_COUNTERS[])
    best_kappa = res.kappa
    @printf("[%s bank=%s] wall=%.1fs knitro_status=%s n_eval=%d best_Delta=%s\n",
        label, bank_on, wall, string(res.knitro_status), res.n_eval, string(best_kappa))
    print_restricted_dual_bank_counters(c)
    return (label = label, bank_on = bank_on, wall = wall, knitro_status = res.knitro_status,
        n_eval = res.n_eval, best_delta = best_kappa, counters = c)
end

results = []

lp("="^100, "\nFLEXIBLE CM\n", "="^100)
for bank_on in (false, true)
    push!(results, run_arm("flexibleCM", bank_on, b -> run_cm_upper_checkpointed(copy(W0_PLAIN);
        W = W, delta = DELTA, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
        maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "flexcm_bank$b"),
        label = "flexcm", checkpoint_interval_s = 1000.0, verbose = false,
        use_exact_cache = true, use_dual_bank = b, A_coordinate_mode = :legacy_z)))
end

lp("="^100, "\nCOMMON FRECHET\n", "="^100)
for bank_on in (false, true)
    push!(results, run_arm("commonFrechet", bank_on, b -> run_cm_upper_checkpointed(copy(W0_PLAIN);
        W = W, delta = DELTA, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
        marginal_restriction = :common_frechet, cm_hessian_backend = :structured,
        maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "frechet_bank$b"),
        label = "frechet", checkpoint_interval_s = 1000.0, verbose = false,
        use_exact_cache = true, use_dual_bank = b, A_coordinate_mode = :legacy_z)))
end

lp("="^100, "\nCM+mean/ZC\n", "="^100)
for bank_on in (false, true)
    push!(results, run_arm("cmMeanZC", bank_on, b -> run_cm_upper_checkpointed(copy(W0_MEANZC);
        W = W, delta = DELTA, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "meanzc_bank$b"),
        label = "meanzc", checkpoint_interval_s = 1000.0, verbose = false,
        use_exact_cache = true, use_dual_bank = b, A_coordinate_mode = :legacy_z)))
end

lp("="^100, "\nORIGIN-ZC\n", "="^100)
for bank_on in (false, true)
    push!(results, run_arm("originZC", bank_on, b -> run_originzc_upper_checkpointed(copy(W0_ORIGINZC);
        W = W, delta = DELTA, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "originzc_bank$b"),
        label = "originzc", checkpoint_interval_s = 1000.0, verbose = false,
        use_exact_cache = true, use_dual_bank = b, A_coordinate_mode = :legacy_z)))
end

lp("="^100, "\nSUMMARY TABLE\n", "="^100)
@printf("%-15s %-6s %8s %10s %8s %10s %8s %8s %8s\n",
    "family", "bank", "wall_s", "status", "n_eval", "best_Delta", "queries", "hits", "wsf")
for r in results
    @printf("%-15s %-6s %8.1f %10s %8d %10.6g %8d %8d %8d\n",
        r.label, r.bank_on, r.wall, string(r.knitro_status), r.n_eval, r.best_delta,
        r.counters.queries, r.counters.hits, r.counters.warm_start_failures)
end
lp("PHASE2_DUAL_BANK_BENCHMARK_DONE")
