# Focused re-run of Phase 2's origin-ZC arm only (the other three families' data is already
# solid and consistent across two full runs -- see RESTRICTED_DUAL_BANK_FINAL_DECISION_2026-07-26.md).
# Avoids re-running flexibleCM/commonFrechet/cmMeanZC again just to reach origin-ZC.
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
const BUDGET = 90.0
const OUT = joinpath(_D4E, "..", "..", "results", "phase2_dual_bank_2026-07-26")
mkpath(OUT)

lp("Building D=20 real context + calibration w0 ..."); flush(stdout)
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0))
const NU0_ORIGINZC_LOG = begin
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    log.(nu0)
end
const W0_ORIGINZC = vcat(w_calib, NU0_ORIGINZC_LOG)

function run_arm(label::AbstractString, bank_on::Bool, run_fn::Function)
    reset_restricted_dual_bank_counters!()
    t0 = time()
    res = run_fn(bank_on)
    wall = time() - t0
    c = deepcopy(RESTRICTED_DUAL_BANK_COUNTERS[])
    @printf("[%s bank=%s] wall=%.1fs knitro_status=%s n_eval=%d best_Delta=%s\n",
        label, bank_on, wall, string(res.knitro_status), res.n_eval, string(res.kappa))
    print_restricted_dual_bank_counters(c)
    return (label = label, bank_on = bank_on, wall = wall, knitro_status = res.knitro_status,
        n_eval = res.n_eval, best_delta = res.kappa, counters = c)
end

lp("="^100, "\nORIGIN-ZC\n", "="^100)
results = []
for bank_on in (false, true)
    push!(results, run_arm("originZC", bank_on, b -> run_originzc_upper_checkpointed(copy(W0_ORIGINZC);
        W = W, delta = DELTA, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        maxtime_real = BUDGET, ckpt_dir = joinpath(OUT, "originzc_bank$b"),
        label = "originzc", checkpoint_interval_s = 1000.0, verbose = false,
        use_exact_cache = true, use_dual_bank = b, A_coordinate_mode = :legacy_z)))
end

lp("="^100, "\nSUMMARY\n", "="^100)
for r in results
    @printf("%-15s bank=%-6s wall=%6.1f status=%8s n_eval=%3d best_Delta=%.6g queries=%d hits=%d wsf=%d\n",
        r.label, r.bank_on, r.wall, string(r.knitro_status), r.n_eval, r.best_delta,
        r.counters.queries, r.counters.hits, r.counters.warm_start_failures)
end
lp("PHASE2_ORIGINZC_ONLY_DONE")
