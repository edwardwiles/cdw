_D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_cplus.jl",
          "pairwise_quantile_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, Random
lp(xs...) = (println(xs...); flush(stdout))

const L    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

lp("="^96)
lp("HESSIAN ASSEMBLY BLOCK PROFILE  L=", L, "  reps=", REPS, "  julia threads=", Threads.nthreads())
lp("="^96)

t0 = time()
ctx_raw = d20_real_setup_design(W = 100_000, δ = 0.1, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("context in ", round(time() - t0, digits = 1), "s")

Z  = pairwise_quantile_frechet_features(ctx.U, ctx.μHat)
Q  = pairwise_quantile_fixed_cutoffs(Z, L; cutoff_source = :frechet_theoretical, mu_frechet = ctx.μHat)
op = PairwiseQuantileOperator(Z, L, Q)
D  = op.D
nrow = n_total_rows(D, L)
lp("n_total_rows = ", nrow, "   HfullR = ", round(nrow^2 * 8 / 2^30, digits = 2), " GiB")

tabs  = PairwiseQuantileHessianTables(op)
tls   = build_pairwise_quantile_thread_scratch(D, op.npair, L)
state = PairwiseQuantileMassState(D, L)
layout = PairwiseQuantileMassLayout(D, L)
set_pairwise_quantile_masses!(state, uniform_mass_raw(layout), layout)

Random.seed!(20260811)
h = rand(op.W) .+ 0.5
HRR = Matrix{Float64}(undef, nrow, nrow)
npk = div(nrow * (nrow + 1), 2)
hvec = Vector{Float64}(undef, npk)

function timeit(label, f, reps)
    f()                      # warm / JIT
    ts = Float64[]
    for _ in 1:reps
        GC.gc(false)
        t = time(); f(); push!(ts, time() - t)
    end
    m = minimum(ts)
    lp(@sprintf("  %-46s  min %8.3f s   median %8.3f s", label, m, sort(ts)[cld(length(ts), 2)]))
    m
end

lp("\nper-call block timings (min of ", REPS, " reps, after a warm call):")
t_tab  = timeit("build_..._hessian_tables!   (T1-T4 scatter)", () -> build_pairwise_quantile_hessian_tables!(tabs, op, h, tls), REPS)
t_fill = timeit("fill_..._hessian_raw!       (NOT threaded)",  () -> fill_pairwise_quantile_hessian_raw!(HRR, op, tabs), REPS)
t_cen  = timeit("center_and_scale_...!       (threaded)",      () -> center_and_scale_pairwise_quantile_hessian!(HRR, op, state, tabs), REPS)
t_pack = timeit("pack_upper_...!             (reference, serial)", () -> pack_upper_pairwise_quantile_hessian!(hvec, HRR, nrow), REPS)

# Reference point: a SERIAL fill! of the same buffer. fill_raw! now zeroes its output with a
# threaded loop, so this is no longer "the memset component of fill_raw!" -- it is the serial cost
# fill_raw! used to pay and no longer does. If it exceeds fill_raw! itself, that is the headline,
# not an inconsistency.
t_zero = timeit("reference: SERIAL fill!(HfullR, 0.0)", () -> fill!(HRR, 0.0), REPS)

tot = t_tab + t_fill + t_cen + t_pack
lp(@sprintf("\n  %-46s  %8.3f s", "TOTAL of the four blocks", tot))
for (lab, t) in (("tables", t_tab), ("fill_raw", t_fill), ("centering", t_cen), ("pack", t_pack))
    lp(@sprintf("     %-12s %8.3f s   %5.1f%%", lab, t, 100 * t / tot))
end
lp(@sprintf("\n  a SERIAL zero of the output buffer alone costs %.3f s; fill_raw! now does that AND",
    t_zero))
lp(@sprintf("  all of its scatter work in %.3f s (%.2fx the serial memset).", t_fill, t_zero / t_fill))
