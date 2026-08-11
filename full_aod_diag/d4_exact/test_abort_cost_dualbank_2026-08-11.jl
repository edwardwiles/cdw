# How much work does KNITRO still do on a DOOMED inner solve? (2026-08-11)
#
# `lower_limit = -10` is this codebase's abort mechanism: the FG callback reports
# `obj = -KN_INFINITY` once `f <= lower_limit`, and since `Delta_dual = -f`
# (operator_verification.jl:512) that is exactly a threshold abort at `Delta* >= 10`. It fires at
# EVERY FG evaluation, so `ThresholdAbortState` would be the same certificate in the same place.
#
# What is NOT known is its COST PROFILE. If a solve is doomed from its very first evaluation -- the
# warm start already gives f = -15, say -- does KNITRO return immediately, or does it still build a
# Hessian and take steps first? Aggregate counters cannot answer this: they average doomed and
# healthy solves together. This records EVERY inner solve separately via INNER_SOLVE_TRACE:
#
#   f_first      objective at the FIRST FG call (i.e. at the warm start KNITRO was handed)
#   first_below  1-based index of the first FG call at/under lower_limit (0 = never)
#   n_fg,n_hess  callbacks KNITRO actually issued
#   wall         seconds
#
# The decisive cell is "doomed at the initial point" (first_below == 1): its n_hess and wall are
# exactly what a pre-solve check could save, and its n_fg tells whether KNITRO even needed more than
# the one evaluation a cached/dual-bank pre-check would itself cost.
#
# Usage: julia --project=. -t 10 .../test_abort_cost_2026-08-11.jl [budget_s]
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Serialization, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const BUDGET = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1200.0
const KK = 3; const KSTAR = 2; const W = 100_000; const DELTA = 1.0
const CKPT = joinpath(_D4E, "..", "..", "results", "ozc_cross_production_smoke_2026-08-09",
                      "K3_W100000_sobol_randomized_upper8h_2026-08-10", "ozc_cross_K3_W100000_latest.jls")
const OUT = joinpath(_D4E, "..", "..", "results", "abort_cost_bank_study_2026-08-11")

lp("="^104)
lp("ABORT COST: what does a DOOMED inner solve actually cost, and when is it knowable?")
lp("="^104)

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0, inner_lower_limit = -10.0)
lp("inner_lower_limit = ", ctx0.obj.lower_limit, "   (Delta_dual = -f, so this aborts at Delta* >= ",
   -ctx0.obj.lower_limit, ")")

ck = load_cm_checkpoint_v10(CKPT); bf = ck.best_feasible
w0 = collect(Float64, bf.w)
lp("starting from the 8h run's own incumbent: gp=", bf.gp, "  Delta=", bf.Delta,
   "  (~82% of attempts fail there)")
rm(OUT; force = true, recursive = true); mkpath(OUT)

INNER_SOLVE_TRACE[] = Any[]
t0 = time()
result = run_originzc_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :sobol_randomized,
    draw_seed = 20260719, distribution_restriction = :origin_specific_moments_zero_covariance,
    K_mean = KK, K_pair = KK, power_target_layout = :origin_by_power_cross,
    originzc_profiled_level = KSTAR, inner_lower_limit = -10.0,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    ckpt_dir = OUT, run_id = "abort_cost_bank", label = "abort_cost_bank", use_dual_bank = true,
    checkpoint_interval_s = 1e9, maxtime_real = BUDGET, verbose = true)
wall = time() - t0
tr = INNER_SOLVE_TRACE[]; INNER_SOLVE_TRACE[] = nothing
serialize(joinpath(OUT, "inner_solves.jls"), tr)     # persist BEFORE analysing
lp("\n  per-solve trace serialized (", length(tr), " solves) -> ", joinpath(OUT, "inner_solves.jls"))

function report(tr, wall, n_eval)
    lp("\n", "="^104)
    @printf("%d inner solves in %.0f s of driver wall   (driver n_eval=%d => %d rejected)\n",
            length(tr), wall, n_eval, length(tr) - n_eval)
    lp("="^104)
    groups = ("doomed at the FIRST FG call (first_below==1)" => filter(r -> r.first_below == 1, tr),
              "crossed the limit LATER (first_below>1)"      => filter(r -> r.first_below > 1, tr),
              "never crossed the limit (first_below==0)"     => filter(r -> r.first_below == 0, tr))
    @printf("  %-46s %5s %9s %9s %10s %12s\n", "group", "n", "med n_fg", "med n_hess", "med wall", "med f_first")
    for (name, g) in groups
        isempty(g) && (@printf("  %-46s %5d %9s %9s %10s %12s\n", name, 0, "-", "-", "-", "-"); continue)
        @printf("  %-46s %5d %9.1f %9.1f %9.2fs %12.4g\n", name, length(g),
                median(getfield.(g, :n_fg)), median(getfield.(g, :n_hess)),
                median(getfield.(g, :wall)), median(getfield.(g, :f_first)))
    end
    doomed = filter(r -> r.first_below == 1, tr)
    if !isempty(doomed)
        lp("\n  -- the decisive group: doomed from the warm start KNITRO was handed --")
        @printf("     n=%d   total wall %.1f s (%.1f%% of the driver's %.0f s)\n",
                length(doomed), sum(getfield.(doomed, :wall)),
                100 * sum(getfield.(doomed, :wall)) / wall, wall)
        @printf("     n_hess: min=%d median=%.1f max=%d      <-- Hessian calls made on a solve already known doomed\n",
                minimum(getfield.(doomed, :n_hess)), median(getfield.(doomed, :n_hess)), maximum(getfield.(doomed, :n_hess)))
        @printf("     n_fg:   min=%d median=%.1f max=%d      <-- a pre-check costs exactly ONE of these\n",
                minimum(getfield.(doomed, :n_fg)), median(getfield.(doomed, :n_fg)), maximum(getfield.(doomed, :n_fg)))
        @printf("     statuses: %s\n", join(sort(unique(getfield.(doomed, :status))), ", "))
        sv = sum(getfield.(doomed, :wall)) - length(doomed) * (isempty(tr) ? 0.0 : median(getfield.(tr, :wall)) / max(median(getfield.(tr, :n_fg)), 1))
        @printf("     => a pre-solve check that skipped these entirely would save ~%.1f s of %.0f s (%.1f%%)\n",
                sum(getfield.(doomed, :wall)), wall, 100 * sum(getfield.(doomed, :wall)) / wall)
    end
    later = filter(r -> r.first_below > 1, tr)
    isempty(later) || @printf("\n  NOTE %d solves only crossed the limit after %d..%d FG calls -- a pre-check on the\n       warm start CANNOT catch these; only the in-solve limit can.\n",
                              length(later), minimum(getfield.(later, :first_below)), maximum(getfield.(later, :first_below)))
end
report(tr, wall, result.n_eval)
flush(stdout)
