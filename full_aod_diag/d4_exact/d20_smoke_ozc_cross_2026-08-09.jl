# D20 real-data smoke test for OZC-CROSS (2026-08-09 task, phase 3): repeats the D4
# convergence+residual checks (smoke_ozc_cross_d4_2026-08-09.jl) at real D=20 Brazil-Korea data
# under destination_sample=:exclude_row (the actual production default) -- per this repo's own
# precedent (fullA-zc-profiled-focal-sigmaminus1-mean-production-ready memory), a new ZC-family
# feature validated only under D4/:all_legacy-equivalent conditions is NOT sufficient evidence for
# production readiness; :exclude_row's rectangular D x D_dest layout is where real bugs have been
# found before (autarky_cf.jl's D^2 reshape). OZC-CROSS's own restriction block is origin-indexed
# only (never touches D_dest), so it is not expected to be exposed to that SPECIFIC bug class, but
# this is a real, not assumed, check under the real production configuration.
#
# Starts at a MODEST W (not the full W=100,000 production scale) to get a fast first read on
# whether the wiring is even correct at D20/:exclude_row before spending wall-clock on realism.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/d20_smoke_ozc_cross_2026-08-09.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

W = 100_000
println("Building D20 real-data context (W=$W, destination_sample=:exclude_row, sigma=3, production recipe)...")
flush(stdout)
t_ctx = @elapsed ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
println("ctx built in $(t_ctx)s. D=$(ctx.D) D_dest=$(ctx.D_dest) μHat=$(ctx.μHat)")
flush(stdout)

D = ctx.D; Ddest = ctx.D_dest
# D20's free-parameter space uses the pivot-elimination (powered_aspace) encoding for A_od, NOT a
# direct theta0_up[free_idx] assignment like D4 -- confirmed the hard way (first attempt with the
# D4 convention gave x_free0 entries up to 6.6e6 and nStatus=-300; see
# reference-multistart-seed-generator / feedback-campaign-seed-w0-encoding-not-raw-theta-free
# memory for the same lesson). Matches test_originzc_operator_correctness.jl's own d20 branch
# exactly (build_pivot_elimination -> pivot_reduce(log(Aod)) -> pivot_expand -> exp, at the
# UNPERTURBED calibration point, i.e. no *1.01 on gp0).
pe_g = build_pivot_elimination(ctx)
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe_g)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0, zfree0)
x_free_calib = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe_g))))
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    println("\n==== D20 (:exclude_row, W=$W) OZC-CROSS K_mean=$K_mean K_pair=$K_pair (K_pair^2=$(K_pair^2) combos/origin-pair) ====")
    flush(stdout)
    layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    t_build = @elapsed pcx = build_originzc_cross_production_context(ctx, CS, layout)
    println("    context build: $(t_build)s  n_pair=$(pcx.aug.n_pair)  n_mean=$(pcx.aug.n_mean)")
    flush(stdout)

    t_solve = @elapsed (base, verify) = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
    println("    inner solve: $(t_solve)s  inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)  m_mean=$(verify.m_mean)")
    flush(stdout)
    check("K=$K_mean/$K_pair: inner solve converged (nStatus in (0,-100,-101,-103))", verify.inner_status in (0, -100, -101, -103))
    check("K=$K_mean/$K_pair: KKT moment residual ~0", verify.max_abs_moment_kkt_resid < 1e-4)

    m_weights = base.m_star
    worst_mean = 0.0
    for k in 1:K_mean
        νo_k = mean_targets(layout, νfull0, k, D)
        r = recovered_mean_residuals_origin(m_weights, pcx.aug.Zraw_all[k], νo_k)
        worst_mean = max(worst_mean, maximum(abs.(r)))
    end
    check("K=$K_mean/$K_pair: worst mean-block residual ~0 (max=$worst_mean)", worst_mean < 1e-4)

    levels = cross_pair_level_index(K_pair)
    worst_pair = 0.0
    worst_pair_level = (0, 0)
    for (klin, (k1, k2)) in enumerate(levels)
        νprod = pair_targets(layout, νfull0, klin, D)
        r = recovered_pair_residuals_origin(m_weights, pcx.aug.Zpairraw_all[klin], νprod)
        m = maximum(abs.(r))
        if m > worst_pair
            worst_pair = m; worst_pair_level = (k1, k2)
        end
    end
    println("    worst cross-pair residual: $worst_pair at level (k1,k2)=$worst_pair_level")
    check("K=$K_mean/$K_pair: worst cross-pair residual ~0 (max=$worst_pair)", worst_pair < 1e-4)
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
