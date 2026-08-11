# Evaluate the paper_upper_v1 ORIGIN_ZC delta=1 best point under BOTH the diagonal (:origin_by_power)
# and the CROSS (:origin_by_power_cross) restriction families, to measure the nesting gap in Delta*.
# 2026-08-10, user request.
#
# WHY THIS SHAPE: the point arrives as a bare 439-vector w = [gp(1); a_nonpivot(379); eta_nu(59)]
# in :powered_aspace coordinates with Variant D's focal eta OMITTED (dense index 22 of 60). Decoding
# that by hand is exactly the class of thing that goes silently wrong here (see the handoff README's
# own warning, and memory feedback-campaign-seed-w0-encoding-not-raw-theta-free). So this script does
# NOT decode anything itself: it hands w0 straight to `run_originzc_upper_checkpointed`, which applies
# its own `xf_from_w_econ` (:powered_aspace -> cm_z_from_a -> x_free_from_w) and its own
# `scatter_nu_eff(aml, exp.(w[381:439]), originzc_profiled_nu_value(xf, ctx))`. Whatever the driver
# does for a normal run, it does here.
#
# The DIAGONAL arm is the control, and it is not optional: it must reproduce the handoff's stated
# Delta* = 0.9999667279174992. If it does not, the transfer is wrong and the cross number means
# nothing. (CLAUDE.md: control against the unmodified family before concluding anything.)
#
# The eta block transfers VERBATIM between the two families -- confirmed by reading
# cm_originzc_cross_target_layout.jl:38-39: OriginByPowerCrossLayout has n_eta = K_mean*D = 60 and a
# byte-identical `target_index`, i.e. the cross extension adds NO new outer parameters; its pair
# targets are PRODUCTS nu[o,k1]*nu[p,k2] of the same nus. So there are no new etas to initialize and
# the handoff README's "new off-diagonal etas need their own initialization" caveat does not bite.
#
# Usage: julia --project=. -t 10 .../eval_originzc_handoff_point_2026-08-10.jl <diag|cross> [budget_s]
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
using Printf, Serialization
using SpecialFunctions: gamma
lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = length(ARGS) >= 1 ? ARGS[1] : error("eval_originzc_handoff_point: pass <diag|cross>")
const BUDGET = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2400.0
FAMILY in ("diag", "cross") || error("family must be diag or cross, got $FAMILY")

const HANDOFF = "/bbkinghome/edav/repo_scratch/paper_upper_v1/handoff_originzc_delta1/originzc_delta1_best_point.jls"
const W       = 100_000
const DELTA   = 1.0
const KK      = 3
const KSTAR   = 2          # Variant D: originzc_profiled_level = 2 = sigma-1
const DESIGN  = :sobol_randomized

# POINT=handoff (default) evaluates the paper_upper_v1 delta=1 best point. POINT=calibration
# evaluates the CALIBRATION point instead, which is the only way to get a FINITE Delta* for both
# families and hence a quantitative nesting ratio -- at the handoff point the cross inner solve is
# infeasible (-300), so the gap there is not a finite number.
const POINT = get(ENV, "POINT", "handoff")
pt = deserialize(HANDOFF)
w0 = collect(Float64, pt.w)
lp("="^100)
lp("ORIGIN_ZC delta=1 handoff point -> ", uppercase(FAMILY), " family    budget=", BUDGET, "s")
lp("="^100)
lp("handoff: gp=", pt.gp, "  Delta_star=", pt.Delta_star, "  kappa=", pt.kappa)
lp("w0 length = ", length(w0), " (expect 439 = 1 gp + 379 a_nonpivot + 59 eta_nu)")
length(w0) == 439 || error("expected a 439-vector, got $(length(w0))")

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = DESIGN, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx0.D; Ddest = ctx0.D_dest
lp("ctx: D=", D, " Ddest=", Ddest, " sigma=", ctx0.σ, " muHat=", ctx0.μHat, " bi=", ctx0.bi)

# Structural assertions on the transfer, BEFORE spending a solve on it.
layout_d = OriginByPowerLayout(D, KK, KK)
layout_x = OriginByPowerCrossLayout(D, KK, KK)
aml_d = ActiveMeanLayout(layout_d, ctx0.bi, KSTAR, D)
aml_x = ActiveMeanLayout(layout_x, ctx0.bi, KSTAR, D)
lp("diagonal layout: n_eta=", n_eta(layout_d), " active=", aml_d.n_eta_active, " omit_dense_idx=", aml_d.dense_omit_idx)
lp("cross    layout: n_eta=", n_eta(layout_x), " active=", aml_x.n_eta_active, " omit_dense_idx=", aml_x.dense_omit_idx)
@assert n_eta(layout_d) == n_eta(layout_x) == 60 "eta space must be identical across the two families"
@assert aml_d.n_eta_active == aml_x.n_eta_active == 59 "Variant D active eta count must be 59 in both"
@assert aml_d.dense_omit_idx == aml_x.dense_omit_idx == 22 "the omitted focal slot must be dense index 22 in both (handoff README)"
lp("=> eta block transfers VERBATIM: identical n_eta, identical omitted slot. No new etas to seed.")
lp("gp (w0[1]) = ", w0[1], "   matches handoff: ", w0[1] == pt.gp)
nu = exp.(w0[381:439])
lp("nu = exp.(w0[381:439]) range: [", minimum(nu), ", ", maximum(nu), "]  (README says [1.0760748687756763, 1.5988108982487386])")

if POINT == "calibration"
    # Same construction as production_smoke_ozc_cross_2026-08-09.jl (reused, not re-derived):
    # calibration point in :powered_aspace coordinates, eta0 from the THEORETICAL population mean
    # Gamma(1 - mu*k) -- never a sample average of the draws the restriction is imposed on -- with
    # the Variant D omitted slot dropped.
    pe0 = build_pivot_elimination(ctx0); theta0 = cm_fixed_theta(ctx0); xy0 = precompute_cm_aspace_xy(ctx0)
    xfc = ctx0.θ0_up[ctx0.free_idx]
    zc  = pivot_reduce(log.(reshape(xfc[2:end], D, Ddest)), pe0)
    wac = vcat(xfc[1], cm_a_from_z(zc, theta0, xy0, pe0))
    nu0_dense = Vector{Float64}(undef, n_eta(layout_d))
    for k in 1:KK, o in 1:D
        nu0_dense[target_index(layout_d, o, k)] = gamma(1 - ctx0.μHat * k)
    end
    eta0 = [log(nu0_dense[d]) for d in 1:length(nu0_dense) if d != aml_d.dense_omit_idx]
    global w0 = vcat(wac, eta0)
    lp("\n>> POINT=calibration: replaced w0 with the CALIBRATION point (len=", length(w0), "), gp=", w0[1])
    lp(">> nu0 (theoretical Gamma(1-mu*k)) = ", [gamma(1 - ctx0.μHat * k) for k in 1:KK])
end

const LAYOUT_SYM = FAMILY == "cross" ? :origin_by_power_cross : :origin_by_power
const LABEL = "ozc_$(FAMILY)_handoffpt"
const OUT = joinpath(_D4E, "..", "..", "results", "originzc_handoff_point_2026-08-10", "$(POINT)_$(FAMILY)")
rm(OUT; force = true, recursive = true); mkpath(OUT)

lp("\n>> run_originzc_upper_checkpointed  power_target_layout=", LAYOUT_SYM,
   "  K_mean=K_pair=", KK, "  originzc_profiled_level=", KSTAR)
lp(">> the driver applies its OWN xf_from_w_econ and scatter_nu_eff to w0 -- nothing decoded here.")
t0 = time()
result = run_originzc_upper_checkpointed(w0;
    W = W, delta = DELTA, draw_design = DESIGN, draw_seed = 20260719,
    distribution_restriction = :origin_specific_moments_zero_covariance,
    K_mean = KK, K_pair = KK,
    power_target_layout = LAYOUT_SYM,
    originzc_profiled_level = KSTAR,
    inner_lower_limit = -10.0,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    ckpt_dir = OUT, run_id = LABEL, label = LABEL,
    checkpoint_interval_s = 600.0, maxtime_real = BUDGET, verbose = true)
wall = time() - t0

lp("\n", "="^100)
@printf("RESULT %s at handoff point: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d\n",
        uppercase(FAMILY), wall, string(result.knitro_status), result.n_eval, result.n_grad)
if !isempty(result.trace)
    r1 = result.trace[1]
    @printf("  EVAL 1 (= the handoff point itself): gp=%.16f  Delta_star=%.16f  feasible=%s verified=%s\n",
            r1.gp, r1.Delta, string(r1.feasible), string(r1.verified))
    @printf("  handoff diagonal Delta* for comparison:                    %.16f\n", pt.Delta_star)
    @printf("  ratio  Delta*_%s / Delta*_handoff = %.6f     difference = %+.6f\n",
            FAMILY, r1.Delta / pt.Delta_star, r1.Delta - pt.Delta_star)
end
lp("  full trace (", length(result.trace), " evals):")
for r in result.trace
    @printf("    eval %3d t=%7.1fs gp=%.10f Delta=%.10f feasible=%s verified=%s\n",
            r.idx, r.t, r.gp, r.Delta, string(r.feasible), string(r.verified))
end
if result.best !== nothing
    @printf("  best feasible: gp=%.10f Delta=%.10f at eval %d ; kappa=%.6f%%\n",
            result.best.gp, result.best.Delta, result.best.n_eval, 100 * result.kappa)
else
    lp("  NO feasible point found (expected for the cross family if Delta* > 1 at this point)")
end
lp("="^100)
