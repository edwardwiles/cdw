# PRODUCTION-STYLE outer-loop smoke test for OZC-CROSS through the REAL checkpointed production
# driver `run_originzc_upper_checkpointed` (cm_originzc_checkpoint.jl) -- 2026-08-09.
#
# This is the end-to-end wiring test: full checkpoint/resume, exact-point cache, Variant D
# (originzc_profiled_level=2), and crucially the driver's OWN DEFAULT gradient backend
# `cm_gradient_backend=:cplus`, which now dispatches to the newly-implemented
# `cm_originzc_cross_production_gradient_cplus` (cm_originzc_cross_cplus.jl). That C+ path was
# A/B-verified against the reference gradient at machine precision (rel 1e-16..1e-18) across
# K=1/1,2/2,3/3 with and without Variant D by verify_cross_cplus_ab_d4_2026-08-09.jl -- run that
# first if anything here looks wrong.
#
# NOT the lightweight single-shot `run_originzc_cross_upper` (originzc_cross_outer_driver_*.jl),
# which was only ever a smoke tool and has no checkpoint/dual-bank/exact-cache machinery.
#
# nu0 SEEDING -- deliberately NOT copied from smoke_delta1_originzc.jl, which seeds
# `nu0[o] = mean(U[:,o]^k)`. That is wrong twice over for this family and both errors are already
# on record in this repo: (1) the restricted feature is z = U^(-mu) NOT U, so a power of U is the
# wrong transform entirely (memory: feedback-power-weighted-features-need-z-not-u-transform); and
# (2) the target must be the THEORETICAL population mean, not a sample average of the same draws the
# restriction is imposed on (memory: zc-cmzc-exclude-row-k2-k3-production-2026-08-07). Correct value:
# E[z^k] = E[U^(-mu*k)] = Gamma(1 - mu*k) for U ~ Exp(1).
#
# Usage: julia --project=. -t 8 .../production_smoke_ozc_cross_2026-08-09.jl <K> <W> <budget_s> [design]
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics
using SpecialFunctions: gamma
lp(xs...) = (println(xs...); flush(stdout))

const KK      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const W       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000
const BUDGET  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 900.0
const DESIGN  = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :sobol_randomized   # driver's own default
const DELTA   = 1.0
const KSTAR   = 2      # = sigma-1 at sigmaHat=3.0 -> Variant D active
# Run-level (NOT scientific) overrides, 2026-08-10 -- see the twin comment in
# production_smoke_cmzc_cross_2026-08-09.jl. These change wall-budget bookkeeping and where output
# lands, never what economic problem is solved.
const RUN_TAG   = get(ENV, "RUN_TAG", "")
const CKPT_INT  = parse(Float64, get(ENV, "CKPT_INTERVAL_S", "60.0"))
const BLAS_THR  = haskey(ENV, "BLAS_THREADS") ? parse(Int, ENV["BLAS_THREADS"]) : ZC_GRAM_BLAS_THREADS_DEFAULT[]
const OUT = joinpath(_D4E, "..", "..", "results", "ozc_cross_production_smoke_2026-08-09",
                     "K$(KK)_W$(W)_$(DESIGN)" * (isempty(RUN_TAG) ? "" : "_$(RUN_TAG)"))
rm(OUT; force = true, recursive = true); mkpath(OUT)

lp("=== OZC-CROSS PRODUCTION SMOKE: K_mean=K_pair=$KK  W=$W  budget=$(BUDGET)s  draw_design=$DESIGN ===")
lp("    driver=run_originzc_upper_checkpointed  gradient_backend=:cplus (DEFAULT)  Variant D kstar=$KSTAR")
lp("    run_tag=", isempty(RUN_TAG) ? "(none)" : RUN_TAG, "  blas_threads=", BLAS_THR,
   "  checkpoint_interval_s=", CKPT_INT, "  julia_threads=", Threads.nthreads(), "  out=", OUT)

# ---- ctx built with the IDENTICAL config the driver will use internally, so w0 is consistent ----
ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = DESIGN, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx0.D; Ddest = ctx0.D_dest
lp("ctx built. D=$D Ddest=$Ddest sigma=$(ctx0.σ) muHat=$(ctx0.μHat) bi=$(ctx0.bi)")

# ---- w0 in the driver's A_coordinate_mode=:powered_aspace coordinates (its default) ----
pe0   = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0    = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib  = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib  = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)

# ---- eta0: theoretical Gamma(1-mu*k), then drop the Variant D omitted coordinate ----
layout0  = OriginByPowerCrossLayout(D, KK, KK)
aml0     = ActiveMeanLayout(layout0, ctx0.bi, KSTAR, D)
nu0_dense = Vector{Float64}(undef, n_eta(layout0))
for k in 1:KK, o in 1:D
    nu0_dense[target_index(layout0, o, k)] = gamma(1 - ctx0.μHat * k)
end
eta0_active = [log(nu0_dense[d]) for d in 1:length(nu0_dense) if d != aml0.dense_omit_idx]
lp("n_eta dense=$(n_eta(layout0)) -> active=$(aml0.n_eta_active) (omitted dense idx $(aml0.dense_omit_idx))")
lp("w0 length = $(length(w_a_calib) + length(eta0_active))  (econ $(length(w_a_calib)) + eta $(length(eta0_active)))")

t0 = time()
result = run_originzc_upper_checkpointed(vcat(w_a_calib, eta0_active);
    W = W, delta = DELTA, draw_design = DESIGN, draw_seed = 20260719,
    distribution_restriction = :origin_specific_moments_zero_covariance,
    K_mean = KK, K_pair = KK,
    power_target_layout = :origin_by_power_cross,          # <-- OZC-CROSS
    originzc_profiled_level = KSTAR,                        # <-- Variant D
    inner_lower_limit = -10.0,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    ckpt_dir = OUT, run_id = "ozc_cross_K$(KK)_W$(W)", label = "ozc_cross_K$(KK)_W$(W)",
    blas_threads = BLAS_THR,
    checkpoint_interval_s = CKPT_INT, maxtime_real = BUDGET, verbose = true)
wall = time() - t0

lp("="^95)
@printf("RESULT OZC-CROSS K=%d/%d W=%d design=%s: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
        KK, KK, W, string(DESIGN), wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
if result.best !== nothing
    @printf("  best feasible: gp=%.10f  Delta=%.10f  found at eval %d (t=%.1fs)\n",
            result.best.gp, result.best.Delta, result.best.n_eval, result.best.t)
    @printf("  kappa = 1 - gp^(sigma/(sigma-1)) = %.6f%%\n", 100 * result.kappa)
else
    lp("  NO feasible point found within budget")
end
lp("  trace (ALL ", length(result.trace), " evals -- the search path, needed for the write-up):")
for r in result.trace
    @printf("    eval %3d t=%7.1fs gp=%.8f Delta=%.8f feasible=%s verified=%s\n",
            r.idx, r.t, r.gp, r.Delta, string(r.feasible), string(r.verified))
end
ck = joinpath(OUT, "ozc_cross_K$(KK)_W$(W)_latest.jls")
lp("  checkpoint written: ", isfile(ck), "  -> ", ck)
if isfile(ck)
    # load_cm_checkpoint_v10 (NOT load_cm_checkpoint_v5): this driver writes OriginZCCheckpointV10,
    # a distinct struct from the CM-family chain; the v5 loader tries the CM layouts and hard-errors.
    c = load_cm_checkpoint_v10(ck)
    @printf("  checkpoint reports: power_target_layout=%s K_mean=%d K_pair=%d n_eta_stored=%d origin_D=%d moment_layout_version=%d\n",
            string(c.power_target_layout), c.origin_K_mean, c.origin_K_pair, length(c.eta_nu), c.origin_D, c.origin_moment_layout_version)
    @printf("  checkpoint layout tag round-trips as OZC-CROSS: %s\n", string(c.power_target_layout) == "origin_by_power_cross")
    @printf("  checkpoint eta length matches Variant D active count (%d): %s\n",
            aml0.n_eta_active, length(c.eta_nu) == aml0.n_eta_active)
end
lp("="^95)
