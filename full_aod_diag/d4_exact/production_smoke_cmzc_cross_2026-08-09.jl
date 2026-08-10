# PRODUCTION-STYLE outer-loop smoke test for CM+ZC-CROSS through the REAL checkpointed production
# driver `run_cm_upper_checkpointed` (cm_checkpoint.jl) -- 2026-08-09.
#
# End-to-end wiring test: full checkpoint/resume (schema 11, which carries the new
# `meanzc_target_layout` field), exact-point cache (family_tag=:cm_meanzc_cross, so it can never
# collide with the diagonal family's entries), Variant D (`meanzc_profiled_level=2`), and crucially
# the driver's OWN DEFAULT gradient backend `cm_gradient_backend=:cplus`, which dispatches to the
# newly-implemented `cm_meanzc_cross_production_gradient_cplus` (cm_meanzc_cross_cplus.jl). That C+
# path was A/B-verified against the reference gradient at machine precision (rel 1e-16..1e-19)
# across K=1/1,2/2,3/3 with and without Variant D by verify_cmzc_cross_cplus_ab_d4_2026-08-09.jl --
# run that first if anything here looks wrong.
#
# Structural twin of production_smoke_ozc_cross_2026-08-09.jl, including its nu0 seeding rationale:
# the target is the THEORETICAL population mean E[z^k] = E[U^(-mu*k)] = Gamma(1 - mu*k) for
# U ~ Exp(1) -- NOT mean(U[:,o]^k), which is wrong twice over (wrong transform: the feature is
# z = U^(-mu), not U; and a sample average of the very draws the restriction is imposed on rather
# than the population mean). Under common marginals this is ONE shared value per level, not one per
# origin, so nu0 is length K_mean rather than K_mean*D.
#
# Usage: julia --project=. -t 8 .../production_smoke_cmzc_cross_2026-08-09.jl <K> <W> <budget_s> [design] [resume]
#   pass a 5th argument "resume" to RESUME from the checkpoint the same invocation previously wrote
#   (same OUT dir, not wiped) instead of starting fresh.
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
using Printf, Dates, Statistics
using SpecialFunctions: gamma
lp(xs...) = (println(xs...); flush(stdout))

const KK      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const W       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000
const BUDGET  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 900.0
const DESIGN  = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :sobol_randomized   # driver's own default
const RESUME  = length(ARGS) >= 5 && ARGS[5] == "resume"
const DELTA   = 1.0
const KSTAR   = 2      # = sigma-1 at sigmaHat=3.0 -> Variant D active
const CM_L    = 50
const LABEL   = "cmzc_cross_K$(KK)_W$(W)"
# Run-level (NOT scientific) overrides, 2026-08-10. These change wall-budget bookkeeping and where
# output lands -- never what economic problem is solved -- so an env default is appropriate here and
# does not fall under CLAUDE.md's "no defaults on a scientific parameter" rule. RUN_TAG isolates
# concurrent invocations that share (K, W, design) into their own ckpt_dir instead of colliding.
const RUN_TAG   = get(ENV, "RUN_TAG", "")
const CKPT_INT  = parse(Float64, get(ENV, "CKPT_INTERVAL_S", "60.0"))
const BLAS_THR  = haskey(ENV, "BLAS_THREADS") ? parse(Int, ENV["BLAS_THREADS"]) : nothing
const OUT = joinpath(_D4E, "..", "..", "results", "cmzc_cross_production_smoke_2026-08-09",
                     "K$(KK)_W$(W)_$(DESIGN)" * (isempty(RUN_TAG) ? "" : "_$(RUN_TAG)"))
RESUME || rm(OUT; force = true, recursive = true)
mkpath(OUT)
const CKPT = joinpath(OUT, "$(LABEL)_latest.jls")

lp("=== CM+ZC-CROSS PRODUCTION SMOKE: K_mean=K_pair=$KK  W=$W  budget=$(BUDGET)s  draw_design=$DESIGN  resume=$RESUME ===")
lp("    driver=run_cm_upper_checkpointed  gradient_backend=:cplus (DEFAULT)  Variant D kstar=$KSTAR  L=$CM_L")
lp("    run_tag=", isempty(RUN_TAG) ? "(none)" : RUN_TAG, "  blas_threads=", BLAS_THR === nothing ? "(driver default)" : BLAS_THR,
   "  checkpoint_interval_s=", CKPT_INT, "  julia_threads=", Threads.nthreads(), "  out=", OUT)

SNAPS = nested_grid_sequence([10, 20, 50])
PROBS = SNAPS[CM_L]

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = DESIGN, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx0.D; Ddest = ctx0.D_dest
lp("ctx built. D=$D Ddest=$Ddest sigma=$(ctx0.σ) muHat=$(ctx0.μHat) bi=$(ctx0.bi)")

# ---- w0 in the driver's A_coordinate_mode=:powered_aspace coordinates (its default) ----
pe0    = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0    = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib  = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib  = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)

# ---- eta0: theoretical Gamma(1-mu*k), ONE shared value per level, then drop the Variant D slot ----
layout0   = SharedByPowerCrossLayout(KK, KK)
aml0      = ActiveMeanLayout(layout0, ctx0.bi, KSTAR, D)
nu0_dense = [gamma(1 - ctx0.μHat * k) for k in 1:KK]
@assert length(nu0_dense) == n_eta(layout0)
eta0_active = [log(nu0_dense[d]) for d in 1:length(nu0_dense) if d != aml0.dense_omit_idx]
lp("n_eta dense=$(n_eta(layout0)) -> active=$(aml0.n_eta_active) (omitted dense idx $(aml0.dense_omit_idx) == kstar $(KSTAR))")
lp("nu0 (theoretical Gamma(1-mu*k)) = ", nu0_dense)
lp("w0 length = $(length(w_a_calib) + length(eta0_active))  (econ $(length(w_a_calib)) + eta $(length(eta0_active)))")

if RESUME
    isfile(CKPT) || error("production_smoke_cmzc_cross: resume requested but no checkpoint at $CKPT")
    lp(">> RESUMING from $CKPT")
end

t0 = time()
result = run_cm_upper_checkpointed(RESUME ? nothing : vcat(w_a_calib, eta0_active);
    W = W, delta = DELTA, draw_design = DESIGN, draw_seed = 20260719,
    L = CM_L, contrasts = :orthonormal, probs = PROBS, include_truncated_moment = true,
    cm_extension = :cm_plus_moments, meanzc_K_mean = KK, meanzc_K_pair = KK,
    meanzc_target_layout = :shared_by_power_cross,          # <-- CM+ZC-CROSS
    meanzc_profiled_level = KSTAR,                           # <-- Variant D
    A_coordinate_mode = :powered_aspace,
    inner_lower_limit = -10.0,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    ckpt_dir = OUT, run_id = LABEL, label = LABEL,
    resume_from = RESUME ? CKPT : nothing, blas_threads = BLAS_THR,
    checkpoint_interval_s = CKPT_INT, maxtime_real = BUDGET, verbose = true)
wall = time() - t0

lp("="^95)
@printf("RESULT CM+ZC-CROSS K=%d/%d W=%d design=%s resume=%s: wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
        KK, KK, W, string(DESIGN), string(RESUME), wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
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
lp("  checkpoint written: ", isfile(CKPT), "  -> ", CKPT)
if isfile(CKPT)
    # load_cm_checkpoint (the CM-family chain) -- this driver writes CMCheckpointV11 as of the
    # 2026-08-09 CM+ZC-CROSS schema bump. Field is `checkpoint_reason`, not `stop_reason`.
    c = load_cm_checkpoint(CKPT)
    @printf("  checkpoint reports: schema=%d meanzc_target_layout=%s cm_extension=%s K_mean=%d K_pair=%d n_eta_stored=%d reason=%s\n",
            c.schema, string(c.meanzc_target_layout), string(c.cm_extension), c.meanzc_K_mean, c.meanzc_K_pair,
            length(c.eta_nu), string(c.checkpoint_reason))
    @printf("  checkpoint layout tag round-trips as CM+ZC-CROSS: %s\n", string(c.meanzc_target_layout) == "shared_by_power_cross")
    @printf("  checkpoint eta length matches Variant D active count (%d): %s\n",
            aml0.n_eta_active, length(c.eta_nu) == aml0.n_eta_active)
end
mf = joinpath(OUT, "$(LABEL)_backend_manifest.json")
if isfile(mf)
    fam = match(r"\"family\"\s*:\s*\"([^\"]+)\"", read(mf, String))
    lp("  backend manifest family = ", fam === nothing ? "(not found)" : fam.captures[1],
       "   (must be cm_meanzc_cross, NOT cm_meanzc -- provenance record must not mislabel the family)")
end
lp("="^95)
