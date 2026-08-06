# 2026-08-05 truncated-power task: THE deliverable gate -- calls the REAL production driver
# (run_cm_upper_checkpointed), not a diagnostic/direct archC_base_state call, for the two-family
# (eq.35+eq.36) flexible-CM spec at real D20 data. Confirms Architecture C (winner-bin H_EC +
# T12/T22 H_CC + the :cm_lookup operator FG) actually works end-to-end through the same driver a
# real campaign would use, with NO dense G/H (OperatorPsiBundle) and NO diagnostic opt-in.
const D4X = @__DIR__
cd(D4X)
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
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics, Dates
lp(xs...) = (println(xs...); flush(stdout))

function calib_w0_cm(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    return vcat(x_free_calib[1], vec(a_calib))
end

const W = 100_000   # 2026-08-05: bumped from 20,000 after a genuine nStatus=-300 infeasibility at
# W=20,000 with draw_design=:sobol_randomized -- this repo's own well-documented D20 W-sensitivity
# finding (real-scale feasibility needs W>=~80,000), not a code bug (D4 gates already confirm
# Architecture C is bit-exact vs Architecture A). W=100,000 + :pseudorandom matches
# final_four_family_gate_2026-07-28.jl's own known-good real-production-driver convention exactly.
const DELTA = 1.0
lp("Building real D20 context (W=$W)...")
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
w0 = calib_w0_cm(ctx0, pe0, theta0, xy0)
lp("Context built. D=", ctx0.D, " length(w0)=", length(w0))

const OUT = joinpath("/bbkinghome/edav/repo_scratch/cm-add-truncated-power-moments-2026-08-05/d20_archc_real_ckpt")
rm(OUT; force = true, recursive = true); mkpath(OUT)

lp("="^100)
lp("Calling run_cm_upper_checkpointed: include_truncated_moment=true, cm_extension=:cm_only, W=$W, L=10")
lp("(the REAL production driver -- OperatorPsiBundle, no dense G/H, no diagnostic opt-in)")
lp("="^100)
const L_ = 10
probs_ = collect(range(1 / L_, (L_ - 1) / L_, length = L_))   # explicit cutpoints -- this driver
# requires `probs` explicitly (no re-derive-from-L default), matching this repo's own
# no-implicit-scientific-defaults convention.
t0 = time()
r = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L_, contrasts = :anchored, probs = probs_, include_truncated_moment = true,
    cm_extension = :cm_only, marginal_restriction = :common_flexible,
    ckpt_dir = OUT, run_id = "archc_twofamily_d20", label = "archc_twofamily_d20",
    checkpoint_interval_s = 3600.0, maxtime_real = 90.0, verbose = true)
wall = time() - t0
lp("="^100)
lp("run_cm_upper_checkpointed returned in $(round(wall, digits=1))s")
lp("knitro_status = ", r.knitro_status)
lp("n_eval=", r.n_eval, " n_grad=", r.n_grad)
lp("kappa = ", hasproperty(r, :kappa) ? r.kappa : "n/a")
lp("dense_CM_G materializations this run = ", NO_DENSE_G_COUNTERS[].dense_CM_G_materializations)
lp("="^100)

ok = r.knitro_status in (0, -100, -101, -102, -103)
no_dense_cm_g = NO_DENSE_G_COUNTERS[].dense_CM_G_materializations == 0
println(ok ? "PASS" : "FAIL", "  real production driver call succeeded (knitro_status=$(r.knitro_status))")
println(no_dense_cm_g ? "PASS" : "FAIL", "  zero dense CM-grid G materializations (genuine no-dense-G Architecture C, not a silent fallback)")
if !ok || !no_dense_cm_g
    exit(1)
end
println("ALL D20 PRODUCTION-DRIVER GATES PASS")
