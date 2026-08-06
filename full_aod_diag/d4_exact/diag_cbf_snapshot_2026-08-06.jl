# Section-5 diagnostic (cm-meanzc-frechet-outer-production-closeout-2026-08-06): instrument the
# ACTUAL first cb_F! call inside the real public run_cm_upper_checkpointed driver at real
# production settings (sigma=3, W=100000, L=50, draw_design=:sobol_randomized, draw_seed=20260719,
# destination_sample=:exclude_row, exclude_diagonal_gravity=true, Brazil-Korea gravity exclusion,
# cm_extension=:cm_plus_moments with K_mean=1/K_pair=1 per configs/fullA_production_2026-08-03.toml),
# freeze an exact snapshot right before cm_screen_precheck! runs, then run the screen two ways on
# the frozen inputs: (A) the exact production call path already run inside cb_F!, (B) a direct
# pure-call replica using CM_LIVE_PCX_STASH[]'s pcx.ctx_cm (the SAME object cb_F! itself used).
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf
lp(xs...) = (println(xs...); flush(stdout))

lp("=== build production ctx (real settings) ===")
ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)
pe = build_pivot_elimination(ctx)

L = 50
probs = cm_equal_grid_probs(L)
K_mean = 1; K_pair = 1
nu_bounds = meanzc_default_nu_bounds(ctx, K_mean)
lp("=== computing a genuine calibration-consistent nu0 guess (verify4 method: mean of aug.Zraw_all[1]) ===")
pcx_nu0 = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
                                              include_truncated_moment = true, meanzc_basis = :direct, probs = probs,
                                              moment_representation = :operator)
nu1_guess = sum(pcx_nu0.aug.Zraw_all[1]) / length(pcx_nu0.aug.Zraw_all[1])
lp("nu1_guess = ", nu1_guess, "  log(nu1_guess) = ", log(nu1_guess))
eta0 = [log(nu1_guess)]
w0 = vcat(cm_w0_from_calibration(ctx, pe, :powered_aspace), eta0)
lp("length(w0) = ", length(w0))

CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
CKPT = mktempdir()
lp("=== calling run_cm_upper_checkpointed (real public driver) ===")
local result
try
    result = run_cm_upper_checkpointed(w0; W = 100_000, delta = 1.0,
        draw_design = :sobol_randomized, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs,
        include_truncated_moment = true,
        cm_hessian_backend = :structured,
        marginal_restriction = :common_flexible,
        cm_extension = :cm_plus_moments, meanzc_K_mean = K_mean, meanzc_K_pair = K_pair,
        meanzc_nu_bounds = nu_bounds, meanzc_basis = :direct,
        exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
        σHat = 3.0, destination_sample = :exclude_row,
        A_coordinate_mode = :powered_aspace,
        ckpt_dir = CKPT, run_id = "diagcbf", label = "diagcbf",
        checkpoint_interval_s = 600.0, maxtime_real = 60.0, verbose = true)
    lp("=== driver returned (no throw): knitro_status=", result.knitro_status, " n_eval=", result.n_eval)
catch e
    lp("=== driver threw: typeof=", typeof(e), " msg=", sprint(showerror, e)[1:min(end, 500)])
end

pcx_live = CM_LIVE_PCX_STASH[]
lp("=== CM_LIVE_PCX_STASH[] === isnothing=", pcx_live === nothing)
if pcx_live !== nothing
    ctx_cm = pcx_live.ctx_cm
    lp("typeof(ctx_cm) = ", typeof(ctx_cm))
    lp("ctx_cm.D=", ctx_cm.D, " ctx_cm.D_dest=", hasproperty(ctx_cm, :D_dest) ? ctx_cm.D_dest : ctx_cm.D)
    lp("ctx_cm.pairwise === nothing : ", ctx_cm.pairwise === nothing)

    # Reconstruct EXACTLY what cb_F!'s first call computed, using the driver's own decode fns.
    theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
    xf_from_w_econ(w_econ) = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe)
    D2_econ = length(w0) - K_mean
    xf = xf_from_w_econ(w0[1:D2_econ])
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    lp("max|xf - x_free_calib| = ", maximum(abs.(xf .- x_free_calib)),
       "  max relative = ", maximum(abs.(xf .- x_free_calib) ./ max.(abs.(x_free_calib), 1e-300)))

    θ_full = CS.reconstruct_full(xf, ctx_cm.m)
    Pmat = target_shares(ctx_cm)
    a = compute_a_od(θ_full, ctx_cm)
    lp("θ_full hash=", hash(θ_full), " length=", length(θ_full))
    lp("Pmat hash=", hash(Pmat))
    lp("a hash=", hash(a), " extrema=", extrema(a))

    lp("=== (B) direct pure-call replica: pairwise_certificate on frozen (a, ctx_cm.pairwise, Pmat) ===")
    presB = pairwise_certificate(a, ctx_cm.pairwise, Pmat)
    lp("B: infeasible=", presB.infeasible, " worst=(o=", presB.worst_o, ",d=", presB.worst_d, ",k=", presB.worst_k, ") slack=", presB.worst_slack)

    lp("=== (A) direct pure-call replica via cm_screen_precheck! itself on frozen xf ===")
    try
        cm_screen_precheck!(xf, ctx_cm)
        lp("A: PASSED (no exception)")
    catch e2
        lp("A: THREW ", typeof(e2), " ", sprint(showerror, e2))
    end

    lp("=== repeat (B) a second time on a FRESH copy of xf (mutation-detection) ===")
    xf2 = copy(xf)
    θ_full2 = CS.reconstruct_full(xf2, ctx_cm.m)
    a2 = compute_a_od(θ_full2, ctx_cm)
    lp("max|a - a2| = ", maximum(abs.(a .- a2)))
    presB2 = pairwise_certificate(a2, ctx_cm.pairwise, Pmat)
    lp("B2: infeasible=", presB2.infeasible, " worst=(o=", presB2.worst_o, ",d=", presB2.worst_d, ")")

    if presB.infeasible
        o = presB.worst_o; d = presB.worst_d; k = presB.worst_k
        lp("=== certificate detail for worst pair (o=", o, ",d=", d, ",k=", k, ") ===")
        lp("Pmat[o,d]=", Pmat[o, d], "  a[o,d]=", a[o, d], "  a[k,d]=", a[k, d], "  M[o,k]=", ctx_cm.pairwise.M[o, k])
        lp("slack = M[o,k] - (a[k,d]-a[o,d]) = ", ctx_cm.pairwise.M[o, k] - (a[k, d] - a[o, d]))
    end

    lp("=== control: build a FRESH flexible_cm ctx_cm via build_cm_production_context, SAME ctx, screen SAME xf ===")
    pcx_flex = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
        include_truncated_moment = true, moment_representation = :operator)
    lp("pcx_flex.ctx_cm.pairwise === pcx_live.ctx_cm.pairwise : ", pcx_flex.ctx_cm.pairwise === pcx_live.ctx_cm.pairwise)
    try
        cm_screen_precheck!(xf, pcx_flex.ctx_cm)
        lp("flexible_cm control: PASSED (no exception)")
    catch e3
        lp("flexible_cm control: THREW ", typeof(e3), " ", sprint(showerror, e3))
    end
else
    lp("pcx_live is nothing -- driver did not reach context construction (see thrown/returned message above)")
end
lp("=== DONE ===")
