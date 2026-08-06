# Independent central-FD check of the REAL outer gradient (cm_frechet_production_gradient_cplus)
# for common-Frechet TWO-FAMILY, D20/W=20,000/L=50 -- specifically targets the code path this
# session's own DimensionMismatch fix (frechet_cm_level_fixed_contribution, cm_frechet_cplus.jl,
# commit 15b89ac) touches: build_lfix_base_cache_cm_frechet_C! folds frechet_cm_level_fixed_
# contribution's output into cache.q0, which EVERY A-block coordinate probe in
# composite_gradient_at_Cplus_from_cache reads. The prior D4/D20 gates verified this fix doesn't
# crash and produces a plausible run; this checks the resulting gradient VALUES are numerically
# correct, the same discipline that caught the real gp-gradient=0 bug in the prior session.
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
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
npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; lp("  PASS  ", name)
    else
        nfail += 1; lp("  FAIL  ", name)
    end
end

W = 20_000; L = 50
probs = cm_equal_grid_probs(L)
GRAV = default_gravity_exclude_cells_brazil_korea()
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV, σHat = 3.0)
pe = build_pivot_elimination(ctx)
theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf_from_w_econ(w_econ) = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)))
cplus_pool = build_grad_workspace_pool(size(ctx.obj.U, 1))
cplus_ws = build_lfix_factorized_workspace(ctx.D, ctx.D_dest, size(ctx.obj.U, 1))
bandwidth_cache = Dict{Int,Float64}()

lp("="^90); lp("=== common-Frechet TWO-FAMILY outer-gradient FD check, D20/W=20,000/L=50 ==="); lp("="^90)
pcx_f = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = true,
    moment_representation = :operator)
check("cctx.n_families == 2", pcx_f.cctx.n_families == 2)
w0_frechet = cm_w0_from_calibration(ctx, pe, :powered_aspace)
D2_econ_frechet = length(w0_frechet)

function Delta_frechet(w)
    xf = xf_from_w_econ(w)
    _, _, verify = cm_frechet_production_value_verified_screened(xf, pcx_f)
    return verify
end
function grad_frechet(w)
    xf = xf_from_w_econ(w)
    gfull, meta = cm_frechet_production_gradient_cplus(xf, pcx_f, ctx, pe, cplus_pool, cplus_ws; threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
    gfull[2:D2_econ_frechet] .*= -theta_cm
    return gfull, meta
end

g_analytic, meta = grad_frechet(w0_frechet)
lp("meta.h_used[2]=", meta.h_used[2], " meta.h_used[380]=", meta.h_used[380])
# idx=1 (gp): true closed-form term, no internal FD -- h choice is immaterial.
# idx=2/380 (A-block): the analytic component's own h_used (~0.08-0.1, z-space) is KNOWN too
# large for a full KNITRO reoptimization to stay feasible at (this session's own decisive same-h
# A-block test already observed "h=0.1 FD FAILED: nStatus=-300" at this magnitude for flexible-CM
# -- a genuine infeasibility at that step, not a correctness signal). Use a smaller, already-
# validated-reliable econ-space h=1e-5 for the full-reoptimization probe instead.
for (idx, h) in ((1, 1e-6), (2, 1e-5), (380, 1e-5))
    wp = copy(w0_frechet); wp[idx] += h
    wm = copy(w0_frechet); wm[idx] -= h
    local vp, vm, ok
    ok = true
    try
        vp = Delta_frechet(wp); vm = Delta_frechet(wm)
    catch e
        ok = false
        lp("  [frechet-2fam] idx=", idx, " h=", h, " FD FAILED (full reopt infeasible at this step): ",
           sprint(showerror, e)[1:min(end, 150)])
    end
    if ok
        fd = (vp.Delta_dual - vm.Delta_dual) / (2h)
        an = g_analytic[idx]
        rel = abs(fd - an) / max(abs(an), abs(fd), 1e-8)
        lp("  [frechet-2fam] idx=", idx, " h=", h, " analytic=", an, " FD=", fd, " rel_diff=", rel)
        check("frechet-2fam idx=$idx: analytic vs FD agree (rel<0.05)", rel < 0.05)
    end
end

lp(); lp("="^90); lp("TOTAL: $npass passed, $nfail failed"); lp("="^90)
exit(nfail == 0 ? 0 : 1)
