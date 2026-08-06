# Section 8 gate (cm-meanzc-frechet-outer-production-closeout-2026-08-06): central finite-
# difference check of the ACTUAL outer constraint value (Delta_dual) and its analytic gradient,
# replicating cb_F!/cb_G!'s exact bodies (cm_checkpoint.jl) -- same xf_from_w_econ decode, same
# production gradient functions (cm_meanzc_production_gradient_cplus /
# cm_frechet_production_gradient_cplus), same A-coordinate rescale -- rather than a new gradient
# formula. Real D20/W=20,000/L=50 production settings.
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
xf_from_w_econ(w_econ) = x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe)
cplus_pool = build_grad_workspace_pool(size(ctx.obj.U, 1))
cplus_ws = build_lfix_factorized_workspace(ctx.D, ctx.D_dest, size(ctx.obj.U, 1))
bandwidth_cache = Dict{Int,Float64}()
D2 = 1 + ctx.D * ctx.D_dest - 1   # gp + nonpivot A block, matches D2_econ convention (length(w0))

function fd_check(label, Delta_fn, grad_fn_meta, w0::Vector{Float64}, idxs::Vector{Int}; h_default = 1e-6)
    g_analytic, meta = grad_fn_meta(w0)
    for idx in idxs
        # feedback-fd-bandwidth-mismatch-looks-like-a-bug: the A-block analytic gradient
        # (a_block_fd_component_Cplus!) is ITSELF an internal adaptive-bandwidth FD scheme --
        # comparing it against an external FD probe at a DIFFERENT (fixed) h produces a false
        # gap. Match h to the SAME per-coordinate bandwidth the analytic computation itself
        # selected (meta.h_used[idx]) whenever available (idx>=2); index 1 (gp) is a true closed-
        # form analytic term (gamma_component_analytic, no internal FD at all) so h_default is
        # used there and the choice of h is immaterial.
        h = (idx >= 2 && idx <= length(meta.h_used) && meta.h_used[idx] > 0) ? meta.h_used[idx] : h_default
        wp = copy(w0); wp[idx] += h
        wm = copy(w0); wm[idx] -= h
        Dp = Delta_fn(wp); Dm = Delta_fn(wm)
        fd = (Dp - Dm) / (2h)
        an = g_analytic[idx]
        rel = abs(fd - an) / max(abs(an), abs(fd), 1e-8)
        lp("  [", label, "] idx=", idx, " h=", h, " analytic=", an, " FD=", fd, " rel_diff=", rel)
        check("$label idx=$idx: analytic vs FD agree (rel<0.05)", rel < 0.05)
    end
end

lp("="^90); lp("=== CM+ZC outer-gradient FD check, D20/W=20,000/L=50 ==="); lp("="^90)
K_mean = 1; K_pair = 1
nu_bounds = meanzc_default_nu_bounds(ctx, K_mean)
pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
    include_truncated_moment = true, meanzc_basis = :direct, probs = probs, moment_representation = :operator)
nu1_guess = sum(pcx.aug.Zraw_all[1]) / length(pcx.aug.Zraw_all[1])
w0_cmzc = vcat(cm_w0_from_calibration(ctx, pe, :powered_aspace), [log(nu1_guess)])
D2_econ_cmzc = length(w0_cmzc) - K_mean

function Delta_cmzc(w)
    xf = xf_from_w_econ(w[1:D2_econ_cmzc])
    νvec = exp.(w[D2_econ_cmzc+1:end])
    _, _, verify = cm_meanzc_production_value_verified_screened(xf, νvec, pcx)
    return verify.Delta_dual
end
function grad_cmzc(w)
    xf = xf_from_w_econ(w[1:D2_econ_cmzc])
    νvec = exp.(w[D2_econ_cmzc+1:end])
    gfull, meta = cm_meanzc_production_gradient_cplus(xf, νvec, pcx, ctx, pe, cplus_pool, cplus_ws; threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
    gfull[2:D2_econ_cmzc] .*= -theta_cm
    return gfull, meta
end
# directions: gp (1), two A-coordinates (2, D2_econ_cmzc), the eta_nu coordinate (end)
fd_check("cmzc", Delta_cmzc, grad_cmzc, w0_cmzc, [1, 2, D2_econ_cmzc, length(w0_cmzc)])
d1_cf_cmzc = ctx.D * ctx.D_dest + 1
lp("  [cmzc] obj.outer_constr_index=", pcx.ctx_cm.obj.outer_constr_index, " d1_cf=", d1_cf_cmzc,
   " lambda_cf_condition(oci-1>=d1_cf)=", pcx.ctx_cm.obj.outer_constr_index - 1 >= d1_cf_cmzc)

lp("="^90); lp("=== common-Frechet outer-gradient FD check, D20/W=20,000/L=50 ==="); lp("="^90)
pcx_f = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = false,
    moment_representation = :operator)
w0_frechet = cm_w0_from_calibration(ctx, pe, :powered_aspace)
D2_econ_frechet = length(w0_frechet)

function Delta_frechet(w)
    xf = xf_from_w_econ(w)
    _, _, verify = cm_frechet_production_value_verified_screened(xf, pcx_f)
    return verify.Delta_dual
end
function grad_frechet(w)
    xf = xf_from_w_econ(w)
    gfull, meta = cm_frechet_production_gradient_cplus(xf, pcx_f, ctx, pe, cplus_pool, cplus_ws; threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
    gfull[2:D2_econ_frechet] .*= -theta_cm
    return gfull, meta
end
fd_check("frechet", Delta_frechet, grad_frechet, w0_frechet, [1, 2, D2_econ_frechet])
d1_cf_frechet = ctx.D * ctx.D_dest + 1
lp("  [frechet] obj.outer_constr_index=", pcx_f.ctx_cm.obj.outer_constr_index, " d1_cf=", d1_cf_frechet,
   " lambda_cf_condition(oci-1>=d1_cf)=", pcx_f.ctx_cm.obj.outer_constr_index - 1 >= d1_cf_frechet)

lp(); lp("="^90); lp("TOTAL: $npass passed, $nfail failed"); lp("="^90)
exit(nfail == 0 ? 0 : 1)
