# Cross-family confirmation of the decisive same-h A-block finding (task section 3.5): repeat ONE
# ordinary-A direction (idx=2) at ONE reliable h (1e-5, the plateau region from the flexible-CM
# control run) for CM+ZC and common-Fréchet, checking A (production, forced h) against B
# (independent full-rescan, no top-3 shortcut) -- the decisive leg (A vs B agreed to ~1e-10-1e-16
# relative for flexible-CM at every h/coordinate tested). Not rerunning the full h-grid/C-reopt
# matrix per the task's own "do not rerun a giant matrix unless the control test fails" guidance.
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
using Printf, LinearAlgebra
lp(xs...) = (println(xs...); flush(stdout))

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
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)

const IDX = 2
const H = 1e-5
const HZ = theta_cm * H

function delta_full_rescan_generic(cache, ctx_cm, w_econ_pert::AbstractVector, coord_idx::Int)
    xf = xf_from_w_econ(w_econ_pert)
    θ_full = CS.reconstruct_full(xf, ctx_cm.m)
    D = cache.D; Ddest = cache.Ddest; σ = cache.σ
    Wn = cache.W
    _, logCCp, _ = constCons_matrix(θ_full, ctx_cm)
    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache.baseIndex && d == cache.baseIndex, cells)
    q = copy(cache.q0)
    contrib_new = Vector{Float64}(undef, Wn)
    for d in affected_dests
        @inbounds for ω in 1:Wn
            bo = 1; bs = logCCp[1, d] + cache.ref.mulU[ω, 1]
            for o in 2:D
                v = logCCp[o, d] + cache.ref.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
            pTσ_wo = pTσ_from_score(bs, σ)
            d1w = d + (bo - 1) * Ddest
            contrib_new[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
        end
        q .-= contrib_new .- @view(cache.contrib0[:, d])
    end
    if cf_touched
        cf_new = Vector{Float64}(undef, Wn)
        cf_contrib_at_C!(cf_new, cache, θ_full, ctx_cm)
        q .-= cf_new .- cache.cf_contrib0
    end
    psi = similar(q)
    CS.Psi!(psi, q)
    return -(sum(psi) / Wn + cache.ζstar)
end

lp("="^100); lp("=== CROSS-FAMILY same-h confirmation: idx=", IDX, " h=", H, " (z-space h=", HZ, ") ==="); lp("="^100)

# ---- CM+ZC (K_mean=1/K_pair=1, current merged production spec) ----
lp("-"^100); lp("CM+ZC (K_mean=1, K_pair=1)")
K_mean = 1; K_pair = 1
pcx_zc = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored,
    include_truncated_moment = true, meanzc_basis = :direct, probs = probs, moment_representation = :operator)
nu1_guess = sum(pcx_zc.aug.Zraw_all[1]) / length(pcx_zc.aug.Zraw_all[1])
νvec0 = [nu1_guess]
xf0_zc = xf_from_w_econ(w0)
base_zc, verify0_zc = archC_meanzc_verified_state(xf0_zc, νvec0, pcx_zc.ctx_cm, pcx_zc.cctx)
cache_zc = build_lfix_base_cache_cm_meanzc_C!(cplus_ws, xf0_zc, pcx_zc.ctx_cm, base_zc, ctx, pcx_zc.aug, pcx_zc.bins, νvec0)
lp("  calibration verify0_zc: Delta_dual=", verify0_zc.Delta_dual, " inner_status=", verify0_zc.inner_status)
z0mat_zc = log.(reshape(xf0_zc[2:end], ctx.D, ctx.D_dest))
w0_z_zc = vcat(xf0_zc[1], pivot_reduce(z0mat_zc, pe))
gws_zc = cplus_pool.slots[1]
A_z_zc = a_block_fd_component_Cplus!(gws_zc, cache_zc, pcx_zc.ctx_cm, pe, w0_z_zc, IDX, HZ)
A_zc = -theta_cm * A_z_zc
wp = copy(w0); wp[IDX] += H; wm = copy(w0); wm[IDX] -= H
Bp_zc = delta_full_rescan_generic(cache_zc, pcx_zc.ctx_cm, wp, IDX)
Bm_zc = delta_full_rescan_generic(cache_zc, pcx_zc.ctx_cm, wm, IDX)
B_zc = (Bp_zc - Bm_zc) / (2H)
rel_zc = abs(A_zc - B_zc) / max(abs(A_zc), abs(B_zc), 1e-8)
lp("  [cmzc] idx=", IDX, " h=", H, " A(prod,forced)=", A_zc, " B(indep,rescan)=", B_zc, " rel_AB=", rel_zc)

# ---- common-Fréchet (single-family, cdf_only, current merged production spec) ----
lp("-"^100); lp("common-Fréchet (single-family, cdf_only)")
pcx_f = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = false,
    moment_representation = :operator)
xf0_f = xf_from_w_econ(w0)
base_f, verify0_f = archC_frechet_verified_state(xf0_f, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
cache_f = build_lfix_base_cache_cm_frechet_C!(cplus_ws, xf0_f, pcx_f.ctx_cm, base_f, ctx, pcx_f.aug, pcx_f.bins)
lp("  calibration verify0_f: Delta_dual=", verify0_f.Delta_dual, " inner_status=", verify0_f.inner_status)
z0mat_f = log.(reshape(xf0_f[2:end], ctx.D, ctx.D_dest))
w0_z_f = vcat(xf0_f[1], pivot_reduce(z0mat_f, pe))
gws_f = cplus_pool.slots[1]
A_z_f = a_block_fd_component_Cplus!(gws_f, cache_f, pcx_f.ctx_cm, pe, w0_z_f, IDX, HZ)
A_f = -theta_cm * A_z_f
Bp_f = delta_full_rescan_generic(cache_f, pcx_f.ctx_cm, wp, IDX)
Bm_f = delta_full_rescan_generic(cache_f, pcx_f.ctx_cm, wm, IDX)
B_f = (Bp_f - Bm_f) / (2H)
rel_f = abs(A_f - B_f) / max(abs(A_f), abs(B_f), 1e-8)
lp("  [frechet] idx=", IDX, " h=", H, " A(prod,forced)=", A_f, " B(indep,rescan)=", B_f, " rel_AB=", rel_f)

lp("=== DONE ===")
