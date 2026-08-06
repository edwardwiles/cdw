# URGENT follow-up: the FD check (fd_outer_gradient_check_frechet_twofamily_2026-08-06.jl) found
# 18%/171% analytic-vs-reoptimized mismatch at idx=2/idx=380, h=1e-5 -- MUCH worse than flexible-
# CM's own 0.2%/2.8% at the identical h/coordinates. Decisive A-vs-B same-h test (production
# secant vs independent full-rescan, NO reoptimization needed) to determine whether this is a real
# bug in this session's own two-family fix, or something else entirely.
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

lp("="^100); lp("=== A-vs-B same-h test, common-Frechet TWO-FAMILY, idx=2,380, h=1e-5 ==="); lp("="^100)
pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, threaded_bins = true, include_truncated_moment = true,
    moment_representation = :operator)
xf0 = xf_from_w_econ(w0)
base, verify0 = archC_frechet_verified_state(xf0, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
lp("calibration verify0: Delta_dual=", verify0.Delta_dual, " inner_status=", verify0.inner_status)
cache = build_lfix_base_cache_cm_frechet_C!(cplus_ws, xf0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
z0mat = log.(reshape(xf0[2:end], ctx.D, ctx.D_dest))
w0_z = vcat(xf0[1], pivot_reduce(z0mat, pe))

for idx in (2, 380)
    for h in (1e-5, 1e-6)
        h_z = theta_cm * h
        gws = cplus_pool.slots[1]
        A_z = a_block_fd_component_Cplus!(gws, cache, pcx.ctx_cm, pe, w0_z, idx, h_z)
        A_val = -theta_cm * A_z
        wp = copy(w0); wp[idx] += h; wm = copy(w0); wm[idx] -= h
        Bp = delta_full_rescan_generic(cache, pcx.ctx_cm, wp, idx)
        Bm = delta_full_rescan_generic(cache, pcx.ctx_cm, wm, idx)
        B_val = (Bp - Bm) / (2h)
        rel_AB = abs(A_val - B_val) / max(abs(A_val), abs(B_val), 1e-8)
        lp("  idx=", idx, " h=", h, " A(prod)=", A_val, " B(indep,rescan)=", B_val, " rel_AB=", rel_AB)
    end
end
lp("=== DONE ===")
