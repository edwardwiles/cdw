# Decompose the A-block gradient discrepancy per user's framing: (1) is it present in flexible_cm
# (not just CM+ZC/Frechet)? (2) is the mismatch coming from the exact intensive-margin
# (incumbent-winner, price-only) piece, or the winner-switching piece? Real D20/W=20,000/L=50,
# cm_gradient_backend=:cplus (no dense G/H) throughout, matching production.
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
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)

lp("="^90); lp("=== TEST 1: flexible_cm (plain CM, no ZC/Frechet extension) -- is the A-block gap present here too? ==="); lp("="^90)
pcx_flex = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs, threaded_bins = true,
    include_truncated_moment = false, moment_representation = :operator)
xf0 = xf_from_w_econ(w0)
D2_econ = length(w0)

function Delta_flex(w)
    xf = xf_from_w_econ(w)
    _, _, verify = cm_production_value_verified_screened(xf, pcx_flex)
    return verify.Delta_dual
end
gfull_flex, meta_flex = cm_production_gradient_cplus(xf0, pcx_flex, ctx, pe, cplus_pool, cplus_ws; threaded = true,
    h_mode = :cached, bandwidth_cache = bandwidth_cache)
gfull_flex[2:D2_econ] .*= -theta_cm
lp("  meta_flex.h_used[2]=", meta_flex.h_used[2], " meta_flex.h_used[380]=", meta_flex.h_used[380],
   " (the analytic computation's OWN internally-selected bandwidth -- note if these are LARGE, a full")
lp("  re-solve at that same h is expected to be infeasible; the internal method is cheap/envelope-based")
lp("  and does not need full-problem feasibility at that h, unlike an external full-resolve FD probe)")
for idx in (1, 2, 380), h in (1e-6, 1e-3, 1e-1)
    try
        wp = copy(w0); wp[idx] += h; wm = copy(w0); wm[idx] -= h
        fd = (Delta_flex(wp) - Delta_flex(wm)) / (2h)
        an = gfull_flex[idx]
        rel = abs(fd - an) / max(abs(an), abs(fd), 1e-8)
        lp("  [flex_cm] idx=", idx, " h=", h, " analytic=", an, " FD(full-resolve)=", fd, " rel_diff=", rel)
    catch e
        lp("  [flex_cm] idx=", idx, " h=", h, " FD FAILED: ", sprint(showerror, e)[1:min(end,150)])
    end
end

lp("="^90); lp("=== TEST 2: winner-switch count at idx=2 and idx=380, using the analytic computation's own h ==="); lp("="^90)
# Reproduce build_lfix_base_cache_C!'s own winner reference (ref.winner) and directly count how
# many of the W draws change winner at the destinations affected by each coordinate, at h_used[idx].
base_flex = archC_verified_state(xf0, pcx_flex.ctx_cm, pcx_flex.cctx)[1]
cache_flex = build_lfix_base_cache_C!(cplus_ws, xf0, pcx_flex.ctx_cm, base_flex)
for idx in (2, 380), h in (1e-6, 1e-3, 1e-1, meta_flex.h_used[idx])
    cells = affected_cells(pe, idx)
    affected_dests = unique(last.(cells))
    lp("  idx=", idx, " h=", h, " affected_cells=", length(cells), " affected_dests=", affected_dests)
    for new_val in (w0[idx] + h, w0[idx] - h)
        w = copy(w0); w[idx] = new_val
        z = pivot_expand(w[2:end], pe)
        Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, pcx_flex.ctx_cm.m)
        _, logCC_new, _ = constCons_matrix(θ_full, pcx_flex.ctx_cm)
        n_switch = 0
        for d in affected_dests
            for ω in 1:W
                r1 = cache_flex.ref.winner[ω, d]
                bo = 1; bs = logCC_new[1, d] + cache_flex.ref.mulU[ω, 1]
                for o in 2:ctx.D
                    v = logCC_new[o, d] + cache_flex.ref.mulU[ω, o]
                    v < bs && (bs = v; bo = o)
                end
                bo != r1 && (n_switch += 1)
            end
        end
        lp("    new_val=", new_val, " (delta=", new_val - w0[idx], ")  n_winner_switches=", n_switch,
           " / ", length(affected_dests) * W, " (dest,draw) pairs scanned")
    end
end

lp("="^90); lp("=== TEST 3: tiny-h (h=1e-10, guaranteed no winner switch) intensive-margin-only check ==="); lp("="^90)
for idx in (2, 380)
    h = 1e-10
    wp = copy(w0); wp[idx] += h; wm = copy(w0); wm[idx] -= h
    fd = (Delta_flex(wp) - Delta_flex(wm)) / (2h)
    an = gfull_flex[idx]
    rel = abs(fd - an) / max(abs(an), abs(fd), 1e-8)
    lp("  [flex_cm tiny-h] idx=", idx, " h=", h, " analytic=", an, " FD(full-resolve, tiny h)=", fd, " rel_diff=", rel)
end
lp("=== DONE ===")
