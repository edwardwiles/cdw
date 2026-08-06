# Decisive SAME-h A-block gradient comparison (2026-08-06 task, section 3).
#
# The prior session's own diag_ablock_decompose_2026-08-06.jl compared the production analytic
# A-block component (computed once, at its OWN internally-selected bandwidth h~0.08-0.10 in
# z-space) against a full-reoptimized FD secant at DIFFERENT h values (1e-6, 1e-3, 1e-1 in econ
# powered-a-space) -- never recomputing the analytic side at the FD's own h. This script fixes
# that: it forces the SAME econ-space h for all of:
#   A. production fixed-dual secant (a_block_fd_component_Cplus!, forced h, not auto-selected)
#   B. an INDEPENDENT fixed-dual secant, using DENSE obj.moments! (no top-3/dest_contrib shortcut,
#      no shared code with A) at the exact same decoded points -- checks A's own implementation
#      and the a<->z coordinate decode independently.
#   C. a fully reoptimized secant (real verified KNITRO resolve at w0+-h*e_idx)
#   D. the production DEFAULT component (h_mode=:cached, its own auto-selected h) for reference.
# All four are reported in ECON (powered-a-space) units, the actual outer KNITRO coordinate.
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

pcx_flex = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs, threaded_bins = true,
    include_truncated_moment = false, moment_representation = :operator)
xf0 = xf_from_w_econ(w0)
D2_econ = length(w0)

# fixed dual/cache built ONCE at xf0 -- shared by A and B (the "fixed-dual" object both estimate).
base_flex, verify0 = archC_verified_state(xf0, pcx_flex.ctx_cm, pcx_flex.cctx)
cache_flex = build_lfix_base_cache_C!(cplus_ws, xf0, pcx_flex.ctx_cm, base_flex)
lp("Calibration point verify0: Delta_dual=", verify0.Delta_dual, " inner_status=", verify0.inner_status,
   " primal_dual_gap=", verify0.primal_dual_gap, " max_abs_moment_kkt_resid=", verify0.max_abs_moment_kkt_resid)

# internal z-space w0 (SAME index convention as econ w0 -- both ordered by pe.other_idx position;
# cm_z_from_a is a per-coordinate-diagonal affine map z[k] = -theta*(a[k]+logX[k]) - logY[k], so
# econ index idx <-> z-space index idx exactly, no permutation).
z0mat = log.(reshape(xf0[2:end], ctx.D, ctx.D_dest))
w0_z = vcat(xf0[1], pivot_reduce(z0mat, pe))

Wn = cache_flex.W
# NOTE: OperatorPsiBundle (the real production obj type under moment_representation=:operator)
# has NO `.d`/`.moments!`/`H` field by design (operator_psi_bundle.jl) -- the dense
# obj.moments!-based "independent" check from build_lfix_base_cache_C!'s own validate_dense branch
# is structurally incompatible with the production context and cannot be reused here. Instead, B
# is an independent FULL-RESCAN winner determination (brute-force argmin over all D origins, for
# EVERY destination, no top-3 cache, no changed-origins shortcut) -- genuinely separate code from
# dest_contrib_incremental_top3_C!'s own top-3/changed-origin machinery, while still reusing the
# already-validated cheap pieces (CONST_d, lambda*, SW, gammafac, cf_contrib_at_C!, q0) that are
# NOT under question here (only the top-3 shortcut itself is what this check targets).
function delta_full_rescan(w_econ_pert::AbstractVector, coord_idx::Int)
    xf = xf_from_w_econ(w_econ_pert)
    θ_full = CS.reconstruct_full(xf, pcx_flex.ctx_cm.m)
    D = cache_flex.D; Ddest = cache_flex.Ddest; σ = cache_flex.σ
    _, logCCp, _ = constCons_matrix(θ_full, pcx_flex.ctx_cm)
    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache_flex.baseIndex && d == cache_flex.baseIndex, cells)
    q = copy(cache_flex.q0)
    contrib_new = Vector{Float64}(undef, Wn)
    for d in affected_dests
        @inbounds for ω in 1:Wn
            bo = 1; bs = logCCp[1, d] + cache_flex.ref.mulU[ω, 1]
            for o in 2:D
                v = logCCp[o, d] + cache_flex.ref.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
            pTσ_wo = pTσ_from_score(bs, σ)
            d1w = d + (bo - 1) * Ddest
            contrib_new[ω] = (cache_flex.SW[ω] / cache_flex.gammafac) * (cache_flex.CONST_d[d] + cache_flex.λstar[d1w] * pTσ_wo)
        end
        q .-= contrib_new .- @view(cache_flex.contrib0[:, d])
    end
    if cf_touched
        cf_new = Vector{Float64}(undef, Wn)
        cf_contrib_at_C!(cf_new, cache_flex, θ_full, pcx_flex.ctx_cm)
        q .-= cf_new .- cache_flex.cf_contrib0
    end
    psi = similar(q)
    CS.Psi!(psi, q)
    return -(sum(psi) / Wn + cache_flex.ζstar)
end

function delta_reopt(w_econ_pert::AbstractVector)
    xf = xf_from_w_econ(w_econ_pert)
    _, _, verify = cm_production_value_verified_screened(xf, pcx_flex)
    return verify
end

hs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
idxs = [2, 380, 200]

lp("="^100)
lp("=== DECISIVE SAME-h A-block comparison, flexible-CM control, D20/W=20,000/L=50 ===")
lp("="^100)
for idx in idxs
    lp("-"^100); lp("coordinate idx=", idx)
    # D. production default (auto-selected h, h_mode=:cached) -- single reference value
    gD, metaD = cm_production_gradient_cplus(xf0, pcx_flex, ctx, pe, cplus_pool, cplus_ws;
        base = base_flex, verify = verify0, threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    D_val = -theta_cm * gD[idx]
    lp("  D. production DEFAULT: h_used(z-space)=", metaD.h_used[idx], " value(econ)=", D_val)

    for h in hs
        h_z = theta_cm * h
        # A: production routine, forced to this exact h (z-space magnitude h_z)
        gwsA = cplus_pool.slots[1]
        A_z = a_block_fd_component_Cplus!(gwsA, cache_flex, pcx_flex.ctx_cm, pe, w0_z, idx, h_z)
        A_val = -theta_cm * A_z

        # B: independent dense fixed-dual secant at the SAME econ h, exact production decode chain
        wp = copy(w0); wp[idx] += h
        wm = copy(w0); wm[idx] -= h
        Bp = delta_full_rescan(wp, idx); Bm = delta_full_rescan(wm, idx)
        B_val = (Bp - Bm) / (2h)

        # C: fully reoptimized secant (real verified KNITRO resolve)
        local C_val, Cp_res, Cm_res, C_ok
        C_ok = true
        try
            vp = delta_reopt(wp); vm = delta_reopt(wm)
            C_val = (vp.Delta_dual - vm.Delta_dual) / (2h)
            Cp_res = vp; Cm_res = vm
        catch e
            C_ok = false
            C_val = NaN
            lp("    h=", h, " C FAILED: ", sprint(showerror, e)[1:min(end, 160)])
        end

        rel_AB = abs(A_val - B_val) / max(abs(A_val), abs(B_val), 1e-8)
        if C_ok
            rel_AC = abs(A_val - C_val) / max(abs(A_val), abs(C_val), 1e-8)
            rel_BC = abs(B_val - C_val) / max(abs(B_val), abs(C_val), 1e-8)
            unc_bound = 2h * max(abs(A_val), abs(C_val))
            lp("  h=", h, " A(prod,forced)=", A_val, " B(dense,indep)=", B_val, " C(reopt)=", C_val,
                " rel_AB=", rel_AB, " rel_AC=", rel_AC, " rel_BC=", rel_BC,
                " Cp_gap=", Cp_res.primal_dual_gap, " Cm_gap=", Cm_res.primal_dual_gap,
                " Cp_kkt=", Cp_res.max_abs_moment_kkt_resid, " Cm_kkt=", Cm_res.max_abs_moment_kkt_resid,
                " 2h|deriv|~", unc_bound)
        else
            lp("  h=", h, " A(prod,forced)=", A_val, " B(dense,indep)=", B_val, " rel_AB=", rel_AB, " C=FAILED")
        end
    end
end
lp("=== DONE ===")
