# 2026-08-05: sweep L (quantile bin count) at the real D20 calibration point, checking whether the
# near-collinearity found at L=10 (max corr(CDF,POW)=0.9997 at the extreme l=10/p=0.9 bin --
# test_cm_moment_rank_2026-08-05.jl) is an L=10-specific artifact of very extreme quantile bins, and
# whether a smaller L both improves conditioning AND lets the real two-family KNITRO fixed-state
# solve (archC_base_state) actually converge cleanly instead of stalling at nStatus=-400 (as seen at
# L=10 in test_cm_archc_d20_fixedstate_2026-08-05.jl / test_cm_archc_d20_trace_2026-08-05.jl, ruled
# out as W-sensitivity or iteration-budget by prior tests in this same investigation).
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat)

function rank_and_solve_for_L(L::Int)
    probs_ = collect(range(1 / L, (L - 1) / L, length = L))
    aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored, probs = probs_)
    @assert aug.n_families == 2
    objA = aug.obj_cm
    n = objA.outer_constr_index
    M = size(ctx.U, 1)
    K = zeros(M)
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full_calib, objA.U, objA)
    objA.H[:, 1] .= K
    objA.H[:, 2] .= 1.0

    CM = aug.CM
    ncm_cdf = aug.ncm_cdf
    G_full = Matrix(objA.H[:, 3:n+1])

    sv_cm = svdvals(CM)
    sv_full = svdvals(G_full)
    cond_cm = sv_cm[1] / sv_cm[end]
    cond_full = sv_full[1] / sv_full[end]

    nO = length(aug.origins)
    worst_corr = 0.0
    worst_pair = (0, 0)
    for l in 1:L, oi in 1:nO
        c_cdf = @view CM[:, (l-1)*nO + oi]
        c_pow = @view CM[:, ncm_cdf + (l-1)*nO + oi]
        ρ = abs(cor(c_cdf, c_pow))
        if ρ > worst_corr
            worst_corr = ρ
            worst_pair = (l, oi)
        end
    end

    lp(@sprintf("  L=%2d: ncm_cdf=%3d cond(CM)=%10.2f cond(fullG)=%10.2f worst_corr=%.6f at (l=%d,oi=%d)",
                L, ncm_cdf, cond_cm, cond_full, worst_corr, worst_pair[1], worst_pair[2]))

    # ---- real fixed-state KNITRO solve at this L, same production path as the D20 tests ----
    cctx = build_cm_bin_ctx(ctx, aug)
    cf = cf_build(θ_full_calib, ctx; check_ties = false)
    cctx.core_cf_ref[] = cf

    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs_,
        include_truncated_moment = true,
        moment_representation = :operator, inner_fg_backend = :cm_lookup,
        use_archB_moments = false)
    t0 = time()
    try
        base = archC_base_state(copy(x_free_calib), pcx.ctx_cm, pcx.cctx)
        lp(@sprintf("        KNITRO: inner_status=%d (%.1fs) -- %s", base.inner_status, time() - t0,
                     base.inner_status in (0, -100, -101, -102, -103) ? "PASS" : "FAIL"))
    catch e
        lp(@sprintf("        KNITRO: EXCEPTION after %.1fs -- %s", time() - t0, sprint(showerror, e)[1:min(120,end)]))
    end
end

lp("="^100)
lp("L sweep: condition number, worst CDF/POW correlation, and real KNITRO fixed-state outcome")
lp("="^100)
for L in (3, 4, 5, 6, 8, 10)
    rank_and_solve_for_L(L)
end
lp("="^80)
lp("DONE")
