# Phase 10 verifier audit (2026-08-02): extends test_operator_verification_cm_frechet.jl's D=4
# gate with the two genuine gaps found in the checklist audit -- (1) per-block KKT residual
# breakdown (kkt_resid_E/kkt_resid_cm/kkt_resid_level/france_ratio_resid, now returned by
# _verify_inner_solution_operator_cm_core via verify_inner_solution_operator_cm_frechet!) and (2)
# recovered full factual shares (verify_recovered_full_factual_shares, operator_verification.jl).
# Mirrors test_operator_verification_cm_frechet.jl's own setup exactly -- purely additive, does not
# modify that file.
const D4X = @__DIR__
for f in ["context.jl","draw_design.jl","winners.jl","oracle.jl",
          "common_marginals_moments.jl","common_marginals_interval.jl","instrumentation.jl","oracle_fast.jl",
          "gravity_elimination.jl","three_way_derivatives.jl","lfix_incremental.jl",
          "composite_gradient.jl","composite_gradient_fast.jl","gradient_workspace.jl","shared_a_gradient.jl",
          "cm_lookup_kernels.jl","lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "nested_quantile_grids.jl","lfix_factorized.jl","lfix_factorized_workspace.jl","lfix_cm_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_lookup_kernels.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl",
          "cm_frechet_lookup_production.jl","cm_frechet_cplus.jl",
          "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

all_pass = true
println("=== D=4 common-Frechet EXTENDED operator verification gate (Phase 10 audit) ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

for L in (10, 50)
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, cm_hessian_backend = :structured)
    base, verify = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    global all_pass &= verify.inner_status == 0
    @printf("  L=%d: dense verify status=%d\n", L, verify.inner_status)

    cctx = pcx.cctx
    cf = cctx.core_cf_ref[]
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    W = size(ctx.U, 1)
    obj = pcx.ctx_cm.obj
    ov = verify_inner_solution_operator_cm_frechet!(base.ζstar, base.λstar, cf, cctx.L, cctx.nO, cctx.origins,
        cctx.refIndex1, bins_u, cctx.R, pcx.aug.level_targets, obj, W)

    # --- (1) per-block KKT residual breakdown ---
    ok_E = ov.kkt_resid_E < 1e-6
    ok_cm = ov.kkt_resid_cm < 1e-6
    ok_level = ov.kkt_resid_level !== nothing && ov.kkt_resid_level < 1e-6
    ok_france = isfinite(ov.france_ratio_resid) && ov.france_ratio_resid < 1e-6
    block_max = max(ov.kkt_resid_E, ov.kkt_resid_cm, ov.kkt_resid_level)
    ok_agree = abs(block_max - ov.kkt_resid) < 1e-12
    global all_pass &= ok_E && ok_cm && ok_level && ok_france && ok_agree
    @printf("  L=%d  kkt_E=%.3e kkt_cm=%.3e kkt_level=%.3e france=%.3e agree=%s\n",
            L, ov.kkt_resid_E, ov.kkt_resid_cm, ov.kkt_resid_level, ov.france_ratio_resid, ok_agree)

    # --- (2) recovered full factual shares ---
    m_weights, _ = verify_namedtuple_from_operator(ov, obj, W, verify.inner_status)
    rec = verify_recovered_full_factual_shares(base.θ_full0, ctx, cf, m_weights)
    ok_winner = rec.max_winner_mismatch == 0
    ok_ratio = rec.max_share_ratio_diff < 1e-6
    global all_pass &= ok_winner && ok_ratio
    @printf("  L=%d  winner_mismatch=%d share_ratio_diff=%.3e\n", L, rec.max_winner_mismatch, rec.max_share_ratio_diff)
end

println()
println(all_pass ? "ALL PASS" : "SOME FAILURES")
all_pass || error("common-Frechet EXTENDED operator verification d4 gate FAILED")
