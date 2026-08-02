# Phase 10 verifier audit (2026-08-02): extends test_operator_verification_originzc.jl's D=4 gate
# with the two genuine gaps found in the checklist audit -- (1) per-block KKT residual breakdown
# (kkt_resid_E/kkt_resid_mean/kkt_resid_pair/france_ratio_resid, now returned by
# verify_inner_solution_operator_originzc!) and (2) recovered full factual shares
# (verify_recovered_full_factual_shares, operator_verification.jl). Mirrors
# test_operator_verification_originzc.jl's own setup exactly (same includes/ctx/pcx construction) --
# this file is purely additive, does not modify that file.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 0), (1, 1), (2, 2)]
    layout = OriginByPowerLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
    _, base, verify = cm_originzc_production_value_verified(x_free_calib, νfull0, pcx)
    check("K=$K_mean/$K_pair: dense verify succeeded", verify.inner_status == 0)

    octx = pcx.ctx_cm.octx
    cf = octx.core_cf_ref[]
    W = size(ctx.U, 1)
    obj = pcx.ctx_cm.obj
    ov = verify_inner_solution_operator_originzc!(base.ζstar, base.λstar, cf, octx.fg_zc_op, octx.fg_layout, νfull0, obj, W)

    # --- (1) per-block KKT residual breakdown ---
    check("K=$K_mean/$K_pair: kkt_resid_E present+small ($(ov.kkt_resid_E))", ov.kkt_resid_E < 1e-6)
    check("K=$K_mean/$K_pair: kkt_resid_mean present+small ($(ov.kkt_resid_mean))", ov.kkt_resid_mean < 1e-6)
    check("K=$K_mean/$K_pair: kkt_resid_pair present+small ($(ov.kkt_resid_pair))", K_pair == 0 ? ov.kkt_resid_pair == 0.0 : ov.kkt_resid_pair < 1e-6)
    check("K=$K_mean/$K_pair: france_ratio_resid present+small ($(ov.france_ratio_resid))", isfinite(ov.france_ratio_resid) && ov.france_ratio_resid < 1e-6)
    block_max = max(ov.kkt_resid_E, ov.kkt_resid_mean, ov.kkt_resid_pair)
    check("K=$K_mean/$K_pair: max(per-block) == aggregate kkt_resid (block_max=$block_max, kkt_resid=$(ov.kkt_resid))",
          abs(block_max - ov.kkt_resid) < 1e-12)

    # --- (2) recovered full factual shares ---
    m_weights, _ = verify_namedtuple_from_operator(ov, obj, W, verify.inner_status)
    rec = verify_recovered_full_factual_shares(base.θ_full0, ctx, cf, m_weights)
    check("K=$K_mean/$K_pair: recovered full A reproduces factual winner (mismatches=$(rec.max_winner_mismatch))", rec.max_winner_mismatch == 0)
    check("K=$K_mean/$K_pair: recovered full A preserves share ratios (max_diff=$(rec.max_share_ratio_diff))", rec.max_share_ratio_diff < 1e-6)
    @printf "  K=%d/%d  kkt_E=%.3e kkt_mean=%.3e kkt_pair=%.3e france=%.3e  winner_mismatch=%d share_ratio_diff=%.3e\n" K_mean K_pair ov.kkt_resid_E ov.kkt_resid_mean ov.kkt_resid_pair ov.france_ratio_resid rec.max_winner_mismatch rec.max_share_ratio_diff
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
