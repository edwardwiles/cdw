# Phase 10 verifier audit (2026-08-02): extends test_operator_verification_cmmeanzc.jl's D=4 gate
# with the two genuine gaps found in the checklist audit -- (1) per-block KKT residual breakdown
# (kkt_resid_E/kkt_resid_mean/kkt_resid_pair/kkt_resid_cm/france_ratio_resid, now returned by
# verify_inner_solution_operator_cmmeanzc!) and (2) recovered full factual shares
# (verify_recovered_full_factual_shares, operator_verification.jl). Mirrors
# test_operator_verification_cmmeanzc.jl's own setup exactly -- purely additive, does not modify
# that file.
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

for (K_mean, K_pair, L) in [(1, 0, 10), (1, 1, 10), (2, 2, 20)]
    νvec0 = [Float64(factorial(k)) for k in 1:K_mean]
    pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
                                              contrasts = :anchored, meanzc_basis = :direct, inner_fg_backend = :operator)
    _, base, verify = cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx)
    check("K=$K_mean/$K_pair L=$L: dense verify succeeded", verify.inner_status == 0)

    cctx = pcx.cctx
    cf = cctx.core_cf_ref[]
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    W = size(ctx.U, 1)
    obj = pcx.ctx_cm.obj
    ov = verify_inner_solution_operator_cmmeanzc!(base.ζstar, base.λstar, cf, cctx.meanzc_zc_op, cctx.meanzc_zc_layout,
        νvec0, cctx.L, cctx.nO, cctx.origins, cctx.refIndex1, bins_u, cctx.R, obj, W)

    # --- (1) per-block KKT residual breakdown ---
    check("K=$K_mean/$K_pair L=$L: kkt_resid_E present+small ($(ov.kkt_resid_E))", ov.kkt_resid_E < 1e-6)
    check("K=$K_mean/$K_pair L=$L: kkt_resid_mean present+small ($(ov.kkt_resid_mean))", ov.kkt_resid_mean < 1e-6)
    check("K=$K_mean/$K_pair L=$L: kkt_resid_pair present+small ($(ov.kkt_resid_pair))", K_pair == 0 ? ov.kkt_resid_pair == 0.0 : ov.kkt_resid_pair < 1e-6)
    check("K=$K_mean/$K_pair L=$L: kkt_resid_cm present+small ($(ov.kkt_resid_cm))", ov.kkt_resid_cm < 1e-6)
    check("K=$K_mean/$K_pair L=$L: france_ratio_resid present+small ($(ov.france_ratio_resid))", isfinite(ov.france_ratio_resid) && ov.france_ratio_resid < 1e-6)
    block_max = max(ov.kkt_resid_E, ov.kkt_resid_mean, ov.kkt_resid_pair, ov.kkt_resid_cm)
    check("K=$K_mean/$K_pair L=$L: max(per-block) == aggregate kkt_resid (block_max=$block_max, kkt_resid=$(ov.kkt_resid))",
          abs(block_max - ov.kkt_resid) < 1e-12)

    # --- (2) recovered full factual shares ---
    m_weights, _ = verify_namedtuple_from_operator(ov, obj, W, verify.inner_status)
    rec = verify_recovered_full_factual_shares(base.θ_full0, ctx, cf, m_weights)
    check("K=$K_mean/$K_pair L=$L: recovered full A reproduces factual winner (mismatches=$(rec.max_winner_mismatch))", rec.max_winner_mismatch == 0)
    check("K=$K_mean/$K_pair L=$L: recovered full A preserves share ratios (max_diff=$(rec.max_share_ratio_diff))", rec.max_share_ratio_diff < 1e-6)
    @printf "  K=%d/%d L=%d  kkt_E=%.3e kkt_mean=%.3e kkt_pair=%.3e kkt_cm=%.3e france=%.3e  winner_mismatch=%d share_ratio_diff=%.3e\n" K_mean K_pair L ov.kkt_resid_E ov.kkt_resid_mean ov.kkt_resid_pair ov.kkt_resid_cm ov.france_ratio_resid rec.max_winner_mismatch rec.max_share_ratio_diff
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
