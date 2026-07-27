# shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: validates the
# newly-built verify_inner_solution_operator_cm! (flexible-CM, G=[E|C]) against the pre-existing
# dense verification path (archC_verified_state's own dense recompute) at D=4. Mirrors
# test_operator_verification_cmmeanzc.jl's structure exactly, minus the ZC mean/pair block.
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

for L in (10, 50)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    base, verify = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
    check("L=$L: dense verify succeeded (status=$(verify.inner_status))", verify.inner_status == 0)

    cctx = pcx.cctx
    cf = cctx.core_cf_ref[]
    check("L=$L: cf is a real CompressedFactual", cf isa CompressedFactual)
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_operator_cm!(base.ζstar, base.λstar, cf, cctx.L, cctx.nO, cctx.origins,
        cctx.refIndex1, bins_u, cctx.R, pcx.ctx_cm.obj, size(ctx.U, 1))

    check("L=$L: operator-recomputed inner objective is finite ($(ov.f))", isfinite(ov.f))
    check("L=$L: operator KKT residual is small (stationarity, resid=$(ov.kkt_resid))", ov.kkt_resid < 1e-6)
    check("L=$L: operator KKT residual agrees with dense max_abs_moment_kkt_resid (dense=$(verify.max_abs_moment_kkt_resid), operator=$(ov.kkt_resid))",
          abs(ov.kkt_resid - verify.max_abs_moment_kkt_resid) < 1e-6)
    @printf "  L=%d  dense_kkt=%.3e  operator_kkt=%.3e\n" L verify.max_abs_moment_kkt_resid ov.kkt_resid
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
