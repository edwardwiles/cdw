# shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: validates the
# newly-built verify_inner_solution_operator_cm_frechet! (G=[E|C|Level], the genuinely new
# level-anchor verification block) against the pre-existing dense verification path
# (archC_frechet_verified_state's own dense recompute) at D=4. Mirrors
# test_operator_verification_cm.jl's structure, with the extra level_targets/lambda_level slice.
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
using Test, Printf, LinearAlgebra, Random

all_pass = true
println("=== D=4 common-Frechet operator verification gate ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

for L in (10, 50)
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, cm_hessian_backend = :structured)
    base, verify = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    global all_pass &= verify.inner_status == 0
    @printf("  L=%d: dense verify status=%d\n", L, verify.inner_status)

    cctx = pcx.cctx
    cf = cctx.core_cf_ref[]
    is_cf = cf isa CompressedFactual
    global all_pass &= is_cf
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_operator_cm_frechet!(base.ζstar, base.λstar, cf, cctx.L, cctx.nO, cctx.origins,
        cctx.refIndex1, bins_u, cctx.R, pcx.aug.level_targets, pcx.ctx_cm.obj, size(ctx.U, 1))

    ok_f = isfinite(ov.f)
    ok_kkt = ov.kkt_resid < 1e-6
    ok_agree = abs(ov.kkt_resid - verify.max_abs_moment_kkt_resid) < 1e-6
    global all_pass &= ok_f && ok_kkt && ok_agree
    @printf("  L=%d  dense_kkt=%.3e  operator_kkt=%.3e  operator_f=%.6f  agree=%s\n",
            L, verify.max_abs_moment_kkt_resid, ov.kkt_resid, ov.f, ok_agree)
    @test ok_f
    @test ok_kkt
    @test ok_agree
end

println()
println(all_pass ? "ALL PASS" : "SOME FAILURES")
all_pass || error("common-Frechet operator verification d4 gate FAILED")
