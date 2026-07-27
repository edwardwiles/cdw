# verification-defaults task (2026-07-27), Section 6.1: full comparison gate for common-Frechet's
# NEWLY-WIRED archC_frechet_verified_state(...; verification_backend=:operator) against the
# pre-existing :dense_reference path, at D=4 (companion _d20.jl runs real D=20/W=80,000). See
# test_verification_backend_default_cm.jl's header for what each check covers.
const D4X = @__DIR__
for f in ["context.jl","draw_design.jl","winners.jl","oracle.jl",
          "common_marginals_moments.jl","common_marginals_interval.jl","instrumentation.jl","oracle_fast.jl",
          "gravity_elimination.jl","three_way_derivatives.jl","lfix_incremental.jl",
          "composite_gradient.jl","composite_gradient_fast.jl","gradient_workspace.jl","shared_a_gradient.jl",
          "cm_lookup_kernels.jl","lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "nested_quantile_grids.jl","lfix_factorized.jl","lfix_factorized_workspace.jl","lfix_cm_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl",
          "operator_verification.jl","verification_gate_utils.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

println("=== D=4 common-Frechet verification_backend gate (archC_frechet_verified_state) ===")
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
W = size(ctx.U, 1)

for L in (10, 50)
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, cm_hessian_backend = :structured)
    cctx = pcx.cctx
    level_targets = pcx.aug.level_targets

    base_d, verify_d = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, cctx, level_targets; verification_backend = :dense_reference)
    G = CS.select_G_from_H(pcx.ctx_cm.obj, pcx.ctx_cm.obj.H)
    nkkt = min(length(base_d.λstar), size(G, 2))
    g_dense_full = -(1.0 / W) .* (transpose(@view(G[:, 1:nkkt])) * base_d.m_star)

    base_o, verify_o = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, cctx, level_targets; verification_backend = :operator)

    label = "common-Frechet L=$L"
    gcheck("$label: same inner-solve dual point (ζ,λ unaffected by verification_backend)",
           base_d.ζstar == base_o.ζstar && base_d.λstar == base_o.λstar)

    maxerr_draw = maximum(abs.(base_d.m_star .- base_o.m_star))
    gcheck("$label: draw-level dual index (m_star) agrees (max|Δ|=$maxerr_draw)", maxerr_draw < 1e-8)

    cf = cctx.core_cf_ref[]
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_operator_cm_frechet!(base_d.ζstar, base_d.λstar, cf, cctx.L, cctx.nO, cctx.origins,
        cctx.refIndex1, bins_u, cctx.R, level_targets, pcx.ctx_cm.obj, W)
    maxerr_g = maximum(abs.(ov.g_lambda .- g_dense_full))
    gcheck("$label: complete dual gradient (full vector) agrees (max|Δg|=$maxerr_g)", maxerr_g < 1e-8)

    compare_verify_tuples(label, verify_d, verify_o)
    @printf "  %s  dense_Delta_dual=%.10f  operator_Delta_dual=%.10f\n" label verify_d.Delta_dual verify_o.Delta_dual
end

println()
println("Gate tally: ", GATE_TALLY.n_pass, "/", GATE_TALLY.n_checks, " checks passed")
GATE_TALLY.n_pass == GATE_TALLY.n_checks ? println("ALL PASS") : (println("SOME FAILURES"); exit(1))
