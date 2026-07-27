# verification-defaults task (2026-07-27), Section 6.1: full comparison gate for origin-ZC's
# NEWLY-WIRED archOZ_verified_state(...; verification_backend=:operator) against the pre-existing
# :dense_reference path, at D=4. See test_verification_backend_default_cm.jl's header for what each
# check covers.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "operator_verification.jl",
          "verification_gate_utils.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

println("=== D=4 origin-ZC verification_backend gate (archOZ_verified_state) ===")
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
W = size(ctx.U, 1)
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 0), (1, 1), (2, 2)]
    layout = OriginByPowerLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
    octx = pcx.ctx_cm.octx

    base_d, verify_d = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm; verification_backend = :dense_reference)
    G = CS.select_G_from_H(pcx.ctx_cm.obj, pcx.ctx_cm.obj.H)
    nkkt = min(length(base_d.λstar), size(G, 2))
    g_dense_full = -(1.0 / W) .* (transpose(@view(G[:, 1:nkkt])) * base_d.m_star)

    base_o, verify_o = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm; verification_backend = :operator)

    label = "origin-ZC K=$K_mean/$K_pair"
    gcheck("$label: same inner-solve dual point (ζ,λ unaffected by verification_backend)",
           base_d.ζstar == base_o.ζstar && base_d.λstar == base_o.λstar)

    maxerr_draw = maximum(abs.(base_d.m_star .- base_o.m_star))
    gcheck("$label: draw-level dual index (m_star) agrees (max|Δ|=$maxerr_draw)", maxerr_draw < 1e-8)

    cf = octx.core_cf_ref[]
    ov = verify_inner_solution_operator_originzc!(base_d.ζstar, base_d.λstar, cf, octx.fg_zc_op, octx.fg_layout, νfull0, pcx.ctx_cm.obj, W)
    maxerr_g = maximum(abs.(ov.g_lambda .- g_dense_full))
    gcheck("$label: complete dual gradient (full vector) agrees (max|Δg|=$maxerr_g)", maxerr_g < 1e-8)

    compare_verify_tuples(label, verify_d, verify_o)
    @printf "  %s  dense_Delta_dual=%.10f  operator_Delta_dual=%.10f\n" label verify_d.Delta_dual verify_o.Delta_dual
end

println()
println("Gate tally: ", GATE_TALLY.n_pass, "/", GATE_TALLY.n_checks, " checks passed")
GATE_TALLY.n_pass == GATE_TALLY.n_checks ? println("ALL PASS") : (println("SOME FAILURES"); exit(1))
