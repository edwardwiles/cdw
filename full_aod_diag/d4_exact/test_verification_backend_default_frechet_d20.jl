# verification-defaults task (2026-07-27), Section 6.1: real D=20/W=80,000 companion to
# test_verification_backend_default_frechet.jl (L=50, production default). Supersedes
# test_operator_verification_cm_frechet_d20.jl's own narrower KKT-only check with the full
# Section 6.1 comparison set.
const D4X = @__DIR__
for f in ["context.jl","context_real_d20.jl","draw_design.jl","winners.jl","oracle.jl",
          "common_marginals_moments.jl","common_marginals_interval.jl","instrumentation.jl","oracle_fast.jl",
          "gravity_elimination.jl","three_way_derivatives.jl","lfix_incremental.jl",
          "composite_gradient.jl","composite_gradient_fast.jl","gradient_workspace.jl","shared_a_gradient.jl",
          "cm_lookup_kernels.jl","lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "nested_quantile_grids.jl","lfix_factorized.jl","lfix_factorized_workspace.jl","lfix_cm_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl","cm_frechet_lookup_production.jl",
          "operator_verification.jl","verification_gate_utils.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

println("=== Real D=20/W=80,000 common-Frechet verification_backend gate (archC_frechet_verified_state, L=50) ===")
flush(stdout)

ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
W = size(ctx.U, 1)
L = 50

pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored, cm_hessian_backend = :structured)
cctx = pcx.cctx
level_targets = pcx.aug.level_targets

println("Solving dense_reference..."); flush(stdout)
base_d, verify_d = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, cctx, level_targets; verification_backend = :dense_reference)
println("  status=$(verify_d.inner_status)  Delta_dual=$(verify_d.Delta_dual)"); flush(stdout)
G = CS.select_G_from_H(pcx.ctx_cm.obj, pcx.ctx_cm.obj.H)
nkkt = min(length(base_d.λstar), size(G, 2))
g_dense_full = -(1.0 / W) .* (transpose(@view(G[:, 1:nkkt])) * base_d.m_star)

println("Solving :operator..."); flush(stdout)
base_o, verify_o = archC_frechet_verified_state(x_free_calib, pcx.ctx_cm, cctx, level_targets; verification_backend = :operator)
println("  status=$(verify_o.inner_status)  Delta_dual=$(verify_o.Delta_dual)"); flush(stdout)

label = "common-Frechet D20/W=80000 L=$L"
gcheck("$label: same inner-solve dual point (ζ,λ unaffected by verification_backend)",
       base_d.ζstar == base_o.ζstar && base_d.λstar == base_o.λstar)

maxerr_draw = maximum(abs.(base_d.m_star .- base_o.m_star))
gcheck("$label: draw-level dual index (m_star) agrees (max|Δ|=$maxerr_draw)", maxerr_draw < 1e-6)

cf = cctx.core_cf_ref[]
bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
ov = verify_inner_solution_operator_cm_frechet!(base_d.ζstar, base_d.λstar, cf, cctx.L, cctx.nO, cctx.origins,
    cctx.refIndex1, bins_u, cctx.R, level_targets, pcx.ctx_cm.obj, W)
maxerr_g = maximum(abs.(ov.g_lambda .- g_dense_full))
gcheck("$label: complete dual gradient (full vector) agrees (max|Δg|=$maxerr_g)", maxerr_g < 1e-6)

compare_verify_tuples(label, verify_d, verify_o; kkt_tol = 1e-5, obj_tol = 1e-6)
@printf "  %s  dense_Delta_dual=%.10f  operator_Delta_dual=%.10f\n" label verify_d.Delta_dual verify_o.Delta_dual
flush(stdout)

println()
println("Gate tally: ", GATE_TALLY.n_pass, "/", GATE_TALLY.n_checks, " checks passed")
GATE_TALLY.n_pass == GATE_TALLY.n_checks ? println("ALL PASS") : (println("SOME FAILURES"); exit(1))
