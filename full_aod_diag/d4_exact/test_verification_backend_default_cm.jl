# verification-defaults task (2026-07-27), Section 6.1: full comparison gate for flexible-CM's
# NEWLY-WIRED archC_verified_state(...; verification_backend=:operator) against the pre-existing
# :dense_reference path, at D=4 (companion _d20.jl runs the same at real D=20/W=80,000). Covers
# every Section 6.1 item: draw-level dual index (m_star), objective, complete dual gradient
# (full vector, not just KKT residual), KKT residual, feasibility/moment residual, status
# classification, cache admission decision, incumbent admission decision, cold verification (both
# calls below are fresh -- verification_backend selects which backend a FRESH call takes, neither
# reuses the other's scratch).
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

println("=== D=4 flexible-CM verification_backend gate (archC_verified_state) ===")
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
W = size(ctx.U, 1)

for L in (10, 50)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    cctx = pcx.cctx

    # dense_reference FIRST (populates obj.H's CM columns, needed below for the full-gradient
    # cross-check) -- a fresh call, no state shared with the operator call below.
    base_d, verify_d = archC_verified_state(x_free_calib, pcx.ctx_cm, cctx; verification_backend = :dense_reference)
    G = CS.select_G_from_H(pcx.ctx_cm.obj, pcx.ctx_cm.obj.H)
    nkkt = min(length(base_d.λstar), size(G, 2))
    g_dense_full = -(1.0 / W) .* (transpose(@view(G[:, 1:nkkt])) * base_d.m_star)

    # operator, fresh call (cold verification: independent scratch, per operator_verification.jl's
    # own docstrings -- no reuse of base_d's own computation beyond the shared (ζstar,λstar) point).
    base_o, verify_o = archC_verified_state(x_free_calib, pcx.ctx_cm, cctx; verification_backend = :operator)

    label = "flexible-CM L=$L"
    gcheck("$label: same inner-solve dual point (ζ,λ unaffected by verification_backend)",
           base_d.ζstar == base_o.ζstar && base_d.λstar == base_o.λstar)

    maxerr_draw = maximum(abs.(base_d.m_star .- base_o.m_star))
    gcheck("$label: draw-level dual index (m_star, per-draw dPsi(r)) agrees (max|Δ|=$maxerr_draw)", maxerr_draw < 1e-8)

    # Complete dual gradient (full vector, not just the max-abs KKT residual already checked by
    # compare_verify_tuples below): recompute the operator's own g_lambda directly via the
    # standalone verifier (same cf/L/nO/... cctx already carries) and diff element-wise against the
    # dense G'm_weights product built above.
    cf = cctx.core_cf_ref[]
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_operator_cm!(base_d.ζstar, base_d.λstar, cf, cctx.L, cctx.nO, cctx.origins,
        cctx.refIndex1, bins_u, cctx.R, pcx.ctx_cm.obj, W)
    maxerr_g = maximum(abs.(ov.g_lambda .- g_dense_full))
    gcheck("$label: complete dual gradient (full vector) agrees (max|Δg|=$maxerr_g)", maxerr_g < 1e-8)

    compare_verify_tuples(label, verify_d, verify_o)
    @printf "  %s  dense_Delta_dual=%.10f  operator_Delta_dual=%.10f\n" label verify_d.Delta_dual verify_o.Delta_dual
end

println()
println("Gate tally: ", GATE_TALLY.n_pass, "/", GATE_TALLY.n_checks, " checks passed")
GATE_TALLY.n_pass == GATE_TALLY.n_checks ? println("ALL PASS") : (println("SOME FAILURES"); exit(1))
