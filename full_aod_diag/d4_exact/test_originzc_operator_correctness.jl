# port/shared-inner-fg-operator-and-verification-2026-07-26: D=4 (and D=20/W=80,000 via ARGS[1]=="d20")
# correctness gate for origin-ZC's new operator FG (`OriginZCOperatorState`,
# `cm_originzc_lookup_kernels.jl`/`cm_originzc_lookup_production.jl`) against the pre-existing
# dense-reference path (`inner_loop_internal_archgeneric`) -- mirrors
# test_phaseB1_cmlookup_production_correctness.jl's structure (same category of comparison: zeta*,
# lambda*, m_weights, Delta_dual, downstream full gradient, at calibration + a perturbed point,
# both K_mean/K_pair configs).
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/test_originzc_operator_correctness.jl [d4|d20]
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function run_gate(ctx, x_free_calib, x_free_pert; K_configs, W_label)
    D = ctx.D
    nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

    for (K_mean, K_pair) in K_configs
        println("\n---- $W_label K_mean=$K_mean K_pair=$K_pair ----")
        layout = OriginByPowerLayout(D, K_mean, K_pair)
        νfull0 = nu0_origin(K_mean, D)

        pcx_dense = build_originzc_production_context(ctx, CS, layout; fg_backend = :dense_reference)
        pcx_op = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
        check("$W_label K=$K_mean/$K_pair: dense octx tagged correctly", pcx_dense.octx.fg_backend === :dense_reference)
        check("$W_label K=$K_mean/$K_pair: operator octx tagged correctly", pcx_op.octx.fg_backend === :operator)

        for (tag, x_free) in [("calib", x_free_calib), ("perturbed", x_free_pert)]
            _, base_d, verify_d = cm_originzc_production_value_verified(x_free, νfull0, pcx_dense)
            _, base_o, verify_o = cm_originzc_production_value_verified(x_free, νfull0, pcx_op)

            check("$W_label K=$K_mean/$K_pair $tag: inner_status agrees (dense=$(verify_d.inner_status), operator=$(verify_o.inner_status))",
                  verify_d.inner_status == verify_o.inner_status)
            dz = abs(base_d.ζstar - base_o.ζstar)
            check("$W_label K=$K_mean/$K_pair $tag: zeta* agrees (diff=$dz)", dz < 1e-8)
            dl = maximum(abs.(base_d.λstar .- base_o.λstar))
            check("$W_label K=$K_mean/$K_pair $tag: lambda* agrees (max abs diff=$dl)", dl < 1e-6)
            dm = maximum(abs.(base_d.m_star .- base_o.m_star))
            check("$W_label K=$K_mean/$K_pair $tag: m_star agrees (max abs diff=$dm)", dm < 1e-6)
            dd = abs(verify_d.Delta_dual - verify_o.Delta_dual)
            check("$W_label K=$K_mean/$K_pair $tag: Delta_dual agrees (diff=$dd)", dd < 1e-8)
            println("    dense: Delta_dual=$(verify_d.Delta_dual)  n_fg_dense_fallback=$(pcx_op.octx.fg_lookup_st === nothing ? "n/a" : pcx_op.octx.fg_lookup_st.n_dense_econ_fallback)  | operator: Delta_dual=$(verify_o.Delta_dual)")
            check("$W_label K=$K_mean/$K_pair $tag: operator took the compressed-cf path (no dense-fallback calls)",
                  pcx_op.octx.fg_lookup_st === nothing || pcx_op.octx.fg_lookup_st.n_dense_econ_fallback == 0)

            g_d, _ = cm_originzc_production_gradient(x_free, νfull0, pcx_dense, ctx, pe_g; base = base_d, verify = verify_d, threaded = false)
            g_o, _ = cm_originzc_production_gradient(x_free, νfull0, pcx_op, ctx, pe_g; base = base_o, verify = verify_o, threaded = false)
            dg = maximum(abs.(g_d .- g_o))
            check("$W_label K=$K_mean/$K_pair $tag: downstream (g,A_od,eta) gradient agrees (max abs diff=$dg)", dg < 1e-6)
        end
    end
end

pe_g = nothing   # set below (needs ctx first)

if SCALE == "d4"
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    global pe_g = build_pivot_elimination(ctx)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    x_free_pert = x_free_calib .* (1.0 .+ 0.01 .* [isodd(i) ? 1 : -1 for i in eachindex(x_free_calib)])
    run_gate(ctx, x_free_calib, x_free_pert; K_configs = [(1, 0), (1, 1), (2, 2)], W_label = "D4")
elseif SCALE == "d20"
    W = 80000
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                                 draw_seed = 20260719, destination_sample = :exclude_row)
    global pe_g = build_pivot_elimination(ctx)
    D = ctx.D; Ddest = ctx.D_dest
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe_g)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0 * 1.01, zfree0)
    x_free_calib = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe_g))))   # xfc convention: exp+pivot_expand, matching every existing production caller (e.g. test_exclude_row_gateB_meanzc_originzc_k1.jl)
    # NOTE: a naive uniform +/-0.5% perturbation on exp-space Aod values pushed the DENSE reference
    # itself into nStatus=-300 (unbounded) at this real D=20/W=80,000 point (confirmed: dense fails
    # first in run_gate's own call order, before operator is ever reached) -- a test-construction
    # artifact of this specific point's feasible region, not an operator bug. Perturb only the
    # gp0/gamma-prime component (index 1), which every other real D=20 gate in this codebase treats
    # as the safe one-DOF perturbation direction (see calibrated-start-matched-comparison memory).
    x_free_pert = copy(x_free_calib)
    x_free_pert[1] *= 1.001
    run_gate(ctx, x_free_calib, x_free_pert; K_configs = [(1, 1)], W_label = "D20/W=$W")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

println()
println("="^40 * " SUMMARY ($SCALE) " * "="^40)
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
