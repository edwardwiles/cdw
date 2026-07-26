# Phase B1 remediation (production-audit continuation, 2026-07-26) correctness gate:
# inner_fg_backend=:cm_lookup (cm_lookup_production.jl, production Hessian callback UNCHANGED)
# vs :dense_reference, through the REAL production entry points archC_base_state/
# archC_verified_state (cm_production_bundle.jl) -- not a standalone microkernel comparison.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/test_phaseB1_cmlookup_production_correctness.jl [d4|d20]
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, LinearAlgebra, Random, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

function run_gate(ctx, x_free_calib, x_free_pert; L_list, W_label)
    for L in L_list, contrasts in (:anchored, :orthonormal)
        lp("---- $W_label L=$L contrasts=$contrasts ----")
        pcx_dense = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, inner_fg_backend = :dense_reference)
        pcx_lookup = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, inner_fg_backend = :cm_lookup)
        check(pcx_dense.cctx.inner_fg_backend == :dense_reference, "$W_label L=$L $contrasts: dense cctx tagged correctly")
        check(pcx_lookup.cctx.inner_fg_backend == :cm_lookup, "$W_label L=$L $contrasts: lookup cctx tagged correctly")

        for (label, xf) in (("calib", x_free_calib), ("perturbed", x_free_pert))
            base_d, verify_d = archC_verified_state(xf, pcx_dense.ctx_cm, pcx_dense.cctx)
            base_l, verify_l = archC_verified_state(xf, pcx_lookup.ctx_cm, pcx_lookup.cctx)

            check(base_d.inner_status == base_l.inner_status,
                "$W_label L=$L $contrasts $label: inner_status agrees (dense=$(base_d.inner_status), lookup=$(base_l.inner_status))")
            zeta_diff = abs(base_d.ζstar - base_l.ζstar)
            lambda_diff = maximum(abs.(base_d.λstar .- base_l.λstar))
            m_diff = maximum(abs.(base_d.m_star .- base_l.m_star))
            Delta_diff = abs(verify_d.Delta_dual - verify_l.Delta_dual)
            check(zeta_diff < 1e-8, "$W_label L=$L $contrasts $label: zeta* agrees (diff=$zeta_diff)")
            check(lambda_diff < 1e-6, "$W_label L=$L $contrasts $label: lambda* agrees (max abs diff=$lambda_diff)")
            check(m_diff < 1e-6, "$W_label L=$L $contrasts $label: m_weights (obj.arg1) agrees (max abs diff=$m_diff)")
            check(Delta_diff < 1e-9, "$W_label L=$L $contrasts $label: Delta_dual agrees (diff=$Delta_diff)")
            lp("    dense: Delta_dual=$(verify_d.Delta_dual) n_fg n/a  | lookup: Delta_dual=$(verify_l.Delta_dual)")

            # Downstream gradient (cm_production_gradient, composite_gradient_at_fast) must also
            # agree when fed each backend's own base -- gradient math itself does not depend on
            # inner_fg_backend, this confirms the base handoff is complete/consistent (no stray
            # obj.H/obj.arg0 staleness left over from whichever FG kernel produced `base`).
            pe = build_pivot_elimination(ctx)
            g_d, meta_d = cm_production_gradient(xf, pcx_dense, ctx, pe; base = base_d)
            g_l, meta_l = cm_production_gradient(xf, pcx_lookup, ctx, pe; base = base_l)
            g_diff = maximum(abs.(g_d .- g_l))
            check(g_diff < 1e-6, "$W_label L=$L $contrasts $label: downstream gradient agrees (max abs diff=$g_diff)")
        end
    end
end

if SCALE == "d4"
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    Random.seed!(1301)
    x_free_pert = copy(x_free_calib)
    x_free_pert[2:end] .*= exp.(0.04 .* randn(length(x_free_pert) - 1))
    run_gate(ctx, x_free_calib, x_free_pert; L_list = (10, 20, 50), W_label = "D4-square")
elseif SCALE == "d20"
    ctx = d20_real_setup_design(W = 80_000, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    Random.seed!(1301)
    x_free_pert = copy(x_free_calib)
    x_free_pert[2:end] .*= exp.(0.02 .* randn(length(x_free_pert) - 1))
    run_gate(ctx, x_free_calib, x_free_pert; L_list = (50,), W_label = "D20-real-W80000")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

lp("==================== SUMMARY ($SCALE) ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES ($(length(FAILURES))): ", FAILURES)
    exit(1)
end
