# Phase 5.2 remediation (2026-07-26) correctness gate: common-Frechet inner_fg_backend=
# :cm_frechet_lookup (cm_frechet_lookup_production.jl, Architecture C Hessian UNCHANGED) vs
# :dense_reference, through the real production entry point archC_frechet_verified_state
# (cm_frechet_cplus.jl) -- mirrors test_phaseB1_cmlookup_production_correctness.jl's own structure
# exactly, for the CM-plus-level-anchor family instead of plain flexible CM.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/test_phase52_frechet_lookup_correctness.jl [d4|d20]
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"

function run_gate(ctx, x_free_calib, x_free_pert; L_list, W_label)
    for L in L_list, contrasts in (:anchored, :orthonormal)
        lp("---- $W_label L=$L contrasts=$contrasts ----")
        pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
            cm_hessian_backend = :structured, inner_fg_backend = :dense_reference)
        pcx_lookup = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
            cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup)
        check(pcx_dense.cctx.inner_fg_backend == :dense_reference, "$W_label L=$L $contrasts: dense cctx tagged correctly")
        check(pcx_lookup.cctx.inner_fg_backend == :cm_frechet_lookup, "$W_label L=$L $contrasts: lookup cctx tagged correctly")

        for (label, xf) in (("calib", x_free_calib), ("perturbed", x_free_pert))
            base_d, verify_d = archC_frechet_verified_state(xf, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
            base_l, verify_l = archC_frechet_verified_state(xf, pcx_lookup.ctx_cm, pcx_lookup.cctx, pcx_lookup.aug.level_targets)

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
            lp("    dense: Delta_dual=$(verify_d.Delta_dual)  | lookup: Delta_dual=$(verify_l.Delta_dual)")
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
    x_free_pert[2:end] .*= exp.(0.04 .* randn(length(x_free_pert) - 1))
    run_gate(ctx, x_free_calib, x_free_pert; L_list = (50,), W_label = "D20-real-W80000")
else
    error("unknown SCALE=$SCALE")
end

lp("="^40, " SUMMARY ($SCALE) ", "="^40)
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("$(length(FAILURES)) FAILURES:")
    for f in FAILURES
        lp("  FAIL: ", f)
    end
end
