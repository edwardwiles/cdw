# ============================================================================
# D=4 basis-invariance gates (task brief §5/§11): cumulative (Q0) <-> interval
# (Q1) exact transform, for BOTH the CDF and POWER families, plus Delta*/
# primal-weight invariance under the CDF-only feature set (the interval
# kernel this branch ported, cm_frechet_bases_structured.jl, is CDF-only --
# see the port-readiness report for why :cdf_power stays :cumulative-only in
# this pass).
# ============================================================================
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases_q2.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
using Printf, LinearAlgebra, Random, Statistics

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end
function checkapprox(name, a, b; atol=1e-8, rtol=1e-8)
    check(name * "  [a=$a b=$b diff=$(abs(a-b))]", isapprox(a, b; atol=atol, rtol=rtol))
end

println("="^100); println("SETUP"); println("="^100)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; W = size(ctx.U, 1)
const L = 8
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
cfg_frec = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference)
targets = build_frechet_reference_targets(ctx, cfg_frec; L = L)
refIndex1 = ctx.γ.refIndex1
origins = [o for o in 1:D if o != refIndex1]
nO = length(origins)

function solve_delta(obj_cm)
    K, x, nStatus, _, _ = inner_loop_internal_archgeneric(obj_cm, θ_full0; hess_cb_builder = archA_hess_cb_builder)
    base = BaseDualState(collect(x_free_calib), θ_full0, x[1], collect(x[2:end]), copy(obj_cm.arg1), nStatus)
    return delta_dual_from_base(obj_cm, base), nStatus, x
end

println("="^100); println("GATE B1: Q1 (interval) CDF block == transform of Q0 (cumulative) CDF block"); println("="^100)
CM_q0_cdf, _, _ = precalc_frechet_reference_cdf(ctx.U, refIndex1, targets; contrasts = :orthonormal)
CM_q1_cdf, _, _, _ = precalc_frechet_reference_interval(ctx.U, refIndex1, targets; contrasts = :orthonormal)
T_cdf = frechet_block_transform_matrix(nO, L)
check("T_cdf square, size == ncm", size(T_cdf) == (nO*L+L, nO*L+L))
maxerr = maximum(abs.(CM_q1_cdf * T_cdf .- CM_q0_cdf))
println("max|Q1*T - Q0| (CDF) = $maxerr")
check("Q1*T == Q0 (CDF block) to machine precision", maxerr < 1e-10)
check("T_cdf nonsingular", cond(T_cdf) < 1e8)

println("="^100); println("GATE B2: Q1 (interval) POWER block == transform of Q0 (cumulative) POWER block"); println("="^100)
CM_q0_pow, _, _ = precalc_frechet_reference_power_cdf(ctx.U, refIndex1, targets; contrasts = :orthonormal)
CM_q1_pow, _, _, _ = precalc_frechet_reference_power_interval(ctx.U, refIndex1, targets; contrasts = :orthonormal)
maxerr2 = maximum(abs.(CM_q1_pow * T_cdf .- CM_q0_pow))
println("max|Q1*T - Q0| (POWER) = $maxerr2")
check("Q1*T == Q0 (POWER block) to machine precision", maxerr2 < 1e-10)

println("="^100); println("GATE B3: Delta*/primal-weight invariance Q0 vs Q1, feature_set=:cdf_only"); println("="^100)
aug_q0 = build_cm_frechet_augmented_obj_basis(ctx, CS, targets; basis=:cumulative, feature_set=:cdf_only, contrasts=:orthonormal)
aug_q1 = build_cm_frechet_augmented_obj_basis(ctx, CS, targets; basis=:interval, feature_set=:cdf_only, contrasts=:orthonormal)
Delta_q0, ns_q0, x_q0 = solve_delta(aug_q0.obj_cm)
Delta_q1, ns_q1, x_q1 = solve_delta(aug_q1.obj_cm)
check("Q0 solve feasible", frechet_solve_outcome(ns_q0) == :feasible)
check("Q1 solve feasible", frechet_solve_outcome(ns_q1) == :feasible)
checkapprox("Delta*: Q0 == Q1", Delta_q0, Delta_q1; atol=1e-7, rtol=1e-7)
pw_q0 = aug_q0.obj_cm.arg1 ./ sum(aug_q0.obj_cm.arg1)
pw_q1 = aug_q1.obj_cm.arg1 ./ sum(aug_q1.obj_cm.arg1)
maxerr_pw = maximum(abs.(pw_q0 .- pw_q1))
println("max|primal_weights_Q0 - primal_weights_Q1| = $maxerr_pw")
check("primal weights: Q0 == Q1 (1e-6 abs, cross-basis reparam of the SAME draws)", maxerr_pw < 1e-6)

println("="^100); println("GATE B4: Q1 structured (Architecture C) vs Q1 dense (Architecture A) Hessian, CDF-only"); println("="^100)
n_inner = aug_q1.ncore + aug_q1.ncm
nh = div(n_inner*(n_inner+1), 2)
h_dense_q1 = zeros(nh)
aug_q1.obj_cm(x_q1, h = h_dense_q1)
fctx_q1 = build_cm_frechet_interval_bin_ctx(ctx, aug_q1)
aug_q1.obj_cm.moments!(@view(aug_q1.obj_cm.H[:,1]), CS.select_G_from_H(aug_q1.obj_cm, aug_q1.obj_cm.H), θ_full0, aug_q1.obj_cm.U, aug_q1.obj_cm)
aug_q1.obj_cm.H[:,2] .= 1.0
_archC_prep_for_hessian!(aug_q1.obj_cm, x_q1)
h_struct_q1 = zeros(nh)
hessian_cm_frechet_interval_structured!(h_struct_q1, aug_q1.obj_cm, fctx_q1)
maxerr_h1 = maximum(abs.(h_dense_q1 .- h_struct_q1))
println("max|H_dense_Q1 - H_struct_Q1| = $maxerr_h1")
check("Q1 dense vs structured Hessian match to 1e-8 abs", maxerr_h1 < 1e-8)

println()
println("="^100)
@printf "TOTAL: %d PASS, %d FAIL\n" n_pass n_fail
println("="^100)
exit(n_fail == 0 ? 0 : 1)
