# ============================================================================
# Remediation task Part A, finding F1: regression test for the CM divergence identity at a
# TAIL-ACTIVE point (a real point where at least one recovered weight m* exceeds e, so the
# hybrid KL/quadratic divergence's quadratic branch is genuinely nonempty and
# mean(Psi(q*)) > 0 strictly).
#
# Confirms, at such a point:
#   Delta_dual == -(mean(Psi(q*)) + zeta*)                              (definitional identity)
#   (-zeta_star) - Delta_dual == mean(Psi(q*)) > 0                      (F1's exact mechanism)
#
# This is NOT a claim that `-zeta_star ~= Delta_dual` in general -- per the task's explicit
# instruction, that equality is FALSE in the hybrid tail and must not be asserted as if it held.
# The point is to confirm the two diverge by exactly mean(Psi(q*)), strictly positive, at a real
# point where the divergence is nonzero -- i.e. that the fix (cm_checkpoint.jl/cm_outer_driver.jl
# now use verify.Delta_dual / delta_dual_from_base, not -base.ζstar) matters, not just that the
# formula is algebraically consistent.
#
# The base point is a mild perturbation of the real D=20/W=80,000/L=50 calibration point along
# gamma'_focal (confirmed live in remediation_a1_verify_delta_dual_identity.jl to be feasible,
# VerifiedSolved, and tail-active: m_max=5.15>e, frac(m*>e)=0.0003, mean(Psi(q*))=3.33e-5).
# ============================================================================
using Test
const D4X = @__DIR__
include(joinpath(D4X, "context_real_d20.jl"))
include(joinpath(D4X, "winners.jl"))
include(joinpath(D4X, "oracle.jl"))
include(joinpath(D4X, "common_marginals_moments.jl"))
include(joinpath(D4X, "common_marginals_interval.jl"))
include(joinpath(D4X, "instrumentation.jl"))
include(joinpath(D4X, "oracle_fast.jl"))
include(joinpath(D4X, "gravity_elimination.jl"))
include(joinpath(D4X, "three_way_derivatives.jl"))
include(joinpath(D4X, "lfix_incremental.jl"))
include(joinpath(D4X, "composite_gradient.jl"))
include(joinpath(D4X, "composite_gradient_fast.jl"))
include(joinpath(D4X, "cm_lookup_kernels.jl"))
include(joinpath(D4X, "lfix_cm_aware.jl"))
include(joinpath(D4X, "cm_hessian_architectures.jl"))
include(joinpath(D4X, "cm_production_bundle.jl"))
include(joinpath(D4X, "nested_quantile_grids.jl"))
using Printf, LinearAlgebra, Statistics

W = 80000; DELTA = 1.0; L = 50
println(">>> building D20 real-data context, W=$W ...")
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)

snaps = nested_grid_sequence([10, 20, 50])
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])

x_free_calib = ctx.θ0_up[ctx.free_idx]
gp_calib = x_free_calib[1]
x_free_tail = copy(x_free_calib)
x_free_tail[1] = gp_calib * 0.999   # confirmed live: feasible, VerifiedSolved, m_max=5.15>e

K, base, verify = cm_production_value_verified(x_free_tail, pcx)

@testset "CM Delta_dual identity at a tail-active point (F1)" begin
    @test base.inner_status in (0, -100, -101, -103)
    @test is_verified_success(verify)

    zeta_star = base.ζstar
    Delta_dual = verify.Delta_dual
    naive_minus_zeta = -zeta_star

    # independently recompute mean(Psi(q*)) via a second code path (not reusing verify's
    # internals), matching lfix_incremental.jl/three_way_derivatives.jl's fixed_dual_L formula
    obj = pcx.ctx_cm.obj
    oci = obj.outer_constr_index
    θ_full = CS.reconstruct_full(x_free_tail, pcx.ctx_cm.m)
    Wn = size(obj.U, 1)
    Kk = zeros(Wn); G = zeros(Wn, obj.d)
    obj.moments!(Kk, G, θ_full, obj.U, obj)
    q = [-zeta_star - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:Wn]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    mean_Psi_q = sum(Psi_q) / Wn
    m_max = maximum(base.m_star)

    @printf("zeta_star=%.10f  Delta_dual=%.10f  -zeta_star=%.10f  mean(Psi(q*))=%.10f  m_max=%.4f\n",
            zeta_star, Delta_dual, naive_minus_zeta, mean_Psi_q, m_max)

    @test m_max > ℯ   # confirms this point genuinely exercises the quadratic branch
    @test mean_Psi_q > 0   # confirms the divergence between Delta_dual and -zeta_star is nonzero here

    # definitional identity: Delta_dual == -(mean(Psi(q*)) + zeta*)
    @test isapprox(Delta_dual, -(mean_Psi_q + zeta_star); atol = 1e-12, rtol = 1e-10)

    # F1's exact mechanism, per the task's own required check:
    #   (-zeta_star) - Delta_dual ≈ mean(Psi(q*)) > 0
    diff = naive_minus_zeta - Delta_dual
    @test diff > 0
    @test isapprox(diff, mean_Psi_q; atol = 1e-12, rtol = 1e-10)

    # explicitly NOT asserting `-zeta_star ≈ Delta_dual` -- that equality is false here by
    # construction (this IS the tail-active point where they diverge); confirm they are NOT
    # close, as a guard against a future accidental revert of the F1 fix silently going
    # undetected because a real production point happened to be non-tail-active.
    @test !isapprox(naive_minus_zeta, Delta_dual; atol = 1e-8)
end
println("All CM Delta_dual tail-active identity tests passed.")
