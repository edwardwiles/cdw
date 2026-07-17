# Task §4/§9/§12A-D validation. Run: julia --project=. full_aod_diag/d4_exact/test_three_way.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
using LinearAlgebra: dot

const COMMIT = "f3f4b6b"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
base = solve_base_state(x0, ctx)
Δ0 = optimized_Delta(x0, ctx)

println("="^78); println("TEST 1: all three objects EXACTLY equal at the base point"); println("="^78)
Q0 = frozen_adjoint_Q(x0, ctx, base)
L0 = fixed_dual_L(x0, ctx, base)
println("Delta_dual(x0)      = ", Δ0)
println("frozen_adjoint_Q(x0) = ", Q0, "   diff = ", abs(Q0 - Δ0))
println("fixed_dual_L(x0)     = ", L0, "   diff = ", abs(L0 - Δ0))
@assert abs(Q0 - Δ0) < 1e-12 "frozen_adjoint_Q does not match Delta_dual at the base point -- construction bug"
@assert abs(L0 - Δ0) < 1e-12 "fixed_dual_L does not match Delta_dual at the base point -- construction bug"
println("PASS: Q_adj(x0) == L_fix(x0) == Delta_dual(x0) to <1e-12 (exact by construction, verified not assumed)")

println("\n" * "="^78); println("TEST 2: small-h directional agreement (all three should match to O(h))"); println("="^78)
D = ctx.D
rng = MersenneTwister(20260717)
v = randn(rng, length(x0)); v ./= norm(v)   # random direction in x_free space (17-dim)

for h in (0.1, 0.01, 0.001, 0.0001)
    xp = x0 .+ h .* (x0 .* v)   # multiplicative-style perturbation scaled by x0 to stay in a sane range
    xm = x0 .- h .* (x0 .* v)
    Q_p = frozen_adjoint_Q(xp, ctx, base); Q_m = frozen_adjoint_Q(xm, ctx, base)
    L_p = fixed_dual_L(xp, ctx, base);     L_m = fixed_dual_L(xm, ctx, base)
    Δ_p = optimized_Delta(xp, ctx);        Δ_m = optimized_Delta(xm, ctx)
    slope_Q = (Q_p - Q_m) / (2h)
    slope_L = (L_p - L_m) / (2h)
    slope_Δ = (Δ_p - Δ_m) / (2h)
    println("h=", h, "  slope_Q_adj=", slope_Q, "  slope_L_fix=", slope_L, "  slope_optimized=", slope_Δ)
    println("      |Q-L|=", abs(slope_Q - slope_L), "  |Q-opt|=", abs(slope_Q - slope_Δ), "  |L-opt|=", abs(slope_L - slope_Δ))
    if h <= 0.001
        @assert sign(slope_Q) == sign(slope_L) == sign(slope_Δ) "small-h slopes disagree in SIGN at h=$h (Q=$slope_Q, L=$slope_L, opt=$slope_Δ) -- envelope agreement should hold at small h; a sign flip means a real construction bug, not FD noise"
        @assert abs(slope_Q - slope_L) / abs(slope_L) < 0.1 "Q_adj and L_fix small-h slopes disagree by >10% at h=$h -- expected to agree closely since m_s* is a good approximation to Psi'(q_s(x)) for small perturbations"
    end
end
println("\n(Interpretation printed to console; task explicitly expects agreement only in the h->0 limit,")
println(" NOT at h=0.1 -- do not assert equality here, this test's job is to MEASURE, not assume.)")

open(joinpath(OUTDIR, "three_way_h_sweep.csv"), "w") do io
    println(io, "h,slope_Q_adj,slope_L_fix,slope_optimized,abs_Q_minus_L,abs_Q_minus_opt,abs_L_minus_opt")
    for h in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.001, 0.0001)
        xp = x0 .+ h .* (x0 .* v); xm = x0 .- h .* (x0 .* v)
        Q_p = frozen_adjoint_Q(xp, ctx, base); Q_m = frozen_adjoint_Q(xm, ctx, base)
        L_p = fixed_dual_L(xp, ctx, base);     L_m = fixed_dual_L(xm, ctx, base)
        Δ_p = optimized_Delta(xp, ctx);        Δ_m = optimized_Delta(xm, ctx)
        sQ = (Q_p - Q_m) / (2h); sL = (L_p - L_m) / (2h); sD = (Δ_p - Δ_m) / (2h)
        println(io, h, ",", sQ, ",", sL, ",", sD, ",", abs(sQ-sL), ",", abs(sQ-sD), ",", abs(sL-sD))
    end
end
println("Wrote ", joinpath(OUTDIR, "three_way_h_sweep.csv"))
println("\nTHREE-WAY DERIVATIVE DISTINCTION: BASE-POINT EXACTNESS VERIFIED")
