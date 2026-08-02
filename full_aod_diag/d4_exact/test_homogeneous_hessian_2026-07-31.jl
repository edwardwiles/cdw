# ============================================================================
# Task §11 gate: homogeneous H_EE vs. a finite-difference Hessian of the
# ALREADY-VERIFIED homogeneous dual objective (built from
# homogeneous_dual_contraction, not a new hand-derived reference -- an
# independent check by construction).
#
# Uses build_unrestricted_operator_ctx to get a genuine OperatorPsiBundle
# (M/arg0/arg1/arg2/Psi!/dPsi!/ddPsi! fields, confirmed present and NOT on
# the forbidden list) rather than continuing to read the legacy bundle's
# scratch fields.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_contraction_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_hessian_2026-07-31.jl"))
using Random, LinearAlgebra

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
@assert ctx.obj isa OperatorPsiBundle "expected a genuine operator bundle, not the legacy dense one"
D = ctx.D
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(1082026 + 1)

cf = build_compressed_factual(θ0, ctx; check_ties = false)
ncolI = cf.oci - 1
n = 1 + ncolI
W = cf.W
println("D=$D ncolI=$ncolI n=$n W=$W")

# f_new(x) = mean_w(Psi(q_new_w)) + zeta, q_new_w = -zeta - homogeneous_dual_contraction(lambda,...)[w]
function f_new(x::AbstractVector{Float64})
    ζ = x[1]; λ = @view x[2:end]
    t = homogeneous_dual_contraction(λ, cf, ctx, θ0)
    q = -ζ .- t
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return sum(Psi_q) / W + ζ
end

x0 = vcat(0.0, zeros(ncolI))   # a representative, not-too-extreme evaluation point
rng2 = MersenneTwister(999)
x0[2:end] .= 0.05 .* randn(rng2, ncolI)

println("\n" * "="^78); println("TEST 1: analytic homogeneous_winner_pair_hessian! vs. finite-difference Hessian of f_new"); println("="^78)
h_fd_ε = 1e-4
H_fd = zeros(n, n)
for i in 1:n
    for j in i:n
        xpp = copy(x0); xpp[i] += h_fd_ε; xpp[j] += h_fd_ε
        xpm = copy(x0); xpm[i] += h_fd_ε; xpm[j] -= h_fd_ε
        xmp = copy(x0); xmp[i] -= h_fd_ε; xmp[j] += h_fd_ε
        xmm = copy(x0); xmm[i] -= h_fd_ε; xmm[j] -= h_fd_ε
        if i == j
            xp = copy(x0); xp[i] += h_fd_ε
            xm = copy(x0); xm[i] -= h_fd_ε
            H_fd[i, i] = (f_new(xp) - 2 * f_new(x0) + f_new(xm)) / h_fd_ε^2
        else
            v = (f_new(xpp) - f_new(xpm) - f_new(xmp) + f_new(xmm)) / (4 * h_fd_ε^2)
            H_fd[i, j] = v; H_fd[j, i] = v
        end
    end
end

# analytic path: set obj.arg0 = q_new(x0), matching winner_pair_hessian!'s documented precondition
obj = ctx.obj
ζ0 = x0[1]; λ0 = @view x0[2:end]
t0v = homogeneous_dual_contraction(λ0, cf, ctx, θ0)
obj.arg0 .= -ζ0 .- t0v
wctx = build_homogeneous_winner_pair_ctx(cf, ctx, θ0)
h_packed = zeros(n * (n + 1) ÷ 2)
homogeneous_winner_pair_hessian!(h_packed, obj, wctx)

H_analytic = zeros(n, n)
k = 1
for i in 1:n
    for j in i:n
        H_analytic[i, j] = h_packed[k]; H_analytic[j, i] = h_packed[k]
        global k += 1
    end
end

maxdiff = maximum(abs.(H_analytic .- H_fd))
relscale = maximum(abs.(H_fd))
println("max|H_analytic - H_fd| = $maxdiff   (scale ~$relscale, FD step $h_fd_ε => expect ~1e-6 to 1e-4 relative FD error)")

println("--- DIAGNOSTIC: block-by-block ---")
println("H[1,1] analytic=$(H_analytic[1,1])  fd=$(H_fd[1,1])  diff=$(abs(H_analytic[1,1]-H_fd[1,1]))")
row1diff = maximum(abs.(H_analytic[1, 2:end] .- H_fd[1, 2:end]))
println("row1 (zeta,lambda_j) max diff = $row1diff")
blockdiff = maximum(abs.(H_analytic[2:end, 2:end] .- H_fd[2:end, 2:end]))
println("(lambda_i,lambda_j) block max diff = $blockdiff")
diagblockdiff = maximum(abs.(diag(H_analytic[2:end, 2:end]) .- diag(H_fd[2:end, 2:end])))
println("(lambda_i,lambda_i) DIAGONAL max diff = $diagblockdiff")
offdiagmask = H_analytic[2:end, 2:end] .- Diagonal(diag(H_analytic[2:end, 2:end]))
offdiagmask_fd = H_fd[2:end, 2:end] .- Diagonal(diag(H_fd[2:end, 2:end]))
offdiagdiff = maximum(abs.(offdiagmask .- offdiagmask_fd))
println("(lambda_i,lambda_j) OFF-DIAGONAL max diff = $offdiagdiff")
worst = argmax(abs.(H_analytic .- H_fd))
println("worst entry at $worst: analytic=$(H_analytic[worst])  fd=$(H_fd[worst])")

@assert maxdiff < 1e-3 * max(1.0, relscale) "TEST 1 FAILED: homogeneous Hessian does not match finite differences"
println("PASS -- homogeneous H_EE matches a finite-difference Hessian of the independently-verified dual objective")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
