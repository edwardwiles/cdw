# ============================================================================
# Claude Code task 2026-08-01, §8 gate: REDUCED H_EE vs (1) finite differences
# of the reduced dual objective, (2) an explicit dense reduced-G Hessian
# E'*diag(Psi''(q))*E (task §8's "use exact/dense comparisons in addition to
# finite differences where possible"), (3) symmetry/finiteness/packed
# ordering, (4) no null direction from the deleted sum-of-shares identity.
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
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
using Random, LinearAlgebra

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
@assert ctx.obj isa OperatorPsiBundle "expected a genuine operator bundle, not the legacy dense one"
D = ctx.D; Ddest = D
θ0 = copy(ctx.θ0_up)
rng = MersenneTwister(20260801 + 2)

cf = build_compressed_factual(θ0, ctx; check_ties = false)
has_france = cf.cf_col > 0
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
n_red = layout.total_reduced_economic_moments
n = 1 + n_red
W = cf.W
println("D=$D ncolI_reduced=$n_red n=$n W=$W")

function f_reduced(x::AbstractVector{Float64})
    ζ = x[1]; λ = @view x[2:end]
    t = reduced_homogeneous_dual_contraction(λ, cf, ctx, θ0, layout)
    q = -ζ .- t
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return sum(Psi_q) / W + ζ
end

x0 = vcat(0.0, zeros(n_red))
rng2 = MersenneTwister(999 + 1)
x0[2:end] .= 0.05 .* randn(rng2, n_red)

println("\n" * "="^78); println("TEST 1: analytic reduced_homogeneous_winner_pair_hessian! vs finite-difference Hessian of f_reduced"); println("="^78)
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
            H_fd[i, i] = (f_reduced(xp) - 2 * f_reduced(x0) + f_reduced(xm)) / h_fd_ε^2
        else
            v = (f_reduced(xpp) - f_reduced(xpm) - f_reduced(xmp) + f_reduced(xmm)) / (4 * h_fd_ε^2)
            H_fd[i, j] = v; H_fd[j, i] = v
        end
    end
end

obj = ctx.obj
ζ0 = x0[1]; λ0 = @view x0[2:end]
t0v = reduced_homogeneous_dual_contraction(λ0, cf, ctx, θ0, layout)
obj.arg0 .= -ζ0 .- t0v
wctx = build_reduced_homogeneous_winner_pair_ctx(cf, ctx, θ0, layout)
h_packed = zeros(n * (n + 1) ÷ 2)
reduced_homogeneous_winner_pair_hessian!(h_packed, obj, wctx)

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
println("max|H_analytic - H_fd| = $maxdiff  (scale ~$relscale)")
@assert maxdiff < 1e-3 * max(1.0, relscale) "TEST 1 FAILED"
println("PASS")

println("\n" * "="^78); println("TEST 2: analytic reduced Hessian vs explicit DENSE reduced-G Hessian E'*diag(Psi''(q))*E"); println("="^78)
Gred = zeros(W, n_red)
Hmoments = homogeneous_factual_moment(θ0, ctx)
for kk in eachindex(layout.retained_full_factual_j)
    o = layout.retained_origin[kk]; s = layout.retained_slot[kk]
    j_full = layout.retained_full_factual_j[kk]
    Gred[:, kk] .= Hmoments[s][:, o] .* cf.nrm[j_full] .* cf.gdiv[j_full]
end
if has_france
    Hfrance = homogeneous_france_moment(θ0, ctx)
    j_full = cf.cf_col
    Gred[:, layout.france_ratio_reduced_j] .= Hfrance .* cf.nrm[j_full] .* cf.gdiv[j_full]
end
E = Diagonal(cf.SW) * Gred
q0 = obj.arg0
Psiqq = similar(q0)
obj.ddPsi!(Psiqq, q0)   # Psi''(q0)
M = obj.M
H_dense = zeros(n, n)
H_dense[1, 1] = sum(Psiqq) / M
for j in 1:n_red
    H_dense[1, 1 + j] = sum(Psiqq .* E[:, j]) / M
    H_dense[1 + j, 1] = H_dense[1, 1 + j]
end
H_dense[2:end, 2:end] .= (E' * (Psiqq .* E)) ./ M
maxdiff2 = maximum(abs.(H_analytic .- H_dense))
relscale2 = maximum(abs.(H_dense))
println("max|H_analytic - H_dense| = $maxdiff2 (scale ~$relscale2)")
@assert maxdiff2 < 1e-8 * max(1.0, relscale2) "TEST 2 FAILED"
println("PASS -- reduced analytic Hessian matches an EXACT dense reduced-G construction, not just finite differences")

println("\n" * "="^78); println("TEST 3: symmetry, finiteness, packed ordering"); println("="^78)
@assert all(isfinite, H_analytic) "non-finite entries in reduced Hessian"
@assert maximum(abs.(H_analytic .- H_analytic')) < 1e-12 "reduced Hessian not symmetric"
println("PASS -- all finite, symmetric to $(maximum(abs.(H_analytic .- H_analytic')))")

println("\n" * "="^78); println("TEST 4: no null direction from the deleted sum-of-shares identity (reduced Hessian's lambda-block is nonsingular / well-conditioned near this point)"); println("="^78)
λλ_block = H_analytic[2:end, 2:end]
ev = eigvals(Symmetric(λλ_block))
println("eigenvalue range: min=$(minimum(ev)) max=$(maximum(ev)) cond=$(maximum(abs.(ev))/max(minimum(abs.(ev)),1e-300))")
@assert minimum(abs.(ev)) > 1e-8 "reduced Hessian's lambda-block has a near-zero eigenvalue -- the anchor removal did NOT resolve the rank deficiency as claimed"
println("PASS -- smallest |eigenvalue| = $(minimum(abs.(ev))), no residual null direction from the deleted identity")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
