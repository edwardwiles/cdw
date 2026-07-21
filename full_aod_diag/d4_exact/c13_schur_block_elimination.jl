# Continuation (branch diag/fullA-d4-exact-cm-hessian-arch), Section 10:
# numerical block elimination experiment. NOT a new Hessian architecture --
# a linear-algebra reorganization of how the SAME Newton system H*z=-g is
# solved, applied to Architecture C's (already block-structured) Hessian at
# L=50, calibration point, converged (zeta*,lambda*). Compares:
#   (1) direct Cholesky factorization + solve of the full n x n H
#   (2) Schur-complement block elimination: factor H_CC (the LARGE block,
#       ncm x ncm -- ncm=150 > NCORE=18 at L=50), form S = H_EE - H_EC H_CC^-1 H_EC',
#       factor S (small, NCORE x NCORE), recover the CM multiplier step z_C via
#       back-substitution through the H_CC factor.
# This does NOT claim a closed form for the CM multipliers -- the CM moments
# still vary nonlinearly with the draws inside the CC conjugate; this is a
# linear-algebra reorganization of ONE Newton system solve at a fixed point,
# nothing more.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
using Printf, LinearAlgebra, BenchmarkTools, Statistics, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)

L = 50
aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
obj = aug.obj_cm
NCORE = aug.ncore; ncm = aug.ncm; n = obj.outer_constr_index
println("L=$L  NCORE=$NCORE  ncm=$ncm  n=$n")

# ---- converge to the actual inner optimum (Architecture A, dense, trusted) first ----
obj.x .= NaN
K, x_star, nStatus, n_fg, n_hess = inner_loop_internal_profiled(obj, θ_full)
@assert nStatus in (0, -100, -101, -103) "inner solve failed nStatus=$nStatus"
println("converged: nStatus=$nStatus  ||x*||=", norm(x_star))

# ---- Hessian + gradient at x* (converged point -- gradient should be ~0 at the KKT point
#      for the UNCONSTRAINED (no active bounds) directions; still a valid, real RHS to
#      exercise the linear solve, not a synthetic one) ----
g = zeros(n)
obj(x_star, g)   # length(g)>0, length(theta)==0 branch: fills g w.r.t (zeta,lambda)
@printf("||g(x*)||_inf = %.3e  (near 0 confirms x* is a genuine KKT point)\n", maximum(abs.(g)))

hpacked = Vector{Float64}(undef, n*(n+1)÷2)
obj(x_star, h = hpacked)
function unpack_packed(h::AbstractVector, n::Int)
    Mm = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mm[i, j] = h[k]; Mm[j, i] = h[k]
        k += 1
    end
    return Mm
end
H = unpack_packed(hpacked, n)
H_EE = H[1:NCORE, 1:NCORE]
H_EC = H[1:NCORE, NCORE+1:end]
H_CC = H[NCORE+1:end, NCORE+1:end]

# use a REAL RHS: the Newton step solves H*dz = -g (standard Newton direction at a
# near-optimal point); also test a random RHS to avoid conclusions specific to g≈0.
b_kkt = -g
Random.seed!(555)
b_rand = randn(n)

println("\ncond(H_full)  = ", cond(H))
println("cond(H_EE)    = ", cond(H_EE))
println("cond(H_CC)    = ", cond(H_CC))

function direct_solve(H, b)
    F = cholesky(Symmetric(H))
    z = F \ b
    return z
end

function schur_solve(H_EE, H_EC, H_CC, b, NCORE, ncm)
    b_E = @view b[1:NCORE]; b_C = @view b[NCORE+1:end]
    F_CC = cholesky(Symmetric(H_CC))
    X = F_CC \ Matrix(H_EC')            # ncm x NCORE  (H_CC^{-1} H_EC')
    S = H_EE .- H_EC * X                # NCORE x NCORE Schur complement
    F_S = cholesky(Symmetric(S))
    y = F_CC \ b_C                      # ncm
    rhs_E = b_E .- H_EC * y
    z_E = F_S \ rhs_E                   # NCORE
    z_C = F_CC \ (b_C .- H_EC' * z_E)   # ncm  -- "recovered CM multiplier step"
    return vcat(z_E, z_C), S
end

# ---- correctness: Schur-recovered z must match the direct solve to Cholesky/BLAS tolerance ----
for (label, b) in (("KKT (b=-g)", b_kkt), ("random", b_rand))
    z_direct = direct_solve(H, b)
    z_schur, S = schur_solve(H_EE, H_EC, H_CC, b, NCORE, ncm)
    err = maximum(abs.(z_direct .- z_schur))
    @printf("\n[%s] max|z_direct - z_schur| = %.3e   cond(Schur complement S) = %.4e\n", label, err, cond(S))
end

# ---- timing: factorization-only, solve-only, total (BenchmarkTools, several samples) ----
println("\n--- timing (BenchmarkTools, median of samples) ---")

t_direct_fact = @belapsed cholesky(Symmetric($H))
F = cholesky(Symmetric(H))
t_direct_solve = @belapsed $F \ $b_kkt

t_schur_fact = @belapsed begin
    F_CC = cholesky(Symmetric($H_CC))
    X = F_CC \ Matrix($H_EC')
    S = $H_EE .- $H_EC * X
    F_S = cholesky(Symmetric(S))
end
F_CC = cholesky(Symmetric(H_CC))
X = F_CC \ Matrix(H_EC')
S = H_EE .- H_EC * X
F_S = cholesky(Symmetric(S))
t_schur_solve = @belapsed begin
    b_E = @view($b_kkt[1:$NCORE]); b_C = @view($b_kkt[$NCORE+1:end])
    y = $F_CC \ b_C
    rhs_E = b_E .- $H_EC * y
    z_E = $F_S \ rhs_E
    z_C = $F_CC \ (b_C .- $H_EC' * z_E)
end

@printf("Direct:  factorize=%.3e s   solve=%.3e s   total=%.3e s\n", t_direct_fact, t_direct_solve, t_direct_fact + t_direct_solve)
@printf("Schur:   factorize=%.3e s   solve=%.3e s   total=%.3e s\n", t_schur_fact, t_schur_solve, t_schur_fact + t_schur_solve)
@printf("Schur/Direct total ratio = %.3fx  (>1 means Schur elimination is SLOWER)\n", (t_schur_fact + t_schur_solve) / (t_direct_fact + t_direct_solve))

# ---- also time factoring ONLY H_CC in isolation vs the full H, to make explicit that
#      H_CC (ncm x ncm = 150x150) is the dominant cost, not H_EE (18x18) ----
t_HCC_fact_only = @belapsed cholesky(Symmetric($H_CC))
t_HEE_fact_only = @belapsed cholesky(Symmetric($H_EE))
@printf("\ncholesky(H_CC) alone = %.3e s   cholesky(H_EE) alone = %.3e s   cholesky(H_full) = %.3e s\n",
        t_HCC_fact_only, t_HEE_fact_only, t_direct_fact)
