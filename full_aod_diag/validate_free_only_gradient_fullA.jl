# ============================================================================
# Section 4's REQUIRED validation, which the end-to-end D=4 KNITRO comparison
# does NOT actually perform: verify the free-only envelope divergence
# gradient (div_grad_fn! from run_fullA_D4_production.jl) against
#   (1) the existing dense-Jacobian-then-contract path (production
#       PsiObjectiveBundleImplicit's own gradient, at the SAME point, real
#       inner solve, not a fixed synthetic context), and
#   (2) central finite differences of the fixed-inner envelope scalar,
# at several points -- BEFORE trusting any KNITRO end-to-end comparison,
# per spec: "Do not merge the new gradient path if it fails validation."
#
# Run: julia --project=. full_aod_diag/validate_free_only_gradient_fullA.jl
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ADB = joinpath(dirname(@__DIR__), "full_aod_diag", "ad_benchmark")
include(joinpath(ADB, "setup_context.jl"))
include(joinpath(ADB, "derivative_core.jl"))

so, pp = build_ad_context()
D = so.D; bi = AD_PARAMS.baseIndex; σ = AD_PARAMS.σHat; μHat = pp.γ.μHat
@unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
Aod_offset = 3 + D

θ0_up = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
bounds = theoretical_gammaprime_bounds(γ, σ)
θ0_up[3+D] = clamp(θ0_up[3+D], bounds.γp_lo, bounds.γp_hi)

l_full = length(θ0_up)
free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))
fixed_idx = vcat(1, 2, collect(3:2+D))
fixed_vals = θ0_up[fixed_idx]
m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)

const OUTER_OPT = joinpath(dirname(@__DIR__), "full_aod_diag", "csw_outer_25.opt")
const INNER_OPT = joinpath(dirname(@__DIR__), "full_aod_diag", "ek_inner.opt")

obj = PsiObjectiveBundleImplicit(δ = 1.0, find_smallest = true, γ = γ,
    (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = l_full, U = U, N = AD_PARAMS.Jac_W,
    lower_limit = -50, use_cached_x = true,
    outer_loop_opt = OUTER_OPT, inner_loop_opt = INNER_OPT)

# Test points: the starting point, and a genuinely different point (small random perturbation of
# the free A-block, still economically sane / away from bounds)
Random.seed!(99)
θ_pert = copy(θ0_up)
θ_pert[free_idx] .*= (1.0 .+ 0.05 .* randn(length(free_idx)))
test_points = [("theta0_up", θ0_up), ("perturbed", θ_pert)]

for (label, θfull) in test_points
    println("\n================ POINT: ", label, " ================")
    x_free = CS.pack_free(θfull, m)

    # ---- (0) real inner solve at this point (shared by both gradient methods) ----
    objSol, inner_x, nStatus = CS.inner_loop_internal(obj, θfull)
    println("inner solve: nStatus=", nStatus, "  objSol=", objSol)
    ncon_inner = obj.d - obj.outer_constr_index + 2
    obj(inner_x, Float64[], Float64[]; constr = zeros(ncon_inner))
    λ = inner_x[2:end]
    ctx = (U = obj.U, γobj = obj.γ, λ = λ, arg1 = obj.arg1, d = obj.d, outer_constr_index = obj.outer_constr_index)

    # ---- (1) my new free-only gradient (ForwardDiff over x_free via reconstruct_full) ----
    f_free = x -> envelope_scalar_div_ctx(CS.reconstruct_full(x, m), ctx)
    g_free_new = ForwardDiff.gradient(f_free, x_free)
    println("free-only gradient (new): length=", length(g_free_new))

    # ---- (2) existing dense-Jacobian-then-contract path (production, FULL theta) ----
    # obj(inner_x, g_full, theta; jac=jac_full) computes the divergence-constraint row via
    # calculate_jac_θ! (dense Jacobian) + the envelope-theorem contraction -- production's
    # ACTUAL gradient computation, unchanged, called at the SAME point/inner solution.
    g_full_dense = zeros(l_full)
    jac_full = zeros(ncon_inner * l_full)
    obj(inner_x, g_full_dense, θfull; constr = zeros(ncon_inner), jac = jac_full)
    # production fills `jac .= (∂c_∂θ')[:]` where ∂c_∂θ has shape (ncon,l), so ∂c_∂θ' is (l,ncon)
    # and column-major flattening makes jac_full's layout l_full-major: jac_full[1:l_full] is the
    # FULL gradient of constraint row 1 (divergence budget), jac_full[l_full+1:2l_full] is row 2, etc.
    jac_full_mat = reshape(jac_full, l_full, ncon_inner)
    g_dense_free = jac_full_mat[free_idx, 1]
    println("dense-Jacobian gradient (restricted to free_idx): length=", length(g_dense_free))

    relerr_dense = norm(g_free_new .- g_dense_free) / max(norm(g_dense_free), 1e-300)
    println("(1) vs (2) [free-only vs dense-Jacobian, SAME free coordinates]: relerr = ", relerr_dense)
    @assert relerr_dense < 1e-8 "FREE-ONLY GRADIENT DOES NOT MATCH DENSE JACOBIAN -- DO NOT USE"
    println("  PASS (< 1e-8)")

    # ---- (3) central finite differences of the FIXED-INNER envelope scalar ----
    h = 1e-5
    Random.seed!(1234 + hash(label) % 1000)
    for trial in 1:3
        v = randn(length(x_free)); v ./= norm(v)
        fp = f_free(x_free .+ h .* v)
        fm = f_free(x_free .- h .* v)
        fd_directional = (fp - fm) / (2h)
        analytic_directional = dot(g_free_new, v)
        relerr_fd = abs(fd_directional - analytic_directional) / max(abs(fd_directional), 1e-8)
        @printf("  FD check trial %d: analytic=%.8f  FD=%.8f  relerr=%.3e\n", trial, analytic_directional, fd_directional, relerr_fd)
        @assert relerr_fd < 1e-4 "FREE-ONLY GRADIENT FAILS CENTRAL FD CHECK -- DO NOT USE"
    end
    println("  FD directional checks: PASS (< 1e-4)")
end

println("\n================ ALL GRADIENT VALIDATION PASSED ================")
