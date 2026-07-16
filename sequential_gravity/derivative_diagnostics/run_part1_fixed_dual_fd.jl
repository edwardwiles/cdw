# ============================================================================
# Part 1 driver: hard fixed-dual finite-difference envelope derivative.
# Runs end to end: (1.1) identity test Q(A_k,x_k*)=delta*, (1.2) coordinate +
# directional FD gradient, (1.4) adaptive step-size diagnostic with
# winner-switch counts, plus a quick sanity comparison against production's
# existing AD envelope gradient (the full Part-3 comparison table comes
# later; this is just enough to sanity-check Part 1 before building Part 2).
#
# Setup below duplicates (does not `include`, to avoid running the full outer
# KNITRO bound search) the relevant economy/theta_r0 construction from
# sequential_gravity/run_profiled_production.jl -- kept byte-identical in
# substance, shortened to just what Part 1 needs (no sequential loop, no
# destination inversion, no outer KNITRO search).
#
#   DVAL=4 WVAL=8000 julia --project=. sequential_gravity/derivative_diagnostics/run_part1_fixed_dual_fd.jl
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, Printf

const ROOT = dirname(dirname(@__DIR__))
include(joinpath(ROOT, "setup/include_setup.jl"))
include(joinpath(ROOT, "prestep/include_prestep.jl"))
include(joinpath(ROOT, "prepare_cc/include_prepare_cc.jl"))
include(joinpath(ROOT, "moments/include_moments.jl"))
include(joinpath(ROOT, "cc_algo/include_cc_algo.jl"))
include(joinpath(ROOT, "lfd/include_lfd.jl"))
include(joinpath(ROOT, "misc/include_misc.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "..", "focal_moments.jl"))
include(joinpath(@__DIR__, "..", "focal_moments_directgp.jl"))
include(joinpath(@__DIR__, "..", "profiled_gravity.jl"))
using .ProfiledGravity
CS.include(joinpath(@__DIR__, "..", "PsiObjectiveBundleImplicitMethodB.jl"))

include(joinpath(@__DIR__, "fixed_dual_criterion.jl"))
include(joinpath(@__DIR__, "fixed_dual_fd.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "4"))
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "1"))

params = (server=1, user=2, fakeData=FAKEDATA, DFake=DVAL, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=WVAL, seedU=888, importanceSampling=0, importanceSamplingFactor=2,
    stratifiedSampling=0, IndMomentOrder=5, θConstant=0, gravMoment=1, localGravityMoment=0,
    localGravityCrossMoment=0, GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0, NormalizeMoments=0,
    useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1,
    use_Jacobian=0, calc_δ_star_initial=1, Jac_W=WVAL, theta_init=0, runLFD=1, runLFDCounterFactual=1)

setup_output = master_setup(params)
@unpack data, counters = setup_output
D = setup_output.D
@assert D == DVAL
useParams = (; params..., D=D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)
γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat
λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ

θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
local θr0
let γf0 = θr0_orig[3], μ0 = θr0_orig[1]
    global θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
end
const KBOUNDS = theoretical_kappa_bounds(γ, σ)
@printf("D=%d W=%d  mu=%.6f sigma=%.6f  theta*=1/mu=%.6f  beta=theta*/(sigma-1)=%.6f\n",
    D, W, θr0[1], σ, 1 / θr0[1], (1 / θr0[1]) / (σ - 1))
@printf("gamma'_focal* (Frechet) = %.6f   kappa bounds=[%.6f,%.6f]\n", θr0[3], KBOUNDS.κ_min, KBOUNDS.κ_max)

const Acol_offset = 3
const D1 = D + 1   # baseline (gravity-blind) focal moment count

# ============================================================================
# 1.1: identity test, at the Frechet benchmark AND at a nonbenchmark target
# ============================================================================
println("\n" * "="^78); println(">>> PART 1.1: fixed-dual identity test  Q(A_k,x_k*) == delta*(A_k,gamma')"); println("="^78)

γp_lo, γp_hi = KBOUNDS.γp_lo, KBOUNDS.γp_hi
γp_targets = [θr0[3], θr0[3] - 0.15 * (θr0[3] - γp_lo), θr0[3] + 0.15 * (γp_hi - θr0[3])]

identity_results = NamedTuple[]
for γp in γp_targets
    θ = copy(θr0); θ[3] = γp
    r = test_fixed_dual_identity(θ, EK_moments_focal_norm_directgp!, D1, γ, U; l=length(θr0))
    @printf("gamma'_focal=%.6f  delta*=%.8f  Q(A_k,x_k*)=%.8f  abs_diff=%.3e  rel_diff=%.3e  nStatus=%d\n",
        γp, r.δ_star, r.Q_fixed, r.abs_diff, r.rel_diff, r.nStatus)
    push!(identity_results, (γp_target=γp, r...))
end
all_identity_ok = all(r.rel_diff < 1e-6 for r in identity_results)
println(">>> Identity test PASSES (rel_diff < 1e-6 at every target)? ", all_identity_ok)

# ============================================================================
# 1.2/1.3: coordinate + directional FD gradient at the Frechet target
# ============================================================================
println("\n" * "="^78); println(">>> PART 1.2: fixed-dual FD gradient (log-Acol space) at the Frechet target"); println("="^78)

θ0 = copy(θr0)
base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, D1, γ, U; l=length(θr0))
x0 = base.x_star
obj0 = base.obj
@printf("base solve: delta*=%.8f, nStatus=%d\n", base.δ_star, base.nStatus)

const H_FD = 1e-3   # working step (justified by the 1.4 plateau scan below)
t0 = time()
grad_log = fixed_dual_fd_gradient(θ0, γ, U, EK_moments_focal_norm_directgp!, D1, x0, H_FD; l=length(θr0), Acol_offset=Acol_offset, parallel=(Threads.nthreads() > 1))
t_serial = time() - t0
Acol0 = θ0[Acol_offset+1:Acol_offset+D]
grad_level = grad_log ./ Acol0   # d/dAcol = (1/Acol)*d/dlogAcol
@printf("FD gradient (log-Acol space), h=%.0e, runtime=%.3fs, nthreads=%d:\n", H_FD, t_serial, Threads.nthreads())
for o in 1:D
    @printf("  o=%2d  Acol0=%.5f  dQ/dlogAcol=% .6e  dQ/dAcol=% .6e\n", o, Acol0[o], grad_log[o], grad_level[o])
end

# ============================================================================
# 1.4: adaptive step-size diagnostic
# ============================================================================
println("\n" * "="^78); println(">>> PART 1.4: adaptive step-size diagnostic"); println("="^78)

hs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
# focus coordinate: the one with the largest |grad_log| (most informative for winner-switch structure)
r_focus = argmax(abs.(grad_log))
v_coord = zeros(D); v_coord[r_focus] = 1.0
@printf("\n--- single-coordinate direction, o=%d (largest |dQ/dlogAcol|=%.4e) ---\n", r_focus, grad_log[r_focus])
rows_coord = adaptive_stepsize_diagnostic(θ0, obj0, x0, v_coord, hs; Acol_offset=Acol_offset, moments_fn=EK_moments_focal_norm_directgp!, γobj=γ, U=U, D=D, d=D1)
@printf("%8s %10s %10s %14s %14s %14s %10s %10s %10s %8s\n", "h", "n_switch", "frac_sw", "central", "central_h/2", "central_2h", "vs_h/2", "vs_2h", "vs_fwd", "rt(s)")
for r in rows_coord
    @printf("%8.0e %10d %10.5f %14.6e %14.6e %14.6e %10.3e %10.3e %10.3e %8.4f\n",
        r.h, r.n_switch, r.frac_switch, r.central, r.central_half_h, r.central_2h,
        r.central_vs_halfh_reldiff, r.central_vs_2h_reldiff, r.central_vs_forward_reldiff, r.runtime_s)
end
plateau_coord = identify_stable_plateau(rows_coord)
if plateau_coord === nothing
    println(">>> No h in the tested grid satisfies the stability+switch-count criteria for this coordinate direction.")
else
    @printf(">>> Stable plateau (single-coordinate): h=%.0e (n_switch=%d, central=%.6e)\n", plateau_coord.h, plateau_coord.n_switch, plateau_coord.central)
end

Random.seed!(20260714)
v_rand = randn(D); v_rand ./= norm(v_rand)
@printf("\n--- random mixed-destination direction ---\n")
rows_rand = adaptive_stepsize_diagnostic(θ0, obj0, x0, v_rand, hs; Acol_offset=Acol_offset, moments_fn=EK_moments_focal_norm_directgp!, γobj=γ, U=U, D=D, d=D1)
@printf("%8s %10s %10s %14s %14s %14s %10s %10s %10s %8s\n", "h", "n_switch", "frac_sw", "central", "central_h/2", "central_2h", "vs_h/2", "vs_2h", "vs_fwd", "rt(s)")
for r in rows_rand
    @printf("%8.0e %10d %10.5f %14.6e %14.6e %14.6e %10.3e %10.3e %10.3e %8.4f\n",
        r.h, r.n_switch, r.frac_switch, r.central, r.central_half_h, r.central_2h,
        r.central_vs_halfh_reldiff, r.central_vs_2h_reldiff, r.central_vs_forward_reldiff, r.runtime_s)
end
plateau_rand = identify_stable_plateau(rows_rand)
if plateau_rand === nothing
    println(">>> No h in the tested grid satisfies the stability+switch-count criteria for the random direction.")
else
    @printf(">>> Stable plateau (random direction): h=%.0e (n_switch=%d, central=%.6e)\n", plateau_rand.h, plateau_rand.n_switch, plateau_rand.central)
end

# ============================================================================
# Quick sanity comparison against production's existing AD envelope gradient
# (full Part-3 table comes later -- this just checks Part 1's own output is
# sane before building Part 2).
# ============================================================================
println("\n" * "="^78); println(">>> Sanity check: fixed-dual FD gradient vs production's existing AD gradient"); println("="^78)

obj0.moments!(@view(obj0.H[:, 1]), CS.select_G_from_H(obj0, obj0.H), θ0, obj0.U, obj0)
obj0.H[:, 2] .= 1.0
obj0(x0, zeros(length(x0)))   # nonempty g forces a fresh dPsi!(arg1,arg0) refresh at (theta0,x0)
λ_x0 = collect(@view x0[2:end])
Usub = U[1:params.Jac_W, :]
ad_grad_full_raw = ForwardDiff.gradient(θθ -> CS._methodB_envelope_scalar(θθ, EK_moments_focal_norm_directgp!, γ, Usub, λ_x0, obj0.arg1, D1, D1 + 1), θ0)
# _methodB_envelope_scalar returns d(constr[1])/dtheta where constr[1]=-f_raw*1e10 (gradient of the
# 1e10-SCALED OUTER CONSTRAINT, not of f_raw itself): d(f_raw)/dtheta = -ad_grad_full_raw/1e10.
# dual_criterion_fixed_x (and hence grad_log/FD) returns sign_conv*f_raw (sign_conv = -1 iff
# find_smallest, matching inner_loop's own convention) -- apply the SAME sign here for an
# apples-to-apples comparison against the FD gradient.
sign_conv = obj0.find_smallest ? -1.0 : 1.0
ad_grad_full = sign_conv .* (-ad_grad_full_raw ./ 1e10)
ad_grad_level = ad_grad_full[Acol_offset+1:Acol_offset+D]   # production's raw envelope gradient is w.r.t. LEVEL theta directly
ad_grad_log = ad_grad_level .* Acol0                          # convert to log-space for apples-to-apples vs grad_log

@printf("%4s %12s %14s %14s %14s\n", "o", "Acol0", "AD(log-space)", "FD(log-space)", "ratio AD/FD")
for o in 1:D
    @printf("%4d %12.5f %14.6e %14.6e %14.4f\n", o, Acol0[o], ad_grad_log[o], grad_log[o], ad_grad_log[o] / grad_log[o])
end

println("\nPART 1 DONE")
