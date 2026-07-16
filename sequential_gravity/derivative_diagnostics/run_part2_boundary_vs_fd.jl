# ============================================================================
# Part 2 (continued): compare the full dual-integrand conditional
# winner-boundary gradient against the (already-validated, Part 1) fixed-dual
# FD gradient, and against production's existing AD gradient, at the Frechet
# benchmark and at nonbenchmark gamma' targets.
#
#   DVAL=4 WVAL=128000 julia --project=. sequential_gravity/derivative_diagnostics/run_part2_boundary_vs_fd.jl
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
include(joinpath(@__DIR__, "boundary_derivative.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "4"))
const WVAL = parse(Int, get(ENV, "WVAL", "128000"))
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "1"))
const H_FD = parse(Float64, get(ENV, "H_FD", "0.1"))

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

θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
local θr0
let γf0 = θr0_orig[3], μ0 = θr0_orig[1]
    global θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
end
const KBOUNDS = theoretical_kappa_bounds(γ, σ)
μ0 = θr0[1]; β = (1 / μ0) / (σ - 1)
const Acol_offset = 3; const D1 = D + 1
@printf("D=%d W=%d  mu=%.6f sigma=%.6f beta=%.6f  H_FD=%.3g\n", D, W, μ0, σ, β, H_FD)

γp_lo, γp_hi = KBOUNDS.γp_lo, KBOUNDS.γp_hi
γp_targets = [θr0[3], θr0[3] - 0.15 * (θr0[3] - γp_lo), θr0[3] + 0.15 * (γp_hi - θr0[3])]

rows = NamedTuple[]
for γp in γp_targets
    println("\n" * "="^78); @printf(">>> gamma'_focal target = %.6f (Frechet = %.6f)\n", γp, θr0[3]); println("="^78)
    θ0 = copy(θr0); θ0[3] = γp
    base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, D1, γ, U; l=length(θr0))
    x0 = base.x_star; obj0 = base.obj
    @printf("delta*=%.8f  identity rel_diff=%.3e  nStatus=%d\n", base.δ_star, base.rel_diff, base.nStatus)

    t0 = time()
    grad_fd = fixed_dual_fd_gradient(θ0, γ, U, EK_moments_focal_norm_directgp!, D1, x0, H_FD; l=length(θr0), Acol_offset=Acol_offset)
    t_fd = time() - t0

    obj0.moments!(@view(obj0.H[:, 1]), CS.select_G_from_H(obj0, obj0.H), θ0, obj0.U, obj0)
    obj0.H[:, 2] .= 1.0
    obj0(x0, zeros(length(x0)))
    t0 = time()
    grad_boundary, diagpieces = boundary_envelope_gradient(θ0, obj0, x0, γ, U, D, β; Acol_offset=Acol_offset)
    t_boundary = time() - t0

    λ_x0 = collect(@view x0[2:end])
    Usub = U[1:params.Jac_W, :]
    t0 = time()
    ad_grad_full_raw = ForwardDiff.gradient(θθ -> CS._methodB_envelope_scalar(θθ, EK_moments_focal_norm_directgp!, γ, Usub, λ_x0, obj0.arg1, D1, D1 + 1), θ0)
    t_ad = time() - t0
    sign_conv = obj0.find_smallest ? -1.0 : 1.0
    ad_grad_level = sign_conv .* (-ad_grad_full_raw ./ 1e10)[Acol_offset+1:Acol_offset+D]
    Acol0 = θ0[Acol_offset+1:Acol_offset+D]
    ad_grad_log = ad_grad_level .* Acol0

    @printf("\nruntimes (s): FD(D coords)=%.4f  boundary(D coords)=%.4f  AD(1 call)=%.4f\n", t_fd, t_boundary, t_ad)
    @printf("%4s %12s %14s %14s %14s %12s %12s\n", "o", "Acol0", "AD", "fixed_dual_FD", "boundary", "bnd/FD", "AD/FD")
    for o in 1:D
        @printf("%4d %12.5f %14.6e %14.6e %14.6e %12.4f %12.4f\n",
            o, Acol0[o], ad_grad_log[o], grad_fd[o], grad_boundary[o], grad_boundary[o] / grad_fd[o], ad_grad_log[o] / grad_fd[o])
        push!(rows, (γp_target=γp, o=o, ad=ad_grad_log[o], fixed_dual_fd=grad_fd[o], boundary=grad_boundary[o],
            intensive_part=diagpieces.intensive[o], boundary_part=diagpieces.boundary[o],
            bnd_vs_fd_reldiff=abs(grad_boundary[o] - grad_fd[o]) / max(abs(grad_fd[o]), 1e-12),
            ad_vs_fd_reldiff=abs(ad_grad_log[o] - grad_fd[o]) / max(abs(grad_fd[o]), 1e-12)))
    end
    max_bnd_err = maximum(r.bnd_vs_fd_reldiff for r in rows if r.γp_target == γp)
    @printf("\nmax |boundary - FD|/|FD| over coordinates at this target: %.3e\n", max_bnd_err)
end

open(joinpath(@__DIR__, "part2_comparison_D$(DVAL)_W$(WVAL).csv"), "w") do io
    println(io, "gammap_target,o,ad,fixed_dual_fd,boundary,intensive_part,boundary_part,bnd_vs_fd_reldiff,ad_vs_fd_reldiff")
    for r in rows
        println(io, "$(r.γp_target),$(r.o),$(r.ad),$(r.fixed_dual_fd),$(r.boundary),$(r.intensive_part),$(r.boundary_part),$(r.bnd_vs_fd_reldiff),$(r.ad_vs_fd_reldiff)")
    end
end

println("\nPART 2 COMPARISON DONE")
