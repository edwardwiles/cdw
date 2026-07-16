# ============================================================================
# Part 4: Monte Carlo stability across draw counts and seeds, plus the
# sample-exact Frechet negative control.
#
#   julia --project=. sequential_gravity/derivative_diagnostics/run_part4_mc_stability.jl
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
include(joinpath(@__DIR__, "full_profile_resolve.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "4"))
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "1"))
const Acol_offset = 3

"Build the economy (data fixed via seedFakeData=889; MC draws via seedU)."
function build_economy(D::Int, W::Int, seedU::Int)
    params = (server=1, user=2, fakeData=FAKEDATA, DFake=D, seedFakeData=889, counterType=1, counterExplicit=0,
        θHat=0, σHat=2.5, baseIndex=2, W=W, seedU=seedU, importanceSampling=0, importanceSamplingFactor=2,
        stratifiedSampling=0, IndMomentOrder=5, θConstant=0, gravMoment=1, localGravityMoment=0,
        localGravityCrossMoment=0, GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
        useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
        ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0, NormalizeMoments=0,
        useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1,
        use_Jacobian=0, calc_δ_star_initial=1, Jac_W=W, theta_init=0, runLFD=1, runLFDCounterFactual=1)
    so = master_setup(params)
    @assert so.D == D
    useParams = (; params..., D=D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    ps = master_prestep(so.data, so.counters, useParams)
    prep = master_prepare_cc(so.data, so.counters, ps, useParams)
    γ = prep.γ; U = prep.U; focal = params.baseIndex; σ = params.σHat
    θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
    γf0 = θr0_orig[3]; μ0 = θr0_orig[1]
    θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
    KBOUNDS = theoretical_kappa_bounds(γ, σ)
    β = (1 / μ0) / (σ - 1)
    return (γ=γ, U=U, θr0=θr0, KBOUNDS=KBOUNDS, β=β, σ=σ, focal=focal, D=D, W=W, params=params)
end

function directions_for(D::Int, grad_fd::Vector{Float64}, grad_boundary::Vector{Float64}, n_random::Int; seed::Int=20260715)
    dirs = Tuple{String,Vector{Float64}}[]
    push!(dirs, ("neg_fd_grad", -normalize(grad_fd)))
    push!(dirs, ("neg_boundary_grad", -normalize(grad_boundary)))
    for o in 1:D
        e = zeros(D); e[o] = 1.0
        push!(dirs, ("coord_$o", e))
    end
    Random.seed!(seed)
    for i in 1:n_random
        push!(dirs, ("random_$i", normalize(randn(D))))
    end
    return dirs
end

"One full comparison pass (Part-3-style) at a given economy, returns rows + summary."
function run_comparison(eco, γp_targets::Vector{Float64}, hs::Vector{Float64}, n_random::Int)
    γ, U, θr0, D, D1 = eco.γ, eco.U, eco.θr0, eco.D, eco.D + 1
    β = eco.β
    rows = NamedTuple[]
    n_solves = 0
    for γp in γp_targets
        θ0 = copy(θr0); θ0[3] = γp
        base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, D1, γ, U; l=length(θr0))
        x0 = base.x_star; obj0 = base.obj
        grad_fd = fixed_dual_fd_gradient(θ0, γ, U, EK_moments_focal_norm_directgp!, D1, x0, 0.1; l=length(θr0), Acol_offset=Acol_offset)
        obj0.moments!(@view(obj0.H[:, 1]), CS.select_G_from_H(obj0, obj0.H), θ0, obj0.U, obj0)
        obj0.H[:, 2] .= 1.0
        obj0(x0, zeros(length(x0)))
        grad_boundary, _ = boundary_envelope_gradient(θ0, obj0, x0, γ, U, D, β; Acol_offset=Acol_offset)
        λ_x0 = collect(@view x0[2:end])
        ad_grad_full_raw = ForwardDiff.gradient(θθ -> CS._methodB_envelope_scalar(θθ, EK_moments_focal_norm_directgp!, γ, U, λ_x0, obj0.arg1, D1, D1 + 1), θ0)
        sign_conv = obj0.find_smallest ? -1.0 : 1.0
        Acol0 = θ0[Acol_offset+1:Acol_offset+D]
        grad_ad = (sign_conv .* (-ad_grad_full_raw ./ 1e10)[Acol_offset+1:Acol_offset+D]) .* Acol0

        dirs = directions_for(D, grad_fd, grad_boundary, n_random)
        for (name, v) in dirs, h in hs
            ad_slope = dot(grad_ad, v)
            fd_slope = fixed_dual_fd_directional_derivative(θ0, obj0, x0, v, h; Acol_offset=Acol_offset)
            bnd_slope = dot(grad_boundary, v)
            profile_slope, δp, δm, statusp, statusm = full_profile_directional_slope(θ0, γ, U, D1, EK_moments_focal_norm_directgp!, v, h; Acol_offset=Acol_offset)
            n_solves += 2
            push!(rows, (γp_target=γp, direction_id=name, h=h, ad_slope=ad_slope, fd_slope=fd_slope, bnd_slope=bnd_slope,
                profile_slope=profile_slope, δp=δp, δm=δm, statusp=statusp, statusm=statusm))
        end
    end
    return rows, n_solves
end

relerr(a, b) = abs(a - b) / max(abs(b), 1e-8)

function summarize(rows, γp_frechet)
    nondeg = [r for r in rows if r.γp_target != γp_frechet && r.statusp ∈ (0, -100, -101, -103) && r.statusm ∈ (0, -100, -101, -103)]
    fd_e = [relerr(r.fd_slope, r.profile_slope) for r in nondeg]
    bnd_e = [relerr(r.bnd_slope, r.profile_slope) for r in nondeg]
    ad_e = [relerr(r.ad_slope, r.profile_slope) for r in nondeg]
    return (n=length(nondeg), fd_median=median(fd_e), fd_mean=mean(fd_e), fd_max=maximum(fd_e),
        bnd_median=median(bnd_e), bnd_mean=mean(bnd_e), bnd_max=maximum(bnd_e),
        ad_median=median(ad_e), ad_mean=mean(ad_e), ad_max=maximum(ad_e))
end

# ============================================================================
# 4a: MC stability across W
# ============================================================================
println("="^78); println(">>> PART 4a: Monte Carlo stability across draw counts"); println("="^78)

W_grid = [8_000, 32_000, 128_000, 800_000]
hs = [0.03, 0.1]
stability_rows = NamedTuple[]
for W in W_grid
    n_random = W >= 800_000 ? 4 : 10   # scale back directions at the largest W to control runtime
    t0 = time()
    eco = build_economy(DVAL, W, 888)
    γp_lo, γp_hi = eco.KBOUNDS.γp_lo, eco.KBOUNDS.γp_hi
    γp_frechet = eco.θr0[3]
    γp_targets = [γp_frechet, γp_frechet - 0.15 * (γp_frechet - γp_lo), γp_frechet + 0.15 * (γp_hi - γp_frechet)]
    rows, n_solves = run_comparison(eco, γp_targets, hs, n_random)
    s = summarize(rows, γp_frechet)
    dt = time() - t0
    @printf("\nW=%8d (n_random=%2d, %d full-profile solves, %.1fs):\n", W, n_random, n_solves, dt)
    @printf("  FD  vs profile (nondegenerate targets): median=%.4f mean=%.4f max=%.4f\n", s.fd_median, s.fd_mean, s.fd_max)
    @printf("  BND vs profile (nondegenerate targets): median=%.4f mean=%.4f max=%.4f\n", s.bnd_median, s.bnd_mean, s.bnd_max)
    @printf("  AD  vs profile (nondegenerate targets): median=%.4f mean=%.4f max=%.4f\n", s.ad_median, s.ad_mean, s.ad_max)
    push!(stability_rows, (W=W, n_random=n_random, n_solves=n_solves, runtime_s=dt, s...))
    open(joinpath(@__DIR__, "part4_rows_D$(DVAL)_W$(W).csv"), "w") do io
        println(io, "gammap_target,direction_id,h,ad_slope,fd_slope,bnd_slope,profile_slope,deltap,deltam,statusp,statusm")
        for r in rows
            println(io, "$(r.γp_target),$(r.direction_id),$(r.h),$(r.ad_slope),$(r.fd_slope),$(r.bnd_slope),$(r.profile_slope),$(r.δp),$(r.δm),$(r.statusp),$(r.statusm)")
        end
    end
end

open(joinpath(@__DIR__, "part4_stability_summary.csv"), "w") do io
    println(io, "W,n_random,n_solves,runtime_s,n,fd_median,fd_mean,fd_max,bnd_median,bnd_mean,bnd_max,ad_median,ad_mean,ad_max")
    for r in stability_rows
        println(io, "$(r.W),$(r.n_random),$(r.n_solves),$(r.runtime_s),$(r.n),$(r.fd_median),$(r.fd_mean),$(r.fd_max),$(r.bnd_median),$(r.bnd_mean),$(r.bnd_max),$(r.ad_median),$(r.ad_mean),$(r.ad_max)")
    end
end

# ============================================================================
# 4b: seed sensitivity at a moderate W
# ============================================================================
println("\n" * "="^78); println(">>> PART 4b: seed sensitivity (W=32000, 3 seeds)"); println("="^78)

W_seed = 32_000
seeds = [888, 999, 12345]
seed_grads = NamedTuple[]
for sd in seeds
    eco = build_economy(DVAL, W_seed, sd)
    θ0 = copy(eco.θr0)   # Frechet target
    base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, DVAL + 1, eco.γ, eco.U; l=length(θ0))
    x0 = base.x_star; obj0 = base.obj
    grad_fd = fixed_dual_fd_gradient(θ0, eco.γ, eco.U, EK_moments_focal_norm_directgp!, DVAL + 1, x0, 0.1; l=length(θ0), Acol_offset=Acol_offset)
    obj0.moments!(@view(obj0.H[:, 1]), CS.select_G_from_H(obj0, obj0.H), θ0, obj0.U, obj0)
    obj0.H[:, 2] .= 1.0
    obj0(x0, zeros(length(x0)))
    grad_bnd, _ = boundary_envelope_gradient(θ0, obj0, x0, eco.γ, eco.U, DVAL, eco.β; Acol_offset=Acol_offset)
    @printf("seed=%6d  delta*=%.3e  grad_fd=%s\n  grad_bnd=%s\n", sd, base.δ_star, string(round.(grad_fd, sigdigits=4)), string(round.(grad_bnd, sigdigits=4)))
    push!(seed_grads, (seed=sd, grad_fd=grad_fd, grad_bnd=grad_bnd))
end
fd_stack = hcat([g.grad_fd for g in seed_grads]...)
bnd_stack = hcat([g.grad_bnd for g in seed_grads]...)
@printf("\ncross-seed coefficient of variation (std/|mean|) per coordinate:\n")
for o in 1:DVAL
    cv_fd = std(fd_stack[o, :]) / abs(mean(fd_stack[o, :]))
    cv_bnd = std(bnd_stack[o, :]) / abs(mean(bnd_stack[o, :]))
    @printf("  o=%d  FD cv=%.4f   BND cv=%.4f\n", o, cv_fd, cv_bnd)
end

# ============================================================================
# 4c: sample-exact Frechet negative control
# ============================================================================
println("\n" * "="^78); println(">>> PART 4c: sample-exact Frechet negative control"); println("="^78)

for W in W_grid
    eco = build_economy(DVAL, W, 888)
    θ0 = copy(eco.θr0)
    D1 = DVAL + 1
    Gmean = let
        Kt = zeros(W); Gt = zeros(W, D1)
        EK_moments_focal_norm_directgp!(Kt, Gt, θ0, eco.U, (γ=eco.γ,))
        vec(mean(Gt, dims=1))
    end
    base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, D1, eco.γ, eco.U; l=length(θ0))
    @printf("W=%8d  max|E_F[G]| (sample-feasibility check) = %.3e   delta*(A*,gamma'_frechet) = %.3e   identity rel_diff=%.3e\n",
        W, maximum(abs.(Gmean)), base.δ_star, base.rel_diff)
end

# scan directions at the Frechet target for W=128000, confirm no negative divergence
eco128 = build_economy(DVAL, 128_000, 888)
θ0 = copy(eco128.θr0)
D1 = DVAL + 1
base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, D1, eco128.γ, eco128.U; l=length(θ0))
x0 = base.x_star
grad_fd = fixed_dual_fd_gradient(θ0, eco128.γ, eco128.U, EK_moments_focal_norm_directgp!, D1, x0, 0.1; l=length(θ0), Acol_offset=Acol_offset)
obj_tmp = build_fixed_dual_bundle(eco128.γ, eco128.U, length(θ0), D1, EK_moments_focal_norm_directgp!)
obj_tmp.moments!(@view(obj_tmp.H[:, 1]), CS.select_G_from_H(obj_tmp, obj_tmp.H), θ0, obj_tmp.U, obj_tmp)
obj_tmp.H[:, 2] .= 1.0
obj_tmp(x0, zeros(length(x0)))
grad_bnd, _ = boundary_envelope_gradient(θ0, obj_tmp, x0, eco128.γ, eco128.U, DVAL, eco128.β; Acol_offset=Acol_offset)
dirs = directions_for(DVAL, grad_fd, grad_bnd, 10)
hs_neg = [0.001, 0.01, 0.03, 0.1, 0.3]
min_delta = Inf
n_checked = 0
for (name, v) in dirs, h in hs_neg
    _, δp, δm, statusp, statusm = full_profile_directional_slope(θ0, eco128.γ, eco128.U, D1, EK_moments_focal_norm_directgp!, v, h; Acol_offset=Acol_offset)
    global min_delta = min(min_delta, δp, δm)
    global n_checked += 2
    if δp < -1e-6 || δm < -1e-6
        @printf("  !! NEGATIVE DIVERGENCE FOUND: dir=%s h=%.3f deltap=%.3e deltam=%.3e status=(%d,%d)\n", name, h, δp, δm, statusp, statusm)
    end
end
@printf("\nChecked %d re-solved points around A* at the Frechet target: min delta* observed = %.3e (should be >= ~0)\n", n_checked, min_delta)
@printf(">>> No negative divergence found? %s\n", min_delta >= -1e-6)

println("\nPART 4 DONE")
