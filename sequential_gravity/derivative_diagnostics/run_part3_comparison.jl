# ============================================================================
# Part 3 driver: compare AD / fixed-dual FD / conditional boundary / fully
# re-solved profile FD slopes across gamma' targets and directions.
#
#   DVAL=4 WVAL=128000 julia --project=. sequential_gravity/derivative_diagnostics/run_part3_comparison.jl
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
const WVAL = parse(Int, get(ENV, "WVAL", "128000"))
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "1"))
const N_RANDOM_DIRS = parse(Int, get(ENV, "N_RANDOM_DIRS", "10"))
const HS_PROFILE = [parse(Float64, s) for s in split(get(ENV, "HS_PROFILE", "0.03,0.1"), ",")]

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
@printf("D=%d W=%d  mu=%.6f sigma=%.6f beta=%.6f\n", D, W, μ0, σ, β)
t_start = time()

γp_lo, γp_hi = KBOUNDS.γp_lo, KBOUNDS.γp_hi
γp_targets = [θr0[3], θr0[3] - 0.15 * (θr0[3] - γp_lo), θr0[3] + 0.15 * (γp_hi - θr0[3])]

Random.seed!(20260715)
random_dirs = [normalize(randn(D)) for _ in 1:N_RANDOM_DIRS]

all_rows = NamedTuple[]
n_full_solves = 0

for γp in γp_targets
    println("\n" * "="^78); @printf(">>> gamma'_focal target = %.6f (Frechet = %.6f)\n", γp, θr0[3]); println("="^78)
    θ0 = copy(θr0); θ0[3] = γp
    base = test_fixed_dual_identity(θ0, EK_moments_focal_norm_directgp!, D1, γ, U; l=length(θr0))
    x0 = base.x_star; obj0 = base.obj
    @printf("delta*=%.8f  identity rel_diff=%.3e  nStatus=%d\n", base.δ_star, base.rel_diff, base.nStatus)

    grad_fd = fixed_dual_fd_gradient(θ0, γ, U, EK_moments_focal_norm_directgp!, D1, x0, 0.1; l=length(θr0), Acol_offset=Acol_offset)

    obj0.moments!(@view(obj0.H[:, 1]), CS.select_G_from_H(obj0, obj0.H), θ0, obj0.U, obj0)
    obj0.H[:, 2] .= 1.0
    obj0(x0, zeros(length(x0)))
    grad_boundary, _ = boundary_envelope_gradient(θ0, obj0, x0, γ, U, D, β; Acol_offset=Acol_offset)

    λ_x0 = collect(@view x0[2:end])
    Usub = U[1:params.Jac_W, :]
    ad_grad_full_raw = ForwardDiff.gradient(θθ -> CS._methodB_envelope_scalar(θθ, EK_moments_focal_norm_directgp!, γ, Usub, λ_x0, obj0.arg1, D1, D1 + 1), θ0)
    sign_conv = obj0.find_smallest ? -1.0 : 1.0
    ad_grad_level = sign_conv .* (-ad_grad_full_raw ./ 1e10)[Acol_offset+1:Acol_offset+D]
    Acol0 = θ0[Acol_offset+1:Acol_offset+D]
    grad_ad = ad_grad_level .* Acol0

    directions = Tuple{String,Vector{Float64}}[]
    push!(directions, ("neg_fd_grad", -normalize(grad_fd)))
    push!(directions, ("neg_boundary_grad", -normalize(grad_boundary)))
    for o in 1:D
        e = zeros(D); e[o] = 1.0
        push!(directions, ("coord_$o", e))
    end
    for (i, v) in enumerate(random_dirs)
        push!(directions, ("random_$i", v))
    end

    w0 = focal_winners(θ0, EK_moments_focal_norm_directgp!, γ, U, D, D1)

    for (name, v) in directions
        ad_slope = dot(grad_ad, v)
        fd_slope_ref = dot(grad_fd, v)   # coordinate-gradient-based (h=0.1), for reference
        bnd_slope = dot(grad_boundary, v)
        for h in HS_PROFILE
            fd_slope_h = fixed_dual_fd_directional_derivative(θ0, obj0, x0, v, h; Acol_offset=Acol_offset)
            profile_slope, δp, δm, statusp, statusm = full_profile_directional_slope(θ0, γ, U, D1, EK_moments_focal_norm_directgp!, v, h; Acol_offset=Acol_offset)
            global n_full_solves += 2

            θp = copy(θ0); @views θp[Acol_offset+1:Acol_offset+D] .*= exp.(h .* v)
            wp = focal_winners(θp, EK_moments_focal_norm_directgp!, γ, U, D, D1)
            n_switch = count(w0 .!= wp)

            row = (γp_target=γp, direction_id=name, h=h, ad_slope=ad_slope, fixed_dual_fd_slope=fd_slope_h,
                boundary_slope=bnd_slope, full_profile_slope=profile_slope,
                fd_minus_profile=fd_slope_h - profile_slope, boundary_minus_profile=bnd_slope - profile_slope,
                winner_switch_count=n_switch, statusp=statusp, statusm=statusm)
            push!(all_rows, row)
            @printf("%18s h=%5.2f  ad=%11.4e fd=%11.4e bnd=%11.4e profile=%11.4e | fd-prof=%10.3e bnd-prof=%10.3e  nsw=%4d status=(%d,%d)\n",
                name, h, ad_slope, fd_slope_h, bnd_slope, profile_slope, row.fd_minus_profile, row.boundary_minus_profile, n_switch, statusp, statusm)
        end
    end
end

open(joinpath(@__DIR__, "part3_comparison_D$(DVAL)_W$(WVAL).csv"), "w") do io
    println(io, "gamma_target,direction_id,h,ad_slope,fixed_dual_fd_slope,boundary_slope,full_profile_slope,fd_minus_profile,boundary_minus_profile,winner_switch_count,statusp,statusm")
    for r in all_rows
        println(io, "$(r.γp_target),$(r.direction_id),$(r.h),$(r.ad_slope),$(r.fixed_dual_fd_slope),$(r.boundary_slope),$(r.full_profile_slope),$(r.fd_minus_profile),$(r.boundary_minus_profile),$(r.winner_switch_count),$(r.statusp),$(r.statusm)")
    end
end

# ---- summary stats ----
println("\n" * "="^78); println(">>> PART 3 SUMMARY"); println("="^78)
valid = [r for r in all_rows if r.statusp ∈ (0, -100, -101, -103) && r.statusm ∈ (0, -100, -101, -103)]
fd_err = [abs(r.fd_minus_profile) / max(abs(r.full_profile_slope), 1e-8) for r in valid]
bnd_err = [abs(r.boundary_minus_profile) / max(abs(r.full_profile_slope), 1e-8) for r in valid]
ad_err = [abs(r.ad_slope - r.full_profile_slope) / max(abs(r.full_profile_slope), 1e-8) for r in valid]
@printf("n valid rows = %d / %d\n", length(valid), length(all_rows))
@printf("fixed_dual_FD  vs full profile: median relerr=%.4f  mean=%.4f  max=%.4f\n", median(fd_err), mean(fd_err), maximum(fd_err))
@printf("boundary       vs full profile: median relerr=%.4f  mean=%.4f  max=%.4f\n", median(bnd_err), mean(bnd_err), maximum(bnd_err))
@printf("AD             vs full profile: median relerr=%.4f  mean=%.4f  max=%.4f\n", median(ad_err), mean(ad_err), maximum(ad_err))
@printf("\ntotal full-profile KNITRO solves: %d\n", n_full_solves)
@printf("total wall time: %.1f s\n", time() - t_start)
println("\nPART 3 DONE")
