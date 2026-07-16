# ============================================================================
# Part 2.4 driver: verify the conditional winner-boundary Jacobian formula
# against the analytical Frechet raw-moment Jacobian BEFORE applying it to
# the full dual integrand (Part 2's mandatory first gate).
#
#   DVAL=4 WVAL=8000 julia --project=. sequential_gravity/derivative_diagnostics/run_part2_verify_raw_jacobian.jl
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

θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
local θr0
let γf0 = θr0_orig[3], μ0 = θr0_orig[1]
    global θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
end
μ0 = θr0[1]
β = (1 / μ0) / (σ - 1)
gamma_norm = gamma(μ0 * (1 - σ) + 1)
@printf("D=%d W=%d  mu=%.6f sigma=%.6f  beta=%.6f  gamma_norm=%.6f\n", D, W, μ0, σ, β, gamma_norm)

println("\n" * "="^78); println(">>> PART 2.4: verify boundary+intensive raw-moment Jacobian vs analytical Frechet formula"); println("="^78)

r = verify_raw_moment_jacobian(θr0, γ, U, D, β; gamma_norm=gamma_norm)

function offdiag_diag_errors(J_ref, J_test, D)
    diag_err = [abs(J_test[o, o] - J_ref[o, o]) / max(abs(J_ref[o, o]), 1e-12) for o in 1:D]
    off_err = Float64[]
    for o in 1:D, j in 1:D
        o == j && continue
        push!(off_err, abs(J_test[o, j] - J_ref[o, j]) / max(abs(J_ref[o, j]), 1e-12))
    end
    return (diag_max=maximum(diag_err), diag_mean=mean(diag_err), off_max=maximum(off_err), off_mean=mean(off_err))
end

e_intensive = offdiag_diag_errors(r.J_ref, r.J_intensive, D)
e_total = offdiag_diag_errors(r.J_ref, r.J_total, D)
@printf("\nINTENSIVE-only vs reference:      diag maxrel=%.3e meanrel=%.3e | offdiag maxrel=%.3e meanrel=%.3e\n",
    e_intensive.diag_max, e_intensive.diag_mean, e_intensive.off_max, e_intensive.off_mean)
@printf("INTENSIVE+BOUNDARY vs reference:  diag maxrel=%.3e meanrel=%.3e | offdiag maxrel=%.3e meanrel=%.3e\n",
    e_total.diag_max, e_total.diag_mean, e_total.off_max, e_total.off_mean)

println("\nJ_ref:"); display(r.J_ref); println()
println("J_total (intensive+boundary):"); display(r.J_total); println()

# ---- convergence in W: rerun the boundary MC piece at several draw counts using the SAME U (subsample) ----
println("\n--- boundary-term MC convergence vs draw count (using U[1:Wsub,:] subsamples of the SAME draws) ---")
for Wsub in unique(min.([1000, 4000, 8000, 32000, min(W, 128000)], W))
    Usub = U[1:Wsub, :]
    rsub = verify_raw_moment_jacobian(θr0, γ, Usub, D, β; gamma_norm=gamma_norm)
    esub = offdiag_diag_errors(r.J_ref, rsub.J_total, D)
    @printf("Wsub=%7d   diag maxrel=%.3e   offdiag maxrel=%.3e\n", Wsub, esub.diag_max, esub.off_max)
end

pass = e_total.diag_max < 0.05 && e_total.off_max < 0.05
println("\n>>> PART 2.4 GATE ", pass ? "PASSES" : "FAILS", " (intensive+boundary within 5% of the analytical Frechet Jacobian, both diag and offdiag)")
pass || println(">>> DO NOT proceed to production use of the boundary formula until this gate passes.")

println("\nPART 2.4 DONE")
