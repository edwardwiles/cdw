# Phase-1 validation on the REAL pipeline objects (no outer solve, no KNITRO solve).
# Runs setup + prestep + draws exactly as master.jl would, then exercises the profiled-gravity
# core on the actual U/Uσ draws, wages, tariffs, and observed trade shares. Confirms the object
# plumbing (build_log_x from the pipeline's Uσ; λData orientation; wages/τ in the residual) and
# re-runs the mandatory directional-derivative check on real data.
#
#   julia --project=. sequential_gravity/validate_on_pipeline.jl

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra

# pipeline pieces needed to build draws (NOT cc_algo — avoids the KNITRO solve)
include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))

include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
using Printf

# ---- same params as master.jl (the fast 4-country Frechet example) --------------------------
params = (
    server=1, user=2, fakeData=1, DFake=4, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=8000, seedU=888,
    importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0, IndMomentOrder=5,
    θConstant=0, gravMoment=1, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1,
    refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=0, calc_δ_star_initial=1, Jac_W=8000,
    theta_init=0, runLFD=1, runLFDCounterFactual=1,
)

setup_output = master_setup(params)
@unpack data, counters = setup_output
D = setup_output.D
useParams = (; params..., D = D)
prestep_output = master_prestep(data, counters, useParams)

# draws, exactly as master_prepare_cc does
Random.seed!(params.seedU)
SamplingWeight = ones(params.W)
U = drawU(SamplingWeight, useParams)
Ū, Uσ = createUDerivatives!(U, prestep_output, useParams)

μ = prestep_output.μHat
σ = params.σHat
wHat = prestep_output.wHat
τ = data.τData
λData = data.λData
baseIndex = params.baseIndex
W = params.W
ref = 1
RHO = 3e-3

@printf("\n=== real-pipeline validation (D=%d, W=%d, σ=%.2f, μ=%.4f, baseIndex=%d) ===\n", D, W, σ, μ, baseIndex)

# orientation check: shares sum to 1 over origins for each destination
col_sums = vec(sum(λData, dims = 1))
@printf("λData column (over-origin) sums: %s  (should be ~1)\n", string(round.(col_sums, digits=6)))

# log_x from the pipeline's Uσ (= U.^(1-σ) with θConstant=0);  log_x = μ log Uσ
log_x = build_log_x(Uσ, μ)
p_star = fill(1 / W, W)          # F* is uniform over the Exp(1) draws

# ---- invert every destination's observed share column under F* -------------------------------
u_mat = zeros(D, D)
maxerr = 0.0; ok = true
for d in 1:D
    inv = invert_destination(log_x, p_star, λData[:, d]; ref = ref, ρ = RHO, tol = 1e-10, maxit = 300)
    u_mat[:, d] .= inv.u_full
    global maxerr = max(maxerr, inv.max_abs_share_error)
    global ok &= inv.converged
    @printf("  dest %d: converged=%s  share_err=%.2e  iters=%d\n", d, inv.converged, inv.max_abs_share_error, inv.iterations)
end
@printf("all inversions converged=%s  worst share err=%.2e\n", ok, maxerr)

# cross-check: recovered u vs the calibrated u = (σ-1)(log A_od − log w_o − log τ_od), A=AHat.
# (equal up to MC error and the per-column gauge; report the two-way-demeaned gap, gauge-free)
AHat = (((wHat .* τ) ./ (wHat[1] .* τ[1, :]')) .^ (1 / μ)) .* (λData ./ λData[1, :]')
u_calib = (σ - 1) .* (log.(AHat) .- log.(wHat) .- log.(τ))
gap = maximum(abs.(two_way_demean(u_mat) .- two_way_demean(u_calib)))
@printf("‖demean(u_recovered) − demean(u_calibrated)‖∞ = %.2e  (MC-limited)\n", gap)

# ---- gravity residual on real objects --------------------------------------------------------
gr = gravity_residual(u_mat, log.(τ), log.(wHat), σ)
@printf("gravity residual on real data: R_sum=%.6e  R_beta=%.6e  (S_Q=%.4f)\n", gr.R_sum, gr.R_beta, gr.S_Q)

# ---- influence function + directional-derivative check on real objects -----------------------
focal = baseIndex
omitted = [d for d in 1:D if d != focal]
function invert_all(p; warm = nothing)
    um = copy(u_mat)
    for d in omitted
        ui = warm === nothing ? nothing : warm[:, d]
        um[:, d] .= invert_destination(log_x, p, λData[:, d]; ref = ref, ρ = RHO, tol = 1e-12, maxit = 400, u_init = ui).u_full
    end
    return um
end
umat0 = invert_all(p_star)
infl = influence_function(log_x, p_star, umat0, λData, omitted, log.(τ), log.(wHat), σ; ref = ref, ρ = RHO, scale = :R_beta)
@printf("R_beta(F*)=%.6f  E_F[ψ_R]=%.2e\n", infl.R_beta, infl.Eψ)
for pd in infl.per_dest
    @printf("  dest %d: M_d=%.3e cond(H)=%.2e smin=%.2e adj_resid=%.1e share_err=%.1e\n",
            pd.d, pd.M_d, pd.condH, pd.smin_H, pd.adjoint_resid, pd.max_share_err)
end

Random.seed!(7)
worst_rel = 0.0
for dir in 1:3
    h = randn(W); h .-= dot(p_star, h); h ./= maximum(abs.(h))
    predicted = dot(p_star, h .* infl.ψ_R)
    best_rel = Inf
    for t in (1e-3, 3e-4, 1e-4), sgn in (1.0, -1.0)
        tt = sgn * t
        pt = p_star .* (1 .+ tt .* h)
        any(pt .<= 0) && continue
        pt ./= sum(pt)
        grt = gravity_residual(invert_all(pt; warm = umat0), log.(τ), log.(wHat), σ)
        rel = abs((grt.R_beta - infl.R_beta) / tt - predicted) / (abs(predicted) + 1e-12)
        best_rel = min(best_rel, rel)
    end
    global worst_rel = max(worst_rel, best_rel)
    @printf("  dir%d: pred=%.5e  best rel_err=%.2e\n", dir, predicted, best_rel)
end
@printf("\nDIRECTIONAL-DERIVATIVE on real data: worst rel_err=%.2e  =>  %s\n",
        worst_rel, worst_rel < 5e-3 ? "PASS" : "FAIL")
