# Full-A_od-in-outer-loop bound runs, with optional freezes, for comparison against the profiled
# method. Reuses the EXACT setup/prestep/prepare_cc/EK_moments!/outer_loop machinery master.jl and
# ccOuter.jl use (same PsiObjectiveBundleImplicit path, same "csw_outer_loop_settings_cluster.opt"),
# replicating ccOuter.jl's bound-setting logic exactly, with two optional extra freezes read from
# ENV:
#   FREEZE_MU=1            : pin θ_lower[1]=θ_upper[1]=θ_initial[1] (μ never moves in the search).
#                             Safe/exact: at μ=μHat the Δ^A(μ) reparameterization factor in
#                             moments!.jl collapses to 1 identically (cHat's calibration is built
#                             from the SAME μHat), so Aod=Aod_θ throughout — pinning via bounds
#                             (not θConstant, which also changes caching) is behaviorally identical.
#   FREEZE_NONFOCAL_A=1    : additionally pin EVERY A[o,d] for d≠focal (all origins, not just the
#                             usual A[1,d]=1 normalization row) at θ_initial (=1, the calibrated
#                             Frechet baseline) — only the focal column A[·,focal] stays free. Same
#                             outer dimensionality as the profiled method, but WITHOUT profiling:
#                             the non-focal columns are frozen, not inverted to match trade shares
#                             under the evolving F. Isolates whether profiling's inversion step (not
#                             just outer-dimension reduction) is what matters.
#   julia --project=. sequential_gravity/run_fullA_variant.jl   (needs KNITRO env)

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2

include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "moments", "include_moments.jl"))
include(joinpath(@__DIR__, "..", "cc_algo", "include_cc_algo.jl"))
include(joinpath(@__DIR__, "..", "lfd", "include_lfd.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))
using .CounterfactualSensitivity
using Printf

FREEZE_MU = get(ENV, "FREEZE_MU", "0") == "1"
FREEZE_NONFOCAL_A = get(ENV, "FREEZE_NONFOCAL_A", "0") == "1"

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
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(useParams)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)

@unpack θ_initial, θ_initial_low, θ_initial_up, U, γ, numMoments, δ_grid, outer_constr_index,
        inequality_index, nTotalMoments, complement_index = prep
focal = params.baseIndex
JacW = params.Jac_W

# ---- replicate ccOuter.jl's bound-setting exactly (counterType=1, OuterScaling=1, independenceMoment=0) ----
θ_lower = (θ_initial .* 0.0001)[:]
θ_upper = (θ_initial .* 10000)[:]
θ_lower[2] = θ_initial[2]; θ_upper[2] = θ_initial[2]                    # σ fixed
θ_upper[1] = 1 / (θ_initial[2] - 1) - 0.001; θ_lower[1] = 0.001         # μ range

if FREEZE_MU
    θ_lower[1] = θ_initial[1]; θ_upper[1] = θ_initial[1]
else
    θ_upper[1] = min(θ_upper[1], 1 / (θ_initial[2] - 1))
end

Aod_offset = 3 + D    # counterType=1 → no counterType_θ_offset; OuterScaling=1 → +3+D
for i in 1:D          # A[1,d]=1 normalization, every destination column d=i
    idx = Aod_offset + 1 + D * (i - 1)
    θ_upper[idx] = θ_initial[idx]; θ_lower[idx] = θ_initial[idx]
end
if FREEZE_NONFOCAL_A
    for d in 1:D, o in 1:D
        d == focal && continue
        idx = Aod_offset + (d - 1) * D + o
        θ_upper[idx] = θ_initial[idx]; θ_lower[idx] = θ_initial[idx]
    end
end

for i in 1:length(θ_initial)
    if θ_initial[i] < 0
        θ_upper[i] = -10000 * θ_initial[i]; θ_lower[i] = 10000 * θ_initial[i]
    end
end

n_free = count(i -> θ_lower[i] != θ_upper[i], eachindex(θ_lower))
@printf("\n=== full-A variant: FREEZE_MU=%s FREEZE_NONFOCAL_A=%s (D=%d, W=%d), free outer params=%d/%d ===\n",
        FREEZE_MU, FREEZE_NONFOCAL_A, D, params.W, n_free, length(θ_initial))

function build_obj(find_smallest, δval)
    PsiObjectiveBundleImplicit(
        δ = δval, find_smallest = find_smallest, γ = γ, (moments!) = EK_moments!,
        moments_jacobian! = error, d = nTotalMoments, outer_constr_index = outer_constr_index,
        inequality_index = inequality_index, complement_index = complement_index,
        l = size(θ_initial, 1), U = U, N = JacW, lower_limit = -50, use_cached_x = true,
        outer_loop_opt = "csw_outer_loop_settings_cluster.opt", inner_loop_opt = "ek_inner_loop_options.opt",
    )
end

function run_at_delta(δval::Real)
    @printf("\n=== full-A variant at δ=%g ===\n", δval)
    results = Dict{Symbol,Any}()
    for (name, fs, θinit) in ((:upper, false, θ_initial_up), (:lower, true, θ_initial_low))
        @printf("\n----- %s bound, δ=%g -----\n", name, δval); flush(stdout)
        t0 = time()
        obj = build_obj(fs, δval)
        κ, θ1, nStatus, _ = outer_loop(obj, θ_lower, θ_upper, θinit)
        @printf("  κ_%s = %.6f  (status %d)  wall %.1fs\n", name, κ, nStatus, time() - t0)
        results[name] = κ
    end
    @printf("\n=== FULL-A VARIANT RESULTS (FREEZE_MU=%s FREEZE_NONFOCAL_A=%s, δ=%g) ===\n",
            FREEZE_MU, FREEZE_NONFOCAL_A, δval)
    @printf("  κ_lower = %.6f\n", results[:lower])
    @printf("  κ_upper = %.6f\n", results[:upper])
    return results
end

DELTA_GRID = let s = get(ENV, "DELTA_GRID", "")
    isempty(s) ? [δ_grid[1]] : parse.(Float64, split(s, ","))
end

all_results = Dict{Float64,Any}()
for δval in DELTA_GRID
    all_results[δval] = run_at_delta(δval)
end

@printf("\n\n=== SUMMARY ACROSS δ (FREEZE_MU=%s FREEZE_NONFOCAL_A=%s) ===\n", FREEZE_MU, FREEZE_NONFOCAL_A)
@printf("%8s | %12s | %12s\n", "δ", "κ_lower", "κ_upper")
for δval in DELTA_GRID
    r = all_results[δval]
    @printf("%8.4g | %12.6f | %12.6f\n", δval, r[:lower], r[:upper])
end
