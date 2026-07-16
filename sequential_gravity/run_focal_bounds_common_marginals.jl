# D=4 validation: pure-CDF common-marginals restriction (CDW eq. 35) on top of the reduced
# focal-only CC solve. Built directly on run_focal_bounds.jl's harness (no gravity moment, no
# sequential destination inversion, plain outer_loop) -- the simplest available base, matching
# the ask to validate the restriction on a simple D=4 case before wiring it into the full
# sequential-gravity production driver.
#
# Runs the UNRESTRICTED focal-only bounds and the COMMON-MARGINALS-RESTRICTED bounds back-to-back
# at IDENTICAL data/seed/delta, for a direct comparison -- mirroring CDW Figure 4 (blue =
# unrestricted vs orange = common marginals): the restriction should leave the lower bound
# essentially unaffected (F* already has iid, hence common, marginals) and shrink the upper bound
# somewhat (CDW report ~11% at delta=2 for the full D=20 real-data model; no a-priori prediction
# for this D=4 synthetic case beyond "some non-negative shrinkage").
#
#   julia --project=. sequential_gravity/run_focal_bounds_common_marginals.jl
#   CM_L=5 CM_REF=1 DVAL=4 DELTA=1.0 julia --project=. sequential_gravity/run_focal_bounds_common_marginals.jl
# (needs KNITRO env; see SETUP_AND_FINDINGS.md)

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, Printf

# Shared-machine courtesy cap -- see run_profiled_production.jl's identical line for why.
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_NUM_THREADS", "19")))

include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "moments", "include_moments.jl"))
include(joinpath(@__DIR__, "..", "cc_algo", "include_cc_algo.jl"))
include(joinpath(@__DIR__, "..", "lfd", "include_lfd.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))
using .CounterfactualSensitivity
include(joinpath(@__DIR__, "focal_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "4"))
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))
const DELTA = parse(Float64, get(ENV, "DELTA", "1.0"))
const CM_L = parse(Int, get(ENV, "CM_L", "5"))
const CM_REF = parse(Int, get(ENV, "CM_REF", "1"))
const CM_EQ36 = lowercase(get(ENV, "CM_EQ36", "false")) in ("1", "true", "yes")

params = (
    server=1, user=2, fakeData=1, DFake=DVAL, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=WVAL, seedU=888,
    importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0, IndMomentOrder=5,
    θConstant=0, gravMoment=1, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=DELTA,
    refIndex1=CM_REF, OuterLoop=1, UoModel=1, use_Jacobian=0, calc_δ_star_initial=1, Jac_W=WVAL,
    theta_init=0, runLFD=1, runLFDCounterFactual=1,
)

setup_output = master_setup(params)
@unpack data, counters = setup_output
D = setup_output.D
@assert D == DVAL
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)

γ0 = prep.γ; U = prep.U; focal = params.baseIndex; σ = params.σHat
θr = build_focal_theta(prep.θ_initial, D, focal)
l = length(θr)
δ = params.δ_ref

if CM_REF == focal
    @warn "CM_REF ($CM_REF) coincides with the focal/counterfactual country (baseIndex=$focal); allowed, but the paper's convention keeps these roles distinct (fixed reference country for CDF-matching vs. the destination whose GT is bounded)."
end

CM_Moments, cm_thresholds, cm_origins = precalc_common_marginals_cdf(U, CM_REF, CM_L;
    include_truncated_moment = CM_EQ36, μHat = γ0.μHat, σHat = params.σHat)
nO = length(cm_origins)
γ = (; γ0..., CM_Moments = CM_Moments)

d0 = n_focal_moments(D)                  # D+1, unrestricted
d1 = d0 + n_cm_moments(D, CM_L; include_truncated_moment = CM_EQ36)   # +(D-1)*L or +2*(D-1)*L
oci0 = d0 + 1; oci1 = d1 + 1

# reduced θ bounds (mirror ccOuter): fix σ and A[1,focal]; μ∈(0,1/(σ-1)); others wide.
function focal_bounds(θr, σ)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2];  hi[2] = θr[2]                 # σ fixed
    lo[1] = 0.001;  hi[1] = 1 / (σ - 1) - 0.001    # μ
    lo[5] = θr[5];  hi[5] = θr[5]                 # A[1,focal] pinned (=1)
    for i in 1:length(θr)
        if θr[i] < 0; hi[i] = -1e4 * θr[i]; lo[i] = 1e4 * θr[i]; end
    end
    return lo, hi
end
θ_lo, θ_hi = focal_bounds(θr, σ)

# Absolute, @__DIR__-anchored paths -- NOT bare relative filenames. Something mid-script changes
# the process's working directory (a documented issue in this codebase; run_profiled_production.jl
# already works around it the same way), so a relative opt-file path resolves fine for the FIRST
# KNITRO call in a script but silently fails ("could not open file ... for input") for later ones,
# which then return degenerate 1e10/-1e10 sentinel "results" instead of erroring loudly.
const OUTER_OPT_FILE = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "..", "csw_outer_loop_settings_cluster.opt"))
const INNER_OPT_FILE = joinpath(@__DIR__, "..", "ek_inner_loop_options.opt")

make_obj(moments_fn, d, oci, find_smallest) = PsiObjectiveBundleImplicit(
    δ = δ, find_smallest = find_smallest, γ = γ,
    (moments!) = moments_fn, moments_jacobian! = error,
    d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
    l = l, U = U, N = params.Jac_W, lower_limit = -50, use_cached_x = true,
    outer_loop_opt = OUTER_OPT_FILE,
    inner_loop_opt = INNER_OPT_FILE,
)

@printf("\n=== D=%d W=%d delta=%g  common-marginals check: L=%d quantiles, refIndex1=%d, %d non-ref origins, eq36(truncated moment)=%s ===\n",
        D, params.W, δ, CM_L, CM_REF, nO, CM_EQ36)
@printf("moment counts: unrestricted d=%d, common-marginals-restricted d=%d (+%d)\n\n", d0, d1, d1 - d0)

Kchk = zeros(params.W); Gchk0 = zeros(params.W, d0)
EK_moments_focal!(Kchk, Gchk0, θr, U, (γ = γ,))
@printf("point estimate κ(F*) = %.6f\n", Kchk[1])

# Sanity check: at F* (theta_r, no reweighting), the CM moments should be CLOSE to zero (both
# origins are iid Exp(1) draws from the same baseline sampler) but not EXACTLY zero -- z_l is the
# REFERENCE origin's own empirical quantile, so its own column matches exactly by construction,
# while other origins are independent draws from the same distribution and match only up to
# O(1/sqrt(W)) finite-sample noise.
@printf("max |CM moment| at F* (should be small, ~O(1/sqrt(W))=%.4f): %.4f\n\n", 1/sqrt(params.W), maximum(abs.(CM_Moments' * ones(params.W)) ./ params.W))

println(">>> UNRESTRICTED upper bound"); flush(stdout)
κ_upper0, θ_up0, st_up0, _ = outer_loop(make_obj(EK_moments_focal!, d0, oci0, false), θ_lo, θ_hi, copy(θr))
println(">>> UNRESTRICTED lower bound"); flush(stdout)
κ_lower0, θ_low0, st_lo0, _ = outer_loop(make_obj(EK_moments_focal!, d0, oci0, true), θ_lo, θ_hi, copy(θr))

println("\n>>> COMMON-MARGINALS upper bound"); flush(stdout)
κ_upper1, θ_up1, st_up1, _ = outer_loop(make_obj(EK_moments_focal_cm!, d1, oci1, false), θ_lo, θ_hi, copy(θr))
println(">>> COMMON-MARGINALS lower bound"); flush(stdout)
κ_lower1, θ_low1, st_lo1, _ = outer_loop(make_obj(EK_moments_focal_cm!, d1, oci1, true), θ_lo, θ_hi, copy(θr))

@printf("\n=== RESULTS (D=%d, delta=%g, CM_L=%d, CM_REF=%d, eq36=%s) ===\n", D, δ, CM_L, CM_REF, CM_EQ36)
@printf("  point estimate                = %.6f\n", Kchk[1])
@printf("  UNRESTRICTED bounds:            [%.6f, %.6f]  (status lo=%d, up=%d)\n", κ_lower0, κ_upper0, st_lo0, st_up0)
@printf("  COMMON-MARGINALS bounds (L=%d): [%.6f, %.6f]  (status lo=%d, up=%d)\n", CM_L, κ_lower1, κ_upper1, st_lo1, st_up1)
if isfinite(κ_upper0) && κ_upper0 != 0
    @printf("  upper-bound shrinkage: %.1f%%\n", 100 * (1 - (κ_upper1 - κ_lower0) / max(κ_upper0 - κ_lower0, 1e-12)))
end
