# Anchored vs orthonormal common-marginals contrasts: economic-equivalence comparison.
# D=4 fake data, per the ask ("so it doesn't take too long"). Built on the same simple harness as
# run_focal_bounds_common_marginals.jl (no gravity moment, no sequential destination inversion).
#
# Part A: FIXED outer theta (theta_r0, the F*-consistent starting point) -- solve the min-
#         divergence (delta*) inner program under both contrast modes and compare: delta*,
#         recovered LFD, divergence of the recovered LFD, and moment residuals (both raw and
#         mapped back to interpretable anchored per-country-per-threshold form). This is the most
#         important test: if the two coordinate systems disagree here, that's an implementation
#         bug, full-outer-solve differences would not be meaningful to interpret.
# Part B: full outer solve (upper bound) under both contrast modes at L=50 (paper-scale, matching
#         the already-validated D=20/paper-scale run) -- compare final kappa, outer theta, status,
#         wall time, iteration/eval counts.
#
#   julia --project=. sequential_gravity/compare_contrasts.jl
# (needs KNITRO env; see SETUP_AND_FINDINGS.md)

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, Printf

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
const CM_L = parse(Int, get(ENV, "CM_L", "50"))
const CM_REF = parse(Int, get(ENV, "CM_REF", "1"))
const CM_EQ36 = lowercase(get(ENV, "CM_EQ36", "true")) in ("1", "true", "yes")
const OUTER_OPT_FILE = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "..", "csw_outer_loop_settings_cm.opt"))
const INNER_OPT_FILE = joinpath(@__DIR__, "..", "ek_inner_loop_options.opt")
const DELTA_OPT_FILE = joinpath(@__DIR__, "..", "ek_outer_loop_options.opt")  # inner_loop_opt for the Delta (delta*) bundle

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
W = params.W

CM_anch, _, cm_origins = precalc_common_marginals_cdf(U, CM_REF, CM_L;
    include_truncated_moment = CM_EQ36, μHat = γ0.μHat, σHat = σ, contrasts = :anchored)
CM_orth, _, _          = precalc_common_marginals_cdf(U, CM_REF, CM_L;
    include_truncated_moment = CM_EQ36, μHat = γ0.μHat, σHat = σ, contrasts = :orthonormal)
nO = length(cm_origins)
γ_anch = (; γ0..., CM_Moments = CM_anch)
γ_orth = (; γ0..., CM_Moments = CM_orth)

d1 = n_focal_moments(D) + n_cm_moments(D, CM_L; include_truncated_moment = CM_EQ36)
oci1 = d1 + 1

@printf("\n=== anchored vs orthonormal contrasts: D=%d W=%d delta=%g L=%d eq36=%s (d=%d) ===\n\n",
        D, W, δ, CM_L, CM_EQ36, d1)

function focal_bounds(θr, σ)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2];  hi[2] = θr[2]
    lo[1] = 0.001;  hi[1] = 1 / (σ - 1) - 0.001
    lo[5] = θr[5];  hi[5] = θr[5]
    for i in 1:length(θr)
        if θr[i] < 0; hi[i] = -1e4 * θr[i]; lo[i] = 1e4 * θr[i]; end
    end
    return lo, hi
end
θ_lo, θ_hi = focal_bounds(θr, σ)

# ================================================================================================
# PART A: fixed-outer-theta comparison (theta = theta_r, the F*-consistent point)
# ================================================================================================

"Hybrid (KL for m<=e, chi^2 for m>e) divergence of a reweighting p (sum=1) against the uniform
1/W baseline -- identical formula to run_profiled_production.jl's own divergence_of."
function divergence_of(p::AbstractVector, W::Int)
    e = exp(1); acc = 0.0
    @inbounds for s in eachindex(p)
        m = p[s] * W
        if !(m > 0) || !isfinite(m)
            return Inf
        elseif m <= e
            acc += m * log(m) - m + 1
        else
            acc += m^2 / (2e) - e / 2 + 1
        end
    end
    return acc / W
end

"Min-divergence (delta*) inner solve at a FIXED theta, mirroring run_profiled_production.jl's
recover_lfd -- INCLUDING the delta*-non-convergence bug fix (explicit nStatus check, not just
all(isfinite,x)). Returns (p, delta_star, nStatus, ok)."
function solve_delta_star(θ, moments_fn, d, γ, U)
    W = size(U, 1)
    oci = d + 1
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = W, lower_limit = -5000,
        outer_loop_opt = DELTA_OPT_FILE, inner_loop_opt = INNER_OPT_FILE)
    val, x, nStatus = inner_loop(obj, θ)
    ok = nStatus ∈ (0, -100, -101, -103) && all(isfinite, x)
    ok || return fill(1.0 / W, W), NaN, nStatus, false
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), val, nStatus, false
    return LFD ./ s, val, nStatus, true
end

println(">>> PART A: fixed-theta (theta_r0) minimum-divergence comparison"); flush(stdout)
p_anch, δstar_anch, st_anch, ok_anch = solve_delta_star(θr, EK_moments_focal_cm!, d1, γ_anch, U)
p_orth, δstar_orth, st_orth, ok_orth = solve_delta_star(θr, EK_moments_focal_cm!, d1, γ_orth, U)

div_anch = divergence_of(p_anch, W)
div_orth = divergence_of(p_orth, W)
Δp = maximum(abs.(p_anch .- p_orth))
Δp_rel = Δp / max(maximum(abs.(p_anch)), 1e-300)

# moment residuals E_p[moments], mapped back to interpretable anchored (country vs refIndex1)
# coordinates via cm_block_to_anchored_residuals regardless of which mode produced them
G_anch = zeros(W, d1); K_anch = zeros(W); EK_moments_focal_cm!(K_anch, G_anch, θr, U, (γ = γ_anch,))
G_orth = zeros(W, d1); K_orth = zeros(W); EK_moments_focal_cm!(K_orth, G_orth, θr, U, (γ = γ_orth,))
resid_anch_raw = vec(sum(G_anch .* p_anch, dims = 1))
resid_orth_raw = vec(sum(G_orth .* p_orth, dims = 1))
nfocal = n_focal_moments(D)
cm_resid_anch = @view resid_anch_raw[nfocal+1:end]
cm_resid_orth = @view resid_orth_raw[nfocal+1:end]
anch_eq35 = cm_block_to_anchored_residuals(cm_resid_anch, D, CM_L, nO; include_truncated_moment = CM_EQ36, contrasts = :anchored)
orth_eq35 = cm_block_to_anchored_residuals(cm_resid_orth, D, CM_L, nO; include_truncated_moment = CM_EQ36, contrasts = :orthonormal)
if CM_EQ36
    anch_eq35, anch_eq36 = anch_eq35
    orth_eq35, orth_eq36 = orth_eq35
end

@printf("  anchored:     delta*=%.6e  status=%d  ok=%s  divergence(recovered p)=%.6e\n", δstar_anch, st_anch, ok_anch, div_anch)
@printf("  orthonormal:  delta*=%.6e  status=%d  ok=%s  divergence(recovered p)=%.6e\n", δstar_orth, st_orth, ok_orth, div_orth)
@printf("  |delta*_anch - delta*_orth| = %.3e\n", abs(δstar_anch - δstar_orth))
@printf("  max |p_anch - p_orth| = %.3e  (relative to max|p|: %.3e)\n", Δp, Δp_rel)
@printf("  max |anchored-coord eq35 residual|: anch-solve=%.3e  orth-solve=%.3e  cross-diff=%.3e\n",
        maximum(abs, anch_eq35), maximum(abs, orth_eq35), maximum(abs.(anch_eq35 .- orth_eq35)))
if CM_EQ36
    @printf("  max |anchored-coord eq36 residual|: anch-solve=%.3e  orth-solve=%.3e  cross-diff=%.3e\n",
            maximum(abs, anch_eq36), maximum(abs, orth_eq36), maximum(abs.(anch_eq36 .- orth_eq36)))
end
@printf("  max |economic (focal) moment residual|: anch-solve=%.3e  orth-solve=%.3e\n",
        maximum(abs, @view resid_anch_raw[1:nfocal]), maximum(abs, @view resid_orth_raw[1:nfocal]))

# ================================================================================================
# PART B: full outer solve comparison (upper bound), both contrast modes
# ================================================================================================

make_obj(γuse, moments_fn, d, oci, find_smallest) = PsiObjectiveBundleImplicit(
    δ = δ, find_smallest = find_smallest, γ = γuse,
    (moments!) = moments_fn, moments_jacobian! = error,
    d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
    l = l, U = U, N = params.Jac_W, lower_limit = -50, use_cached_x = true,
    outer_loop_opt = OUTER_OPT_FILE,
    inner_loop_opt = INNER_OPT_FILE,
)

function run_bound(label, γuse)
    println("\n>>> PART B: full outer solve ($label, upper bound)"); flush(stdout)
    t0 = time()
    κ, θstar, st, _ = outer_loop(make_obj(γuse, EK_moments_focal_cm!, d1, oci1, false), θ_lo, θ_hi, copy(θr))
    wall = time() - t0
    @printf("  [%s] kappa_upper=%.6f  status=%d  wall=%.1fs\n", label, κ, st, wall)
    return (κ = κ, θstar = θstar, status = st, wall = wall)
end

r_anch = run_bound("ANCHORED", γ_anch)
r_orth = run_bound("ORTHONORMAL", γ_orth)

@printf("\n=== SUMMARY (D=%d, delta=%g, L=%d, eq36=%s) ===\n", D, δ, CM_L, CM_EQ36)
@printf("Part A (fixed theta_r0):\n")
@printf("  delta*         : anchored=%.6e   orthonormal=%.6e   diff=%.3e\n", δstar_anch, δstar_orth, abs(δstar_anch-δstar_orth))
@printf("  status         : anchored=%d   orthonormal=%d\n", st_anch, st_orth)
@printf("  recovered p    : max|diff|=%.3e (rel %.3e)\n", Δp, Δp_rel)
@printf("Part B (full outer solve, upper bound):\n")
@printf("  kappa_upper    : anchored=%.6f   orthonormal=%.6f   diff=%.3e\n", r_anch.κ, r_orth.κ, abs(r_anch.κ - r_orth.κ))
@printf("  status         : anchored=%d   orthonormal=%d\n", r_anch.status, r_orth.status)
@printf("  wall time (s)  : anchored=%.1f   orthonormal=%.1f\n", r_anch.wall, r_orth.wall)
@printf("  max|theta diff|: %.3e\n", maximum(abs.(r_anch.θstar .- r_orth.θstar)))
