# ============================================================================
# Prop 1a exercise: holding A_od FIXED at its prestep/data-consistent value
# (A_od = A_od*, i.e. never varied), trace the schedule (gamma'_focal, delta*)
# where delta* is the MINIMUM divergence (the CC inner solve's own value)
# needed to support that gamma'_focal. No outer loop (nothing is optimized
# over A_od), no destination inversion (other countries' shares are never
# touched), no gravity moment (pure focal-only D+1-moment problem). At each
# grid point theta is FULLY specified -- only theta[3] (gamma'_focal, direct)
# varies; theta[1] (mu), theta[2] (sigma), theta[4:end] (A[.,focal]) are held
# at their theta_r0 values throughout, exactly as run_profiled_production.jl
# constructs theta_r0 (mu_hat, sigma=sigmaHat, A_od-in-the-gammaf=1-gauge).
#
# recover_lfd()/divergence_of() below are REUSED VERBATIM from
# sequential_gravity/verify_batch_solutions.jl (this session's own,
# already-validated cold-cross-verification script) -- not reimplemented.
#
#   FAKEDATA=3 DVAL=20 julia --project=. sequential_gravity/delta_star_schedule.jl
# ============================================================================
include(joinpath(@__DIR__, "focal_moments.jl"))
include(joinpath(@__DIR__, "focal_moments_directgp.jl"))
include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2

include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "moments", "include_moments.jl"))
include(joinpath(@__DIR__, "..", "cc_algo", "include_cc_algo.jl"))
include(joinpath(@__DIR__, "..", "lfd", "include_lfd.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity

const DVAL = parse(Int, get(ENV, "DVAL", "20"))
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "3"))
const GAMMAP_STEP = parse(Float64, get(ENV, "GAMMAP_STEP", "0.001"))

params = (server=1,user=2,fakeData=FAKEDATA,DFake=DVAL,seedFakeData=889,counterType=1,counterExplicit=0,
    θHat=0,σHat=2.5,baseIndex=2,W=WVAL,seedU=888,importanceSampling=0,importanceSamplingFactor=2,
    stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,
    localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,
    NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,
    momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,
    PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,
    δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=WVAL,
    theta_init=0,runLFD=1,runLFDCounterFactual=1)

so = master_setup(params)
useParams = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(so.data, so.counters, useParams)
prep = master_prepare_cc(so.data, so.counters, prestep_output, useParams)
D = so.D; γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat

# theta_r0 construction: BYTE-IDENTICAL to sequential_gravity/run_profiled_production.jl's own
# (mu_hat, sigma, gamma'_focal(direct,adjusted), A[.,focal] in the gamma_focal===1 gauge).
θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
local θr0
let γf0 = θr0_orig[3], μ0 = θr0_orig[1]
    global θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
end

KBOUNDS = theoretical_kappa_bounds(γ, σ)
@printf("theta_r0 = %s\n", θr0)
@printf("KBOUNDS: kappa_min=%.6f kappa_max=%.6f gp_lo=%.6f gp_hi=%.6f\n",
        KBOUNDS.κ_min, KBOUNDS.κ_max, KBOUNDS.γp_lo, KBOUNDS.γp_hi)

# Direct CC-inner-solve call: no LFD/p reconstruction, no separate divergence computation --
# the inner solve's own KNITRO objective value AT THE OPTIMAL (zeta,lambda) already IS the
# min-divergence value by strong duality (that's the whole point of the CC dual: it shows
# existence of a distribution attaining this gamma'_focal at this divergence, without ever
# needing to construct it).
#
# Two grids, BOTH anchored at "the Frechet value" gamma'_focal* = theta_r0[3] (the prestep
# point estimate -- NOT the gamma_focal===1 gauge ceiling of 1.0, which turns out to be a
# genuinely degenerate/unbounded point for the CC inner dual, confirmed reproducible even at
# gamma'=0.999999 -- a known pathology of EL-type divergence duals exactly at a
# perfectly-data-consistent point, not a warm-start artifact). Solve once at gamma'_focal*
# (delta* there should be ~0), then warm-start outward in BOTH directions from that same
# solved (zeta,lambda): grid 1 decreases toward gp_lo, grid 2 increases toward gp_hi=1.0.
const D1 = D + 1
mkobj() = PsiObjectiveBundleDelta(γ = γ, (moments!) = EK_moments_focal_norm_directgp!, moments_jacobian! = error,
    d = D1, outer_constr_index = D1 + 1, inequality_index = Int64[], complement_index = [0 0],
    l = length(θr0), U = U, N = params.Jac_W, lower_limit = -5000, use_cached_x = true,
    outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")

const GP_STAR = θr0[3]
@printf("gamma'_focal* (Frechet/prestep value) = %.6f\n", GP_STAR)

function solve_one(obj, γp)
    θ = copy(θr0); θ[3] = γp
    val, x, nStatus = inner_loop(obj, θ)
    κ = 1 - γp^(σ / (σ - 1))
    ok = nStatus ∈ (0, -100, -101, -103)
    return (γp = γp, κ = κ, δstar = ok ? val : NaN, nStatus = nStatus, ok = ok)
end

function run_grid(obj, γp_start, step, γp_bound; label)
    @printf("\n--- %s (from %.6f toward %.6f, step %.4g) ---\n", label, γp_start, γp_bound, step)
    @printf("%10s %12s %14s %8s\n", "gamma'_foc", "kappa", "delta*", "nStatus")
    rows = NamedTuple[]
    γp = γp_start
    while (step < 0 ? γp >= γp_bound - 1e-9 : γp <= γp_bound + 1e-9)
        r = solve_one(obj, γp)
        @printf("%10.6f %12.6f %14.6g %8d%s\n", r.γp, r.κ, r.δstar, r.nStatus, r.ok ? "" : "   <-- FAILED, stopping")
        push!(rows, r)
        flush(stdout)
        r.ok || break
        γp += step
    end
    return rows
end

const GAMMAP_STEP_ABS = abs(GAMMAP_STEP)

# Anchor solve at the Frechet value, on its own fresh obj so both grids warm-start from the
# identical anchor state.
anchor_obj = mkobj()
anchor = solve_one(anchor_obj, GP_STAR)
@printf("\nANCHOR: gamma'_focal*=%.6f kappa=%.6f delta*=%.6g nStatus=%d ok=%s\n",
        anchor.γp, anchor.κ, anchor.δstar, anchor.nStatus, anchor.ok)
anchor.ok || error("Anchor solve at the Frechet value failed (nStatus=$(anchor.nStatus)) -- cannot proceed.")

# Decreasing grid: fresh obj seeded with the anchor's solved (zeta,lambda) as its starting cache.
dec_obj = mkobj(); dec_obj.x .= anchor_obj.x
dec_rows = run_grid(dec_obj, GP_STAR - GAMMAP_STEP_ABS, -GAMMAP_STEP_ABS, KBOUNDS.γp_lo; label = "DECREASING toward gp_lo")

# Increasing grid: separate fresh obj, also seeded from the anchor.
inc_obj = mkobj(); inc_obj.x .= anchor_obj.x
inc_rows = run_grid(inc_obj, GP_STAR + GAMMAP_STEP_ABS, GAMMAP_STEP_ABS, KBOUNDS.γp_hi; label = "INCREASING toward gp_hi")

results = vcat(reverse(dec_rows), [anchor], inc_rows)

const OUT_PATH = get(ENV, "SCHEDULE_OUT", joinpath(@__DIR__, "delta_star_schedule_out.jld2"))
JLD2.save(OUT_PATH, Dict(
    "results" => results, "anchor" => anchor, "theta_r0" => θr0, "KBOUNDS" => KBOUNDS, "sigma" => σ, "D" => D,
    "gammap_step" => GAMMAP_STEP_ABS, "WVAL" => WVAL))
println("\nSCHEDULE DONE, saved to $OUT_PATH")
