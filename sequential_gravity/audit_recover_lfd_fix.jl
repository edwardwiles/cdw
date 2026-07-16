# ============================================================================
# AUDIT: re-verify a saved theta_star (from a prior run_profiled_production.jl solve, produced
# BEFORE the recover_lfd delta*-non-convergence bug fix) is still valid now that the fix is in
# place. Loads theta_star from a saved JLD2 result file and re-runs seq_gravcol AT THAT THETA
# (cold, no warm start) using the FIXED recover_lfd -- if the recomputed gravity-feasibility,
# R_mean, and divergence(recovered p) still check out (divergence <= budget, as they should have
# all along if the original run was never actually hit by the bug), the original result stands.
# If the fix changes the outcome, the original result was spurious.
#
# This file is a byte-identical PREFIX of run_profiled_production.jl (setup through seq_gravcol's
# definition and the F* point-estimate check) -- see that file for the full production driver.
# Everything below the prefix (the outer-search loop) is REPLACED by the audit logic at the
# bottom of this file.
#
#   JLD2_PATH=/path/to/seq_upper_delta1.0.jld2 CM_L=5 DVAL=20 FAKEDATA=3 WVAL=80000 DELTA=1.0 \
#     julia --project=. sequential_gravity/audit_recover_lfd_fix.jl
# ============================================================================

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, Printf

# Shared-machine courtesy cap: OpenBLAS defaults to using ALL available cores for the
# linear-algebra-heavy parts of the inner/outer solve (Hessian outer-products, ForwardDiff
# Jacobians), which is bad manners on a heavily multi-tenant box. Cap explicitly rather than
# relying on the launching shell to set OPENBLAS_NUM_THREADS. 19 matches this repo's own
# PARALLEL_INVERSION convention (D-1 destinations at D=20) per user instruction.
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_NUM_THREADS", "19")))

include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "moments", "include_moments.jl"))
include(joinpath(@__DIR__, "..", "cc_algo", "include_cc_algo.jl"))
include(joinpath(@__DIR__, "..", "lfd", "include_lfd.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "focal_moments.jl"))
include(joinpath(@__DIR__, "focal_moments_directgp.jl"))
include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
CS.include(joinpath(@__DIR__, "PsiObjectiveBundleImplicitMethodB.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "4"))
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))
# fakeData=1 (default, synthetic Frechet draws) unless overridden -- fakeData=3 is Noah's real
# D=20 dataset (setup/importData.jl), REAL_DATA_DIR-overridable, default real_data/noah_D20.
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "1"))
const OUTER_OPT_FILE = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "..", "full_aod_diag", "csw_outer_25.opt"))
const INNER_OPT_FILE = joinpath(@__DIR__, "..", "full_aod_diag", "ek_inner.opt")
# Destination-inversion share-match tolerance. 1e-8 (the long-standing default, tuned against
# synthetic data) turned out to be slightly too strict for real D=20 data: every "non-converged"
# destination there stalls at a genuine numerical fixed point (confirmed by raising maxit 150->500
# with zero change in the resulting share error) with share_err in [3.7e-8, 7.1e-8] -- an excellent
# match (matches real trade shares to 1 part in ~15-30 million), just narrowly missing 1e-8. User
# confirmed 1e-6 (or looser) is economically fine ("nobody cares about trade shares beyond 2dp").
const DEST_INV_TOL = parse(Float64, get(ENV, "DEST_INV_TOL", "1e-6"))

# Seed the first (currently gravity-blind, D+1-moment) CC solve at each theta with the linearized
# gravity moment from the PREVIOUS theta's converged (p, umat) instead of solving blind. Ported
# from full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl (evaluated there and
# REJECTED as worse on every axis at delta=10/upper -- see
# full_aod_diag/gravity_seeded_initial_solve/README.md -- kept here, env-gated and off by default,
# purely so this task's full delta-grid/both-bound comparison can be produced without a second
# ~400-line driver copy).
const GRAVITY_SEED = lowercase(get(ENV, "GRAVITY_SEED", "false")) in ("1", "true", "yes")

# Common-marginals restriction (CDW eq. 35): (D-1)*CM_L extra, theta-INDEPENDENT CDF-equality
# inner-loop moments (sequential_gravity/common_marginals_moments.jl), appended after the gravity
# column. CM_L=0 (default) is OFF and reproduces the prior unrestricted behavior exactly --
# nothing below changes shape/timing/results when CM_L=0. CM_REF is the fixed reference country
# for the CDF comparison (paper convention: distinct from, and unrelated to, baseIndex/focal).
const CM_L = parse(Int, get(ENV, "CM_L", "0"))
const CM_REF = parse(Int, get(ENV, "CM_REF", "1"))
const CM_ENABLED = CM_L > 0
# eq. 36 companion (truncated (1-σ)-moment condition alongside the pure-CDF eq. 35 match) -- see
# common_marginals_moments.jl's file-level comment for the exponent convention. Doubles the extra
# moment count (nCM below) when on.
const CM_EQ36 = lowercase(get(ENV, "CM_EQ36", "false")) in ("1", "true", "yes")

# Destination-level parallelism for the D-1 omitted-destination inversions (see
# full_aod_diag/sequential_inversion_performance/parallelism_report.md: destinations are exactly
# independent given (log_x, p); validated bit-for-bit identical to serial, ~4.2x on 9 threads).
# Opt-in: set PARALLEL_INVERSION=true AND launch julia with -t N (N >= D-1) for this to do
# anything -- with the default 1 julia thread, Threads.@threads degrades to an ordinary serial
# loop. Off by default so behavior/perf is unchanged unless explicitly requested.
const PARALLEL_INVERSION = lowercase(get(ENV, "PARALLEL_INVERSION", "false")) in ("1", "true", "yes")
if PARALLEL_INVERSION
    BLAS.set_num_threads(1)   # avoid oversubscription vs the outer Threads.@threads over destinations
    @printf("[PARALLEL_INVERSION=true] julia threads=%d, BLAS threads set to 1\n", Threads.nthreads())
    Threads.nthreads() == 1 && @warn "PARALLEL_INVERSION=true but Julia was launched with only 1 thread (-t 1); this will run serially. Relaunch with `julia -t N` (N >= D-1) to get any speedup."
end

# Additive-only timing instrumentation (zero economic/control-flow change): the existing
# r.cache.t_inner/t_grad (cc_algo/outer_eval_cache.jl) separate "inner CC solve" from "outer
# gradient" but don't isolate destination-inversion time (the piece this whole perf effort has been
# about) or influence_function/gravity_residual time as their own buckets. Threads.Atomic since
# PARALLEL_INVERSION runs the destination loop via Threads.@threads -- plain Ref addition there
# would be a data race.
const SEQ_TIMING = (
    dest_inv_s = Threads.Atomic{Float64}(0.0),
    influence_s = Threads.Atomic{Float64}(0.0),
    gravity_resid_s = Threads.Atomic{Float64}(0.0),
    n_dest_inversions = Threads.Atomic{Int}(0),
    n_dest_nonconverged = Threads.Atomic{Int}(0),
)
seq_timing_snapshot() = (dest_inv_s = SEQ_TIMING.dest_inv_s[], influence_s = SEQ_TIMING.influence_s[],
                         gravity_resid_s = SEQ_TIMING.gravity_resid_s[],
                         n_dest_inversions = SEQ_TIMING.n_dest_inversions[],
                         n_dest_nonconverged = SEQ_TIMING.n_dest_nonconverged[])
seq_timing_diff(after, before) = (dest_inv_s = after.dest_inv_s - before.dest_inv_s,
                                  influence_s = after.influence_s - before.influence_s,
                                  gravity_resid_s = after.gravity_resid_s - before.gravity_resid_s,
                                  n_dest_inversions = after.n_dest_inversions - before.n_dest_inversions,
                                  n_dest_nonconverged = after.n_dest_nonconverged - before.n_dest_nonconverged)

params = (
    server=1, user=2, fakeData=FAKEDATA, DFake=DVAL, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=WVAL, seedU=888,
    importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0, IndMomentOrder=5,
    θConstant=0, gravMoment=1, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1,
    refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=0, calc_δ_star_initial=1, Jac_W=WVAL,
    theta_init=0, runLFD=1, runLFDCounterFactual=1,
)

setup_output = master_setup(params)
@unpack data, counters = setup_output
D = setup_output.D
@assert D == DVAL
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)

γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat; JacW = params.Jac_W
ρ = 2e-3; δ = params.δ_ref
Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
omitted = [d for d in 1:D if d != focal]; ref = 1
logτ = log.(τ); logw = log.(wHat)

const nCM = CM_ENABLED ? n_cm_moments(D, CM_L; include_truncated_moment = CM_EQ36) : 0
if CM_ENABLED
    CM_Moments, cm_thresholds, cm_origins = precalc_common_marginals_cdf(U, CM_REF, CM_L;
        include_truncated_moment = CM_EQ36, μHat = γ.μHat, σHat = σ)
    global γ = (; γ..., CM_Moments = CM_Moments)
    @printf("[common marginals ON] L=%d, refIndex1=%d, %d non-ref origins, eq36=%s -> %d extra moments\n",
            CM_L, CM_REF, length(cm_origins), CM_EQ36, nCM)
end

θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
let γf0 = θr0_orig[3], μ0 = θr0_orig[1]
    global θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
end
const KBOUNDS = theoretical_kappa_bounds(γ, σ)

focal_u(θ) = begin
    μ = θ[1]; Acol = θ[4:3+D]
    AodPow = [ (Acol[o]*((wHat[o]*τ[o,focal])/(wHat[1]*τ[1,focal]))^(1/μ)*(λData[o,focal]/λData[1,focal]))^(-μ) for o in 1:D ]
    (σ - 1) .* (log.(1 ./ AodPow) .- logw .- logτ[:, focal])
end

function recover_lfd(θ, moments_fn, d)
    oci = d + 1
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = JacW, lower_limit = -5000,
        # absolute, @__DIR__-anchored paths -- NOT bare relative filenames; see
        # run_focal_bounds_common_marginals.jl's identical comment for why (a mid-script cwd
        # change in this codebase makes a relative opt-file path fail SILENTLY on later KNITRO
        # calls, degrading results rather than erroring loudly). This function is called on every
        # seq_gravcol iteration, far more often than any other KNITRO entry point in this driver,
        # so it's the single highest-value place to make this robust.
        outer_loop_opt = joinpath(@__DIR__, "..", "ek_outer_loop_options.opt"),
        inner_loop_opt = joinpath(@__DIR__, "..", "ek_inner_loop_options.opt"))
    val, x, nStatus = inner_loop(obj, θ)
    # BUG FIX (ported from the sibling D=20 session's fix to this same function, 2026-07-16):
    # inner_loop_internal(obj::PsiObjectiveBundleDelta,...) already computes δ* and the dual
    # multipliers x together in one KNITRO solve, and already classifies nStatus -- but on a
    # REJECTED status (e.g. -300 = KN_RC_UNBOUNDED, observed at extreme far-from-A* candidates)
    # it only NaNs its own cache field `obj.x`, NOT the x it returns to this caller. Checking only
    # `all(isfinite, x)` (as this function used to) therefore silently accepted the RAW, non-NaN'd
    # garbage KNITRO left behind from a genuinely failed/unbounded inner delta* solve, deriving a
    # bogus LFD from it -- internally self-consistent-looking (small gravity residual, in-budget
    # divergence, computed FROM that same bogus p) while actually corresponding to no finite
    # delta*. Reject explicitly on nStatus, matching the acceptable-status convention already used
    # elsewhere in this codebase (e.g. PsiObjectiveBundleImplicitMethodB's inner_loop_internal).
    nStatus ∈ (0, -100, -101, -103) || return fill(1.0 / W, W), false
    all(isfinite, x) || return fill(1.0 / W, W), false
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), false
    return LFD ./ s, true
end

function divergence_of(p::AbstractVector)
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

function seq_gravcol(θ; δ::Real = δ, maxit = 20, tol = 5e-4, warm = nothing,
                     warm_p::Union{Nothing,AbstractVector} = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    (all(isfinite, uf) && all(isfinite, log_x)) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    # invert_all: shared serial/parallel driver for one full pass over the omitted destinations.
    # u_init_fn(d) supplies each destination's Newton warm start (nothing = cold). Also returns
    # `stats`, a length-D Vector{Any} (only `omitted` entries populated) holding each destination's
    # converged dest_stats result (DestInversion.stats), reused directly by influence_function
    # below instead of recomputing an identical O(S·D) pass at the same u; and `all_converged`, a
    # Bool that is false if ANY destination failed to converge.
    #
    # Diagnostics finding (full_aod_diag/gravity_seeded_initial_solve/): at large divergence
    # budgets, a destination's target share can become numerically unreachable under the current
    # LFD weights (ill-conditioned Hessian). invert_destination correctly reports
    # `converged=false` in that case, but historically NOTHING checked it here -- the (possibly
    # garbage) u_full was written into `um` regardless, silently corrupting gravity_residual and
    # producing nonsensical R_mean values (observed up to ~1e272). The Levenberg-Marquardt damped
    # Newton step in profiled_gravity.jl now prevents the numeric blow-up itself in every case
    # tested, but this explicit check is kept as a second, independent line of defense: if ANY
    # destination fails to converge for ANY reason, the caller treats the whole evaluation as
    # infeasible rather than relying on the resulting R_mean happening to be large enough to fail
    # its own tolerance check.
    function invert_all(p_arg, u_init_fn::Function)
        um = zeros(D, D); um[:, focal] .= uf
        stats = Vector{Any}(undef, D)
        all_converged = Threads.Atomic{Bool}(true)
        if PARALLEL_INVERSION
            Threads.@threads for i in eachindex(omitted)
                d = omitted[i]
                t0 = time()
                inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = DEST_INV_TOL,
                                         maxit = 150, ls_iters = 50, u_init = u_init_fn(d))
                Threads.atomic_add!(SEQ_TIMING.dest_inv_s, time() - t0)
                Threads.atomic_add!(SEQ_TIMING.n_dest_inversions, 1)
                inv.converged || Threads.atomic_add!(SEQ_TIMING.n_dest_nonconverged, 1)
                um[:, d] .= inv.u_full
                stats[d] = inv.stats
                inv.converged || (all_converged[] = false)
            end
        else
            for d in omitted
                t0 = time()
                inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = DEST_INV_TOL,
                                         maxit = 150, ls_iters = 50, u_init = u_init_fn(d))
                Threads.atomic_add!(SEQ_TIMING.dest_inv_s, time() - t0)
                Threads.atomic_add!(SEQ_TIMING.n_dest_inversions, 1)
                um[:, d] .= inv.u_full
                stats[d] = inv.stats
                if !inv.converged
                    Threads.atomic_add!(SEQ_TIMING.n_dest_nonconverged, 1)
                    all_converged[] = false
                    verbose && @printf("      dest %d NOT converged: iters=%d share_err=%.2e ‖u‖=%.2e\n",
                            d, inv.iterations, inv.max_abs_share_error, maximum(abs, inv.u_full))
                end
            end
        end
        um, stats, all_converged[]
    end
    invert_omitted(p_arg; warm = nothing) = invert_all(p_arg, d -> warm === nothing ? nothing : warm[:, d])
    # GRAVITY_SEED (env-gated, off by default -- ported from
    # full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl, evaluated there and
    # REJECTED, see README.md in that dir): if a prior theta's converged (p, umat) is available,
    # seed the first CC solve with the linearized gravity moment from that prior point instead of
    # solving blind (D+1 moments). Falls back to the blind solve otherwise -- identical to baseline
    # when GRAVITY_SEED=false or no prior point exists yet.
    local p, ok
    if GRAVITY_SEED && warm !== nothing && warm_p !== nothing
        t0 = time()
        infl_seed = influence_function(log_x, warm_p, warm, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_beta)
        Threads.atomic_add!(SEQ_TIMING.influence_s, time() - t0)
        moments_aug_seed! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj)
            @. G[:, D+2] = infl_seed.ψ_bar + infl_seed.R_beta
            CM_ENABLED && append_cm_moments!(G, D + 2, obj.γ.CM_Moments)
        end
        p, ok = recover_lfd(θ, moments_aug_seed!, D + 2 + nCM)
    elseif CM_ENABLED
        # blind (no gravity linearization yet), but CM moments are theta-independent/exact --
        # enforce them from the very first LFD recovery rather than deferring them like gravity.
        moments_focal_cm! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj)
            append_cm_moments!(G, D + 1, obj.γ.CM_Moments)
        end
        p, ok = recover_lfd(θ, moments_focal_cm!, D + 1 + nCM)
    else
        p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    end
    if !ok
        verbose && println("    [seq] initial recover_lfd (focal-only) FAILED")
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    local umat, R, stats_cur
    try
        umat, stats_cur, all_ok_init = invert_omitted(p; warm = warm)
        if !all_ok_init
            verbose && println("    [seq] initial invert_omitted: at least one destination did NOT converge -- treating as infeasible")
            return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
        end
        t0 = time()
        R = gravity_residual(umat, logτ, logw, σ).R_mean
        Threads.atomic_add!(SEQ_TIMING.gravity_resid_s, time() - t0)
    catch e
        verbose && println("    [seq] initial invert_omitted threw: ", e)
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    isfinite(R) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    verbose && @printf("    [seq] init: R0=%.4e\n", R)
    col = zeros(W); Rcol = 0.0
    for k in 1:maxit
        t0 = time()
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ,
                                  scale = :R_beta, precomputed_stats = stats_cur)
        Threads.atomic_add!(SEQ_TIMING.influence_s, time() - t0)
        col = infl.ψ_bar .+ infl.R_beta
        Rcol = infl.R_beta
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
            CM_ENABLED && append_cm_moments!(G, D + 2, obj.γ.CM_Moments)
        end
        p_cand, okc = recover_lfd(θ, moments_aug!, D + 2 + nCM)
        if !okc
            verbose && println("    [seq] iter $k: augmented recover_lfd FAILED (linearized moment likely unmatchable)")
            break
        end
        α = 1.0; acc = false
        # Damping/line-search warm start: the α=1 trial (tried first, as before) is warm-started
        # from `umat` (the pre-iteration endpoint) exactly as before, and its own result is cached
        # as the "endpoint-1" solution. Subsequent (smaller) trial alphas warm-start each
        # destination by INTERPOLATING between the two known endpoint solutions instead of always
        # restarting from `umat` -- validated ~1.7x fewer total Newton iterations, bit-exact same
        # accepted trajectory vs always-restart (full_aod_diag/sequential_inversion_performance/
        # warm_start_report.md). Every trial is still solved to the same tol=1e-8 exact optimum
        # regardless of warm start, so this only changes solver speed, never the accepted result.
        um_endpoint1 = nothing; stats_endpoint1 = nothing
        for _ in 1:12
            p_try = (1 - α) .* p .+ α .* p_cand
            local um_try, R_try, stats_try, all_ok_try
            try
                if α == 1.0 || um_endpoint1 === nothing
                    um_try, stats_try, all_ok_try = invert_all(p_try, d -> umat[:, d])
                    if um_endpoint1 === nothing
                        um_endpoint1 = um_try; stats_endpoint1 = stats_try
                    end
                else
                    um_try, stats_try, all_ok_try = invert_all(p_try, d -> (1 - α) .* umat[:, d] .+ α .* um_endpoint1[:, d])
                end
                t0 = time()
                R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
                Threads.atomic_add!(SEQ_TIMING.gravity_resid_s, time() - t0)
            catch
                α *= 0.5; continue
            end
            # require BOTH an improving R and every destination having actually converged --
            # a non-converged destination's u_full is not trustworthy even if the resulting
            # R_try happens to look like an improvement (see invert_all's docstring above).
            if all_ok_try && isfinite(R_try) && abs(R_try) < abs(R)
                p = p_try; umat = um_try; R = R_try; stats_cur = stats_try; acc = true; break
            end
            α *= 0.5
        end
        verbose && @printf("    [seq] iter %d: R_mean -> %.4e  accepted=%s  α=%.4f\n", k, R, acc, acc ? α : 0.0)
        acc || break
    end
    div_p = divergence_of(p)
    gravity_ok = abs(R) <= tol
    δ_ok = div_p <= δ * (1 + 1e-6) + 1e-10
    if verbose
        @printf("    [seq] FINAL: R_mean=%.4e gravity_ok=%s  divergence(p)=%.4e (budget δ=%.4g) δ_ok=%s\n",
                R, gravity_ok, div_p, δ, δ_ok)
    end
    return col, R, Rcol, umat, p, gravity_ok && δ_ok
end

function grad_R_theta(θ, umat, p)
    l = length(θ); dRdθ = zeros(l)
    t0 = time()
    gr = gravity_residual(umat, logτ, logw, σ)
    Threads.atomic_add!(SEQ_TIMING.gravity_resid_s, time() - t0)
    fi = free_idx(ref, D)
    Jfocal = ForwardDiff.jacobian(focal_u, θ)
    c_focal = gr.Qt[:, focal] ./ (σ - 1) ./ gr.S_Q
    dRdθ .+= Jfocal' * c_focal
    logp = log.(p); μ0 = θ[1]
    for d in omitted
        st = dest_stats(build_log_x(Uσ, μ0), logp, umat[:, d]; ρ = ρ)
        Hd = free_hessian(st, ρ; ref = ref)
        dsh = ForwardDiff.derivative(μ -> dest_share(build_log_x(Uσ, μ), logp, umat[:, d]; ρ = ρ)[1][fi], μ0)
        dud_dmu = -(Hd \ dsh)
        c_d = gr.Qt[fi, d] ./ (σ - 1) ./ gr.S_Q
        dRdθ[1] += dot(c_d, dud_dmu)
    end
    return dRdθ
end

# mu FIXED per spec section 1 (removed from the differentiated/optimized vector below, not just
# bounds-pinned -- FreeParamMap excludes index 1 entirely from x_free)
const FREEZE_MU = true
function focal_bounds(θr)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2]; hi[2] = θr[2]
    lo[1] = 0.001; hi[1] = 1/(σ-1) - 0.001
    lo[3] = KBOUNDS.γp_lo; hi[3] = KBOUNDS.γp_hi
    if FREEZE_MU
        lo[1] = θr[1]; hi[1] = θr[1]
    end
    lo, hi
end
θ_lo, θ_hi = focal_bounds(θr0)

const INFCOL = 1.0 .+ 0.1 .* sin.(1:W)

function make_stateful_moments(; use_exact_grad::Bool = true, find_smallest::Bool = false, δ::Real = δ)
    lastθ = Ref(fill(NaN, length(θr0)))
    gcol  = Ref(zeros(W))
    lastRmean = Ref(NaN)
    lastRcol  = Ref(NaN)
    lastok = Ref(true)
    dRdθ  = Ref(zeros(length(θr0)))
    neval = Ref(0); nfeas = Ref(0)
    warm  = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    warm_p = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_θ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_κ = Ref(find_smallest ? Inf : -Inf)
    best_warm = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Ktmp = zeros(1); Gtmp = zeros(1, D + 1)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rmean, Rcol, um, p, ok = seq_gravcol(θf; δ = δ, warm = warm[], warm_p = warm_p[])
                lastRmean[] = Rmean; lastRcol[] = Rcol; lastok[] = ok
                if ok
                    gcol[] = c; warm[] = um; warm_p[] = p; nfeas[] += 1
                    dRdθ[] = use_exact_grad ? grad_R_theta(θf, um, p) : zeros(length(θf))
                    EK_moments_focal_norm_directgp!(Ktmp, Gtmp, θf, view(U, 1:1, :), (γ = γ,))
                    κθ = Ktmp[1]
                    if (find_smallest && κθ < best_κ[]) || (!find_smallest && κθ > best_κ[])
                        best_κ[] = κθ; best_θ[] = copy(θf); best_warm[] = copy(um)
                    end
                else
                    gcol[] = INFCOL
                    dRdθ[] = zeros(length(θf))
                end
                neval[] += 1
                if neval[] % 10 == 0
                    @printf("    [θ-eval %d, gravity-feasible %d] seqR_mean=%.2e ok=%s seq_time=%.2fs\n",
                            neval[], nfeas[], lastRmean[], ok, time()-t0); flush(stdout)
                end
                lastθ[] = copy(θf)
            end
        end
        EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θ, Uarg, obj)
        nrow = size(G, 1)
        if eltype(θ) <: ForwardDiff.Dual && lastok[]
            Rlin = lastRcol[] + dot(dRdθ[], θ .- lastθ[])
            @inbounds @views @. G[:, D+2] = (gcol[][1:nrow] - lastRcol[]) + Rlin
        else
            @inbounds @views @. G[:, D+2] = gcol[][1:nrow]
        end
        CM_ENABLED && append_cm_moments!(G, D + 2, obj.γ.CM_Moments)
    end
    return m!, gcol, lastRmean, best_θ, best_κ, best_warm
end

# ---- NEW: cached, free-only outer solve (replaces plain `outer_loop` over the full theta) ----
function make_seq_div_grad_fn!(obj, fpmap)
    d = obj.d; oci = obj.outer_constr_index
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        obj(inner_x, Float64[], Float64[]; constr = zeros(1))   # trigger dPsi!, populate obj.arg1
        λ = collect(@view inner_x[2:end])
        Usub = obj.U[1:obj.N, :]
        f = x -> CS._methodB_envelope_scalar(reconstruct_full(x, fpmap), obj.moments!, obj.γ, Usub, λ, obj.arg1, d, oci)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f, x_free)
        end
        ForwardDiff.gradient!(g_free, f, x_free, cfg_cache[])
        return g_free
    end
end

function outer_solve_nested_cached(find_smallest, θinit; use_exact_grad::Bool = true, δ::Real = δ)
    d = D + 2 + nCM; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm = make_stateful_moments(; use_exact_grad = use_exact_grad, find_smallest = find_smallest, δ = δ)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,
        outer_loop_opt = OUTER_OPT_FILE, inner_loop_opt = INNER_OPT_FILE)

    l_full = length(θinit)
    free_idx = vcat(3, collect(4:3+D))    # gamma'_focal + A[.,focal]
    fixed_idx = [1, 2]                     # mu, sigma
    fixed_vals = θinit[fixed_idx]
    fpmap = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(fpmap) == D + 1

    div_grad_fn! = make_seq_div_grad_fn!(obj, fpmap)
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0)
        g_free[1] = (-1.0)^find_smallest   # K = gamma'_focal = x_free[1] directly
    end

    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = OUTER_OPT_FILE)

    gp = r.θ_min_full[3]
    gp, r.θ_min_full, r.nStatus, best_θ[], best_κ[], best_warm[], r.cache
end

Kchk = zeros(W); Gchk = zeros(W, D + 1); EK_moments_focal_norm_directgp!(Kchk, Gchk, θr0, U, (γ = γ,))
gp2kappa(gp) = 1 - gp^(σ/(σ-1))

# ================================================================================================
# AUDIT LOGIC
# ================================================================================================

const JLD2_PATH = ENV["JLD2_PATH"]
saved = JLD2.load(JLD2_PATH)
@printf("\n=== AUDIT: %s ===\n", JLD2_PATH)
@printf("saved: kappa=%.6f  gamma_p=%.6f  R_mean_at_solution=%.3e  gravity_feasible=%s  nStatus=%d\n",
        saved["kappa"], saved["gamma_p"], saved["R_mean_at_solution"], saved["gravity_feasible"], saved["nStatus"])
@printf("saved: best_feasible_kappa=%.6f  best_feasible_gp=%.6f  best_feasible_gravity_ok=%s\n",
        saved["best_feasible_kappa"], saved["best_feasible_gp"], saved["best_feasible_gravity_ok"])

function audit_theta(label, θaudit)
    isnothing_or_nan = θaudit === nothing || any(isnan, θaudit)
    if isnothing_or_nan
        @printf("  [%s] theta is nothing/NaN -- skipping\n", label)
        return
    end
    t0 = time()
    col, Rmean, Rcol, umat, p, ok = seq_gravcol(θaudit; δ = δ, maxit = 20, tol = 5e-4, verbose = false)
    wall = time() - t0
    divp = ok ? divergence_of(p) : NaN
    κ_audit = gp2kappa(θaudit[3])
    @printf("  [%s] gamma_p=%.6f  kappa=%.6f  R_mean=%.3e  divergence(p)=%.6f  budget=%.4g  gravity+delta_ok=%s  wall=%.1fs\n",
            label, θaudit[3], κ_audit, Rmean, divp, δ, ok, wall)
    if ok
        @printf("      -> RE-VERIFIED: delta* (divergence of the recovered LFD) = %.6f, %s the budget %.4g -- result STANDS.\n",
                divp, divp <= δ * (1 + 1e-6) + 1e-10 ? "within" : "EXCEEDS", δ)
    else
        @printf("      -> RE-VERIFICATION FAILED under the fixed recover_lfd: this theta is NOT gravity/divergence-feasible -- ORIGINAL RESULT WAS SPURIOUS.\n")
    end
    return (ok = ok, div = divp, Rmean = Rmean, κ = κ_audit)
end

println("\n--- Re-verifying theta_star (the raw KNITRO endpoint) with the FIXED recover_lfd ---")
audit_theta("theta_star", saved["theta_star"])

println("\n--- Re-verifying best_feasible_theta (the run's own tracked best-feasible point) with the FIXED recover_lfd ---")
audit_theta("best_feasible_theta", saved["best_feasible_theta"])

println("\nAUDIT DONE")
