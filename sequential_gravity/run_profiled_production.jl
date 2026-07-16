# ============================================================================
# Sequential-profiled production driver: exact-point cache + free-only
# ForwardDiff envelope gradient (mu, sigma genuinely REMOVED from the
# differentiated/optimized vector via FreeParamMap, not just bounds-pinned).
#
# Identical sequential-loop/inversion/warm-start machinery to
# sequential_gravity/run_profiled_D10_methodB.jl (seq_gravcol, grad_R_theta,
# make_stateful_moments -- copied verbatim, byte-identical logic; the ONLY
# change is HOW the outer KNITRO problem is wired: outer_loop_cached +
# FreeParamMap instead of plain outer_loop over the full theta vector).
# D is a parameter (env DVAL, default 4 for a quick validation run before
# scaling to D=10).
#
#   julia --project=. sequential_gravity/run_profiled_production.jl
#   DVAL=10 DELTA_GRID=0.1,1.0,10.0 julia --project=. sequential_gravity/run_profiled_production.jl
# ============================================================================

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, Printf

# Shared-machine courtesy cap: OpenBLAS defaults to using ALL available cores for the
# linear-algebra-heavy parts of the inner/outer solve (Hessian outer-products, ForwardDiff
# Jacobians), which is bad manners on a heavily multi-tenant box. Cap explicitly rather than
# relying on the launching shell to set OPENBLAS_NUM_THREADS. 19 matches this repo's own
# PARALLEL_INVERSION convention (D-1 destinations at D=20). Ported from the sibling
# common-marginals session, 2026-07-16.
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
# Independent hard-max (rho=0) verification of a converged best-feasible point -- see
# hardmax_verify.jl's own docstring and derivative_diagnostics/hardmax_inversion_report.md.
# Wired into run_one_bound below (VERIFY_HARDMAX env var, default on).
include(joinpath(@__DIR__, "hardmax_verify.jl"))
# Common-marginals restriction (CDW eq. 35/36) -- theta-independent extra CDF-equality inner-loop
# moments, appended after the gravity column. Off by default (CM_L=0); see CM_L/CM_REF/CM_EQ36
# below and common_marginals_moments.jl's own file-level comment. Ported from the sibling
# common-marginals session, 2026-07-16.
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
CS.include(joinpath(@__DIR__, "PsiObjectiveBundleImplicitMethodB.jl"))
# gradient_method support (Part 5 of the winner-boundary-derivative task): additive, no effect
# unless outer_solve_nested_cached is called with gradient_method != :pointwise_ad (the default).
include(joinpath(@__DIR__, "derivative_diagnostics", "fixed_dual_criterion.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "fixed_dual_fd.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "full_fixed_dual_criterion.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "full_profile_resolve.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "full_sample_exact_control.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "fixed_A_incumbent.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "boundary_derivative.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "gradient_method_wiring.jl"))
include(joinpath(@__DIR__, "derivative_diagnostics", "full_gradient_method_wiring.jl"))

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
# inner-loop moments (common_marginals_moments.jl), appended after the gravity column. CM_L=0
# (default) is OFF and reproduces the prior unrestricted behavior exactly -- nothing below changes
# shape/timing/results when CM_L=0. CM_REF is the fixed reference country for the CDF comparison
# (paper convention: distinct from, and unrelated to, baseIndex/focal). Ported from the sibling
# common-marginals session, 2026-07-16.
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

# Warm-starting the inner CC dual solve (recover_lfd's KNITRO call). Investigated + validated
# 2026-07-16 (head_to_head/experiment_dual_warmstart*.jl, HANDOFF_2026-07-16_recover_lfd_bug.md):
# mechanically correct and never hurts (identical accepted results, iteration counts equal-or-lower
# vs cold at every call in the controlled 3-point test), so :persist is the default. Three modes:
#   :cold             -- never warm-start (old behavior, every recover_lfd call solves from zero)
#   :reset_per_theta  -- warm-start ACROSS seq_gravcol's own within-theta k=1..maxit augmented
#                        re-solves (re-solving because the linearized gravity moment shifts each
#                        iteration), but reset to cold at the start of every NEW theta
#   :persist          -- same within-theta warm start, PLUS never reset across theta either -- a new
#                        theta's first (blind) recover_lfd call warm-starts from the previous
#                        theta's last converged dual. Only ever cached on a converged solve (nStatus
#                        acceptable) -- a failed solve never poisons the cache.
# Global mutable cache (not threaded through function signatures, to avoid touching the ~50 files
# across this repo that call recover_lfd/seq_gravcol with their existing signatures) -- safe ONLY
# because recover_lfd/seq_gravcol are never called concurrently from multiple threads in this
# codebase (BlackBoxOptim's own fitness evaluation is serial; PARALLEL_INVERSION's Threads.@threads
# is one level BELOW recover_lfd, over destinations, and never re-enters it). If that ever changes
# (e.g. parallel BBO population evaluation), this cache would need to become task-local.
const DUAL_WARM_MODE = Ref{Symbol}(Symbol(get(ENV, "DUAL_WARM_MODE", "persist")))  # :cold | :reset_per_theta | :persist
const _DUAL_CACHE = Dict{Symbol,Any}(:blind => nothing, :aug => nothing)
@assert DUAL_WARM_MODE[] ∈ (:cold, :reset_per_theta, :persist) "DUAL_WARM_MODE must be :cold, :reset_per_theta, or :persist, got $(DUAL_WARM_MODE[])"

# Handles the D+2 (blind, D+1 moments) <-> D+3 (augmented, D+2 moments once gravity is linearized
# in) dimension mismatch: truncates a longer cached dual (drop the newest lambda) or zero-pads a
# shorter one (cold-start only the genuinely-new coordinate), reusing the rest as-is.
function _dual_warmstart_for(target_oci::Int)
    DUAL_WARM_MODE[] === :cold && return nothing
    xa = _DUAL_CACHE[:aug]
    if xa !== nothing
        length(xa) == target_oci && return copy(xa)
        length(xa) >  target_oci && return xa[1:target_oci]
    end
    xb = _DUAL_CACHE[:blind]
    if xb !== nothing
        length(xb) == target_oci && return copy(xb)
        length(xb) <  target_oci && return vcat(xb, zeros(target_oci - length(xb)))
    end
    return nothing
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
    x_init = _dual_warmstart_for(oci)
    use_warm = x_init !== nothing
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = JacW, lower_limit = -5000, use_cached_x = use_warm,
        # absolute, @__DIR__-anchored paths -- NOT bare relative filenames. This codebase's own
        # setup (setup/setwd.jl) can `cd()` into a DIFFERENT sibling worktree mid-script (confirmed
        # this session: sequential_gravity/delta_star_schedule.jl's own JLD2 output landed in
        # trade_robustness_modular instead of trade_robustness_modular_perf for exactly this
        # reason), which makes a relative opt-file path fail SILENTLY on later KNITRO calls,
        # degrading results rather than erroring loudly. This function is called on every
        # seq_gravcol iteration, far more often than any other KNITRO entry point in this driver,
        # so it's the single highest-value place to make this robust. Ported from the sibling
        # common-marginals session, 2026-07-16.
        outer_loop_opt = joinpath(@__DIR__, "..", "ek_outer_loop_options.opt"),
        inner_loop_opt = joinpath(@__DIR__, "..", "ek_inner_loop_options.opt"))
    use_warm && (obj.x .= x_init)
    val, x, nStatus = inner_loop(obj, θ)
    # BUG FIX (2026-07-16): inner_loop_internal(obj::PsiObjectiveBundleDelta,...) already computes
    # δ* and the dual multipliers x together in one KNITRO solve, and already classifies nStatus --
    # but on a REJECTED status (e.g. -300 = KN_RC_UNBOUNDED, observed in practice at extreme
    # far-from-A* candidates reached by the global/population methods) it only NaNs its own cache
    # field `obj.x`, NOT the x it returns to this caller. Checking only `all(isfinite, x)` (as this
    # function used to) therefore silently accepted the RAW, non-NaN'd garbage KNITRO left behind
    # from a genuinely failed/unbounded solve, deriving a bogus LFD from it -- which then looked
    # internally consistent (small gravity residual, in-budget divergence, computed FROM that same
    # bogus p) while the underlying trade shares were violated by several percent. Confirmed via a
    # direct re-run: nStatus was -300 with all(isfinite,x)==true. Reject explicitly on nStatus,
    # matching the same acceptable-status convention already used elsewhere in this codebase (e.g.
    # outer_solve_nested_cached's own probe-solve check).
    nStatus ∈ (0, -100, -101, -103) || return fill(1.0 / W, W), false
    all(isfinite, x) || return fill(1.0 / W, W), false
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), false
    DUAL_WARM_MODE[] !== :cold && (_DUAL_CACHE[d == D + 1 ? :blind : :aug] = x)
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
    # Every top-level seq_gravcol call is, by construction, one genuinely new theta -- so this is
    # the right place to reset the dual-warm-start cache for :reset_per_theta (see DUAL_WARM_MODE
    # above). :persist never resets here; :cold never populates the cache in the first place.
    DUAL_WARM_MODE[] === :reset_per_theta && (_DUAL_CACHE[:blind] = nothing; _DUAL_CACHE[:aug] = nothing)
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
    # lastθ/lastRcol/dRdθ are additionally exposed (beyond the pre-existing gcol/lastRmean) so that
    # an external caller (derivative_diagnostics/full_fixed_dual_criterion.jl) can reconstruct the
    # IDENTICAL frozen affine gravity-moment surrogate this closure uses internally for Dual theta,
    # evaluated at Float64 theta too (needed for finite differences, which perturb Float64, not
    # Dual, theta) -- see full_fixed_dual_criterion.jl::make_frozen_gravity_moments. Purely additive:
    # no change to any existing return value or behavior.
    return m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ, lastRcol, dRdθ, lastok
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

"""
    outer_solve_nested_cached(find_smallest, θinit; use_exact_grad=true, δ=δ, gradient_method=:pointwise_ad)

`gradient_method` selects the divergence-budget outer-constraint gradient:
  - `:pointwise_ad`      -- unchanged existing behavior (ForwardDiff through
                            the hard argmin winner; provably misses the
                            winner-boundary term, see derivative_methods_report.md).
  - `:fixed_dual_fd_full`-- CORRECT: central finite differences of the exact
                            FULL (D+2)-moment fixed-dual criterion (includes
                            the gravity-linearized moment and its own dual
                            multiplier lambda_R in the conjugate argument, not
                            just the D+1 trade/price-index moments). Directly
                            REPLACES the Acol block of g_free (no additive
                            correction). See full_gradient_method_wiring.jl.
  - `:boundary_full`     -- full-(D+2) conditional winner-boundary estimator
                            (analogous correction to :fixed_dual_fd_full).
  - `:fixed_dual_fd`, `:boundary` -- OLD, DEPRECATED (D+1)-reduced additive-
                            correction methods (gradient_method_wiring.jl).
                            These silently dropped lambda_R*G_R from the
                            winner-boundary jump, which is not generally
                            valid since Psi(arg0) is nonlinear in the full
                            conjugate argument. Kept only for A/B regression
                            comparison against the corrected methods -- do
                            NOT use for production/paper results.
For all methods, gamma'_focal's own gradient component (x_free[1]) is the
existing full-(D+2) AD gradient, which is exact for that coordinate (no
winner/argmax dependence on theta[3]).
"""
function outer_solve_nested_cached(find_smallest, θinit; use_exact_grad::Bool = true, δ::Real = δ,
        gradient_method::Symbol = :pointwise_ad, use_var_scaling::Bool = false, scaling_power::Float64 = 1.0)
    d = D + 2 + nCM; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ_st, lastRcol_st, dRdθ_st, lastok_st = make_stateful_moments(; use_exact_grad = use_exact_grad, find_smallest = find_smallest, δ = δ)
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

    div_grad_fn! = if gradient_method == :pointwise_ad
        make_seq_div_grad_fn!(obj, fpmap)
    elseif gradient_method in (:fixed_dual_fd, :boundary)
        # OLD, flawed (D+1)-reduced additive-correction methods -- kept only for A/B regression
        # comparison against the corrected :fixed_dual_fd_full / :boundary_full methods below; do
        # not use for production results (see full_gradient_method_wiring.jl's module docstring).
        make_seq_div_grad_fn_corrected!(obj, fpmap, γ, U, D, (1 / θinit[1]) / (σ - 1), gradient_method)
    elseif gradient_method in (:fixed_dual_fd_full, :boundary_full)
        # nCM (common-marginals extra moments, 0 when CM_ENABLED=false): make_seq_div_grad_fn_full!
        # now threads this through to build the frozen full-(D+2+nCM)-moment fixed-dual criterion
        # correctly, including the CM block -- fixed 2026-07-16 (was previously a hard D+2-only
        # assumption; GRADIENT_METHOD=fixed_dual_fd_full + CM_ENABLED=true used to fail). NOTE:
        # :boundary_full's own gradient path (boundary_envelope_gradient_full) is separately,
        # pre-existingly unimplemented in this codebase (not defined in any included file) --
        # unrelated to common marginals, not addressed here; :fixed_dual_fd_full is the only fully
        # working corrected method.
        make_seq_div_grad_fn_full!(obj, fpmap, γ, U, D, gcol, lastθ_st, lastRcol_st, dRdθ_st, lastok_st, gradient_method; nCM = nCM)
    else
        error("unknown gradient_method $gradient_method")
    end
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0)
        g_free[1] = (-1.0)^find_smallest   # K = gamma'_focal = x_free[1] directly
    end

    # Optional per-variable KNITRO scaling (Part 12-adjacent numerical fix, additive/off by
    # default): a probe inner solve + one gradient evaluation AT theta_init gives a representative
    # constraint-gradient magnitude per free coordinate. Addresses a severe cross-variable scale
    # mismatch found at D=20 real data (gamma'_focal's own constraint-gradient component ~1e8 vs
    # the entire Acol block ~10-11500), which appeared to leave Acol unexplored regardless of
    # gradient_method. This probe solve is thrown away (one extra inner solve, negligible next to
    # the whole outer search) -- it does not seed outer_loop_cached's own cache.
    #
    # IMPORTANT (first attempt got this wrong): scaling EVERY coordinate by 1/|g_probe[i]|,
    # INCLUDING x_free[1]=gamma'_focal, also rescales the OUTER OBJECTIVE's gradient (obj_grad_fn!
    # is a hard-coded +-1 at index 1, 0 elsewhere -- untouched by var_scales, since KNITRO variable
    # scaling is a property of the VARIABLE shared by objective AND constraint). Scaling index 1 by
    # 1/|g_probe[1]| (~1e-9 at D=4) shrinks the SCALED objective gradient to ~0 everywhere, so
    # KNITRO's own KKT check is satisfied trivially at lambda~0 WITHOUT moving at all -- caught
    # immediately: a D=4 test converged in 0 iterations at theta_init with the naive scaling,
    # WORSE than no scaling at all (which genuinely explores and improves kappa). Fixed: leave
    # gamma'_focal's own scale at 1.0 (preserving the objective's natural units exactly), and
    # rescale ONLY the Acol block RELATIVE to gamma's own constraint-gradient magnitude, so each
    # Acol coordinate's constraint-sensitivity becomes comparable in ABSOLUTE size to gamma's own
    # (not driven to some arbitrary O(1) that ignores the objective's own natural scale).
    var_scales = nothing
    if use_var_scaling
        x_free0 = CS.pack_free(θinit, fpmap)
        _, inner_x0, nStatus0 = inner_loop(obj, θinit)
        nStatus0 in (0, -100, -101, -103) || @warn "use_var_scaling: probe inner solve at theta_init did not cleanly converge (status=$nStatus0); scaling may be unreliable"
        g_probe = zeros(length(x_free0))
        div_grad_fn!(g_probe, x_free0, θinit, inner_x0)
        ref_mag = abs(g_probe[1])   # gamma'_focal's own constraint-gradient magnitude -- the reference scale
        floor_mag = 1e-8 * maximum(abs, g_probe)
        # scaling_power=1.0 (full ratio match) OVERCORRECTED at D=4: A moved 383% (vs 31%
        # unscaled) but converged to a WORSE, non-converged (status -102) result in 6x the wall
        # time -- KNITRO evidently takes too-aggressive steps in the newly-inflated Acol
        # directions. scaling_power<1 (e.g. 0.5 = sqrt of the ratio) is a caller-tunable
        # compromise between "no correction" (1.0, i.e. scale=1 for Acol, the pre-scaling
        # default reached via power=0) and "full magnitude match" (power=1).
        var_scales = [1.0; [(ref_mag / max(abs(g_probe[i]), floor_mag))^scaling_power for i in 2:length(g_probe)]]
        @printf("  [use_var_scaling] probe |g_free| range = [%.3e, %.3e] (gamma'=%.3e)  Acol scaleFactors range = [%.3e, %.3e]\n",
                minimum(abs, g_probe), maximum(abs, g_probe), ref_mag, minimum(var_scales[2:end]), maximum(var_scales[2:end]))
    end

    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = OUTER_OPT_FILE, var_scales = var_scales)

    gp = r.θ_min_full[3]
    gp, r.θ_min_full, r.nStatus, best_θ[], best_κ[], best_warm[], r.cache
end

Kchk = zeros(W); Gchk = zeros(W, D + 1); EK_moments_focal_norm_directgp!(Kchk, Gchk, θr0, U, (γ = γ,))
GP_POINT_EST = Kchk[1]
KAPPA_POINT_EST = 1 - GP_POINT_EST^(σ/(σ-1))
@printf("point estimate γ'_focal(F*) = %.6f  ->  kappa point estimate = %.6f\n", GP_POINT_EST, KAPPA_POINT_EST)

gp2kappa(gp) = 1 - gp^(σ/(σ-1))

"""
`GRADIENT_METHOD` env var (default `fixed_dual_fd_full` as of 2026-07-16): selects
`outer_solve_nested_cached`'s `gradient_method` keyword for the WHOLE batch loop below --
`fixed_dual_fd_full` (CORRECT: full-(D+2) fixed-dual finite-difference derivative, see
derivative_diagnostics/full_d2_correction_report.md -- this is the config that actually won the
D=20 real-data 4-method head-to-head comparison, head_to_head/FINAL_REPORT_2026-07-16.md),
`pointwise_ad` (OLD default until 2026-07-16 -- provably misses the winner-boundary term via
ForwardDiff through the hard argmin's Bool; can report false/premature convergence, directly
re-confirmed 2026-07-16: a pointwise_ad+no-scaling run converged in 2 outer iterations to
kappa=0.0377 vs fixed_dual_fd_full's genuine 0.0821 at the identical delta=1.0 budget -- kept
available for A/B comparison only, do not use for anything reported), or the deprecated
reduced-model `fixed_dual_fd`/`boundary` kept only for A/B comparison.
"""
const GRADIENT_METHOD = Symbol(get(ENV, "GRADIENT_METHOD", "fixed_dual_fd_full"))
"""
`USE_VAR_SCALING` env var (default true as of 2026-07-16, matching the config that won the
head-to-head comparison): threads `outer_solve_nested_cached`'s `use_var_scaling` keyword -- see
that function's own comment for what this does and why (a severe gamma'-vs-Acol
constraint-gradient scale mismatch found at D=20 real data that appeared to leave Acol
unexplored regardless of gradient_method).
"""
const USE_VAR_SCALING = lowercase(get(ENV, "USE_VAR_SCALING", "true")) in ("1", "true", "yes")
"""
`SCALING_POWER` env var (default 1.0, matching `outer_solve_nested_cached`'s own default):
threads the `scaling_power` keyword through. BUG FOUND AND FIXED: this was previously NOT
threaded through `run_one_bound` at all -- every batch-loop run using USE_VAR_SCALING=true
(including the D=20 delta-grid run reported as "scaling_power=0.5" in
full_d2_correction_report.md/D20_METHOD_WRITEUP.md) silently used the function default
(1.0), not whatever SCALING_POWER env var was set (that env var only existed in the
standalone debug_scaling_d4.jl script until now). The D=20 delta-grid RESULTS themselves
remain independently verified and valid (see verify_d20_deltagrid.jl) -- only their
documented scaling_power label was wrong; see the correction in full_d2_correction_report.md.
"""
const SCALING_POWER = parse(Float64, get(ENV, "SCALING_POWER", "1.0"))
"""
`VERIFY_HARDMAX` env var (default true): after the outer loop settles on a best-feasible theta,
independently re-check it against the TRUE hard-max (rho=0) economic model (`hardmax_verify.jl`)
before it gets saved/reported, rather than trusting the smoothed (rho>0) inversion's own
gravity-feasibility check alone. Adds ~1-3 min per saved (bound,delta) checkpoint at D=20/W=80000
(see derivative_diagnostics/hardmax_inversion_report.md for timing at other W). Off-switch is
for fast dev iteration only -- keep this on for anything whose kappa/gamma' might get reported.
"""
const VERIFY_HARDMAX = lowercase(get(ENV, "VERIFY_HARDMAX", "true")) in ("1", "true", "yes")

function run_one_bound(name::Symbol, fs::Bool, δval::Real, θinit)
    @printf("\n----- %s bound: gamma'_focal %s, delta=%g (warm-started), gradient_method=%s, use_var_scaling=%s, scaling_power=%.3g -----\n", name, fs ? "MINIMIZED" : "MAXIMIZED", δval, GRADIENT_METHOD, USE_VAR_SCALING, SCALING_POWER); flush(stdout)
    t0 = time()
    timing_before = seq_timing_snapshot()
    gp, θstar, st, bθ, b_gp, bwarm, cache = outer_solve_nested_cached(fs, θinit; use_exact_grad = true, δ = δval, gradient_method = GRADIENT_METHOD, use_var_scaling = USE_VAR_SCALING, scaling_power = SCALING_POWER)
    κ = gp2kappa(gp)
    _, Rθ, _, _, _, okθ = seq_gravcol(θstar; δ = δval)
    CS.summarize(cache; label = "$name bound cache stats")
    seqt = seq_timing_diff(seq_timing_snapshot(), timing_before)
    wall = time() - t0
    @printf("  KNITRO:        gamma'_%s = %.6f -> kappa = %.6f  (status %d)  exact R_mean(θ*) = %.3e  gravity-feasible=%s  wall %.1fs\n",
            name, gp, κ, st, Rθ, okθ, wall)
    @printf("  seq timing:    dest_inversion=%.2fs (n=%d, nonconverged=%d)  influence_fn=%.2fs  gravity_residual=%.2fs\n",
            seqt.dest_inv_s, seqt.n_dest_inversions, seqt.n_dest_nonconverged, seqt.influence_s, seqt.gravity_resid_s)
    if bθ === nothing
        @printf("  best-feasible: NONE FOUND\n")
        return (κ = κ, gp = gp, R = Rθ, ok = okθ, best_κ = NaN, best_gp = NaN, best_θ = nothing, best_ok = false,
                θstar = θstar, cache = cache, nStatus = st, wall = wall, seqt = seqt,
                hardmax_verified = false, hardmax_focal_err = NaN, hardmax_R_mean = NaN,
                hardmax_max_share_err = NaN, hardmax_mean_share_err = NaN, hardmax_homotopy_all_ok = false,
                hardmax_wall = 0.0)
    else
        _, Rb, _, umat_b, p_b, okb = seq_gravcol(bθ; δ = δval, warm = bwarm)
        bκ = gp2kappa(b_gp)
        @printf("  best-feasible: gamma'_%s = %.6f -> kappa = %.6f  exact R_mean = %.3e  gravity-feasible=%s\n", name, b_gp, bκ, Rb, okb)
        if VERIFY_HARDMAX
            hv = verify_hardmax_point(bθ, umat_b, p_b)
            @printf("  HARD-MAX VERIFY: hardmax_verified=%s  R_mean_hardmax=%.3e  focal_err=%.3e  max_share_err=%.3e  mean_share_err=%.3e  homotopy_all_ok=%s  wall=%.1fs\n",
                    hv.verified, hv.R_mean_hardmax, hv.focal_err, hv.max_hard_err, hv.mean_hard_err, hv.homotopy_all_ok, hv.wall)
            hv.verified || @printf("  *** WARNING: this point is NOT independently hard-max-verified -- the reported kappa/gamma' rests on the SMOOTHED model's gravity check only ***\n")
        else
            hv = (verified = missing, focal_err = NaN, R_mean_hardmax = NaN, max_hard_err = NaN,
                  mean_hard_err = NaN, homotopy_all_ok = missing, wall = 0.0)
        end
        return (κ = κ, gp = gp, R = Rθ, ok = okθ, best_κ = bκ, best_gp = b_gp, best_θ = bθ, best_ok = okb,
                θstar = θstar, cache = cache, nStatus = st, wall = wall, seqt = seqt,
                hardmax_verified = hv.verified, hardmax_focal_err = hv.focal_err, hardmax_R_mean = hv.R_mean_hardmax,
                hardmax_max_share_err = hv.max_hard_err, hardmax_mean_share_err = hv.mean_hard_err,
                hardmax_homotopy_all_ok = hv.homotopy_all_ok, hardmax_wall = hv.wall)
    end
end

DELTA_GRID = sort(let s = get(ENV, "DELTA_GRID", "")
    isempty(s) ? [δ] : parse.(Float64, split(s, ","))
end)   # ascending: warm-start chain runs small-delta-first, per spec section 11
const BOUND_ARG = get(ENV, "BOUND", "both")
const OUT_DIR = get(ENV, "OUT_DIR", joinpath(@__DIR__, "batch_out"))
isdir(OUT_DIR) || mkpath(OUT_DIR)

result_path(bound_name, δval) = joinpath(OUT_DIR, "seq_$(bound_name)_delta$(δval).jld2")

function load_if_done(path)
    isfile(path) || return nothing
    d = try
        JLD2.load(path)
    catch
        return nothing
    end
    get(d, "done", false) === true ? d : nothing
end

@printf("\n=== [PRODUCTION: profiled/sequential, cached, free-only ForwardDiff] D=%d W=%d ρ=%g MU_FIXED(removed)=%s DELTA_GRID=%s GRAVITY_SEED=%s ===\n",
        D, W, ρ, FREEZE_MU, DELTA_GRID, GRAVITY_SEED)
@printf("point estimate kappa = %.6f\n", KAPPA_POINT_EST)

# SKIP_BATCH_LOOP (additive, off by default -- default behavior is completely unchanged): lets a
# driver `include` this file purely for its setup/function definitions (economy, theta_r0,
# outer_solve_nested_cached, etc. -- e.g. sequential_gravity/derivative_diagnostics/
# run_part5_gradient_method_comparison.jl) without triggering the default 12-solve batch below.
if lowercase(get(ENV, "SKIP_BATCH_LOOP", "false")) != "true"
for (name, fs) in ((:lower, false), (:upper, true))
    BOUND_ARG in ("both", String(name)) || continue
    θcur = copy(θr0)   # warm-start chain within this bound direction only
    for δval in DELTA_GRID
        path = result_path(name, δval)
        existing = load_if_done(path)
        if existing !== nothing
            @printf("\n----- %s bound, delta=%g -- ALREADY DONE, skipping (resume) -----\n", name, δval)
            # Warm-start the NEXT delta from the verified gravity-FEASIBLE best point, not the
            # raw KNITRO endpoint (which can be, and at D=20/W=80000 often was, gravity-infeasible
            # -- chaining the warm start from an infeasible point propagated a broken starting
            # point through the entire rest of the delta grid, caught when delta=2.0/5.0 both
            # found ZERO feasible points after inheriting delta=1.0's infeasible raw endpoint).
            existing_best = get(existing, "best_feasible_theta", nothing)
            if existing_best !== nothing
                θcur = existing_best
            else
                @printf("  WARNING: no feasible point saved for delta=%g -- falling back to the raw (possibly infeasible) endpoint for warm-starting the next delta\n", δval)
                θcur = existing["theta_star"]
            end
            flush(stdout)
            continue
        end
        r = run_one_bound(name, fs, δval, θcur)
        JLD2.save(path, Dict(
            "method" => "sequential_profiled", "bound" => String(name), "delta" => δval, "D" => D,
            "theta_star" => r.θstar, "gamma_p" => r.gp, "kappa" => r.κ, "nStatus" => r.nStatus,
            "R_mean_at_solution" => r.R, "gravity_feasible" => r.ok,
            "best_feasible_kappa" => r.best_κ, "best_feasible_gp" => r.best_gp,
            "best_feasible_theta" => r.best_θ, "best_feasible_gravity_ok" => r.best_ok,
            # Independent hard-max (rho=0) re-verification of best_feasible_theta -- a NEW,
            # SEPARATE flag, does NOT change the meaning of best_feasible_gravity_ok above (that
            # remains the smoothed-model check exactly as before). See hardmax_verify.jl /
            # derivative_diagnostics/hardmax_inversion_report.md. `missing` if VERIFY_HARDMAX=false.
            "hardmax_verified" => r.hardmax_verified, "hardmax_R_mean" => r.hardmax_R_mean,
            "hardmax_focal_err" => r.hardmax_focal_err, "hardmax_max_share_err" => r.hardmax_max_share_err,
            "hardmax_mean_share_err" => r.hardmax_mean_share_err, "hardmax_homotopy_all_ok" => r.hardmax_homotopy_all_ok,
            "hardmax_verify_wall" => r.hardmax_wall,
            "wall" => r.wall,
            "unique_free_x" => length(Set(rr.x_hash for rr in r.cache.trace)),
            "inner_solves" => r.cache.n_inner_solve, "grad_computations" => r.cache.n_grad_compute,
            "warm_started_inner" => r.cache.n_warm_started, "cold_inner" => r.cache.n_cold,
            "t_inner" => r.cache.t_inner, "t_grad" => r.cache.t_grad,
            "t_dest_inversion" => r.seqt.dest_inv_s, "t_influence_function" => r.seqt.influence_s,
            "t_gravity_residual" => r.seqt.gravity_resid_s,
            "n_dest_inversions" => r.seqt.n_dest_inversions, "n_dest_nonconverged" => r.seqt.n_dest_nonconverged,
            "starting_point_source" => δval == DELTA_GRID[1] ? "theta_r0 (initial)" : "warm-started from prior delta",
            "kappa_point_estimate" => KAPPA_POINT_EST,
            "gravity_seed" => GRAVITY_SEED,
            "gradient_method" => String(GRADIENT_METHOD),
            "use_var_scaling" => USE_VAR_SCALING,
            "scaling_power" => SCALING_POWER,
            "done" => true))
        # Same fix as the resume path above: warm-start the NEXT delta from the verified
        # gravity-feasible best point, never the raw (possibly infeasible) KNITRO endpoint.
        if r.best_θ !== nothing
            θcur = r.best_θ
        else
            @printf("  WARNING: no feasible point found for delta=%g -- falling back to the raw (possibly infeasible) endpoint for warm-starting the next delta\n", δval)
            θcur = r.θstar
        end
    end
end
let gt = seq_timing_snapshot()
    @printf("\n=== TOTAL across this process run: dest_inversion=%.1fs (n=%d, nonconverged=%d)  influence_fn=%.1fs  gravity_residual=%.1fs ===\n",
            gt.dest_inv_s, gt.n_dest_inversions, gt.n_dest_nonconverged, gt.influence_s, gt.gravity_resid_s)
end
println("PRODUCTION_RUN DONE  D=$D")
end # SKIP_BATCH_LOOP guard
