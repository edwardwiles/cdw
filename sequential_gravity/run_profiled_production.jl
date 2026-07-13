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

const DVAL = parse(Int, get(ENV, "DVAL", "4"))
const OUTER_OPT_FILE = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "..", "full_aod_diag", "csw_outer_25.opt"))
const INNER_OPT_FILE = joinpath(@__DIR__, "..", "full_aod_diag", "ek_inner.opt")

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

params = (
    server=1, user=2, fakeData=1, DFake=DVAL, seedFakeData=889, counterType=1, counterExplicit=0,
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
@assert D == DVAL
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)

γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat; JacW = params.Jac_W
ρ = 2e-3; δ = params.δ_ref
Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
omitted = [d for d in 1:D if d != focal]; ref = 1
logτ = log.(τ); logw = log.(wHat)

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
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    val, x, nStatus = inner_loop(obj, θ)
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

function seq_gravcol(θ; δ::Real = δ, maxit = 20, tol = 5e-4, warm = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    (all(isfinite, uf) && all(isfinite, log_x)) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    # invert_all: shared serial/parallel driver for one full pass over the omitted destinations.
    # u_init_fn(d) supplies each destination's Newton warm start (nothing = cold). Also returns
    # `stats`, a length-D Vector{Any} (only `omitted` entries populated) holding each destination's
    # converged dest_stats result (DestInversion.stats), reused directly by influence_function
    # below instead of recomputing an identical O(S·D) pass at the same u.
    function invert_all(p_arg, u_init_fn::Function)
        um = zeros(D, D); um[:, focal] .= uf
        stats = Vector{Any}(undef, D)
        if PARALLEL_INVERSION
            Threads.@threads for i in eachindex(omitted)
                d = omitted[i]
                inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-8,
                                         maxit = 150, ls_iters = 50, u_init = u_init_fn(d))
                um[:, d] .= inv.u_full
                stats[d] = inv.stats
            end
        else
            for d in omitted
                inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-8,
                                         maxit = 150, ls_iters = 50, u_init = u_init_fn(d))
                um[:, d] .= inv.u_full
                stats[d] = inv.stats
                if verbose && !inv.converged
                    @printf("      dest %d NOT converged: iters=%d share_err=%.2e ‖u‖=%.2e\n",
                            d, inv.iterations, inv.max_abs_share_error, maximum(abs, inv.u_full))
                end
            end
        end
        um, stats
    end
    invert_omitted(p_arg; warm = nothing) = invert_all(p_arg, d -> warm === nothing ? nothing : warm[:, d])
    p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    if !ok
        verbose && println("    [seq] initial recover_lfd (focal-only) FAILED")
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    local umat, R, stats_cur
    try
        umat, stats_cur = invert_omitted(p; warm = warm)
        R = gravity_residual(umat, logτ, logw, σ).R_mean
    catch e
        verbose && println("    [seq] initial invert_omitted threw: ", e)
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    isfinite(R) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    verbose && @printf("    [seq] init: R0=%.4e\n", R)
    col = zeros(W); Rcol = 0.0
    for k in 1:maxit
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ,
                                  scale = :R_beta, precomputed_stats = stats_cur)
        col = infl.ψ_bar .+ infl.R_beta
        Rcol = infl.R_beta
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
        end
        p_cand, okc = recover_lfd(θ, moments_aug!, D + 2)
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
            local um_try, R_try, stats_try
            try
                if α == 1.0 || um_endpoint1 === nothing
                    um_try, stats_try = invert_all(p_try, d -> umat[:, d])
                    if um_endpoint1 === nothing
                        um_endpoint1 = um_try; stats_endpoint1 = stats_try
                    end
                else
                    um_try, stats_try = invert_all(p_try, d -> (1 - α) .* umat[:, d] .+ α .* um_endpoint1[:, d])
                end
                R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
            catch
                α *= 0.5; continue
            end
            if isfinite(R_try) && abs(R_try) < abs(R)
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
    gr = gravity_residual(umat, logτ, logw, σ)
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
    best_θ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_κ = Ref(find_smallest ? Inf : -Inf)
    best_warm = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Ktmp = zeros(1); Gtmp = zeros(1, D + 1)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rmean, Rcol, um, p, ok = seq_gravcol(θf; δ = δ, warm = warm[])
                lastRmean[] = Rmean; lastRcol[] = Rcol; lastok[] = ok
                if ok
                    gcol[] = c; warm[] = um; nfeas[] += 1
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
    d = D + 2; oci = d + 1
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
GP_POINT_EST = Kchk[1]
KAPPA_POINT_EST = 1 - GP_POINT_EST^(σ/(σ-1))
@printf("point estimate γ'_focal(F*) = %.6f  ->  kappa point estimate = %.6f\n", GP_POINT_EST, KAPPA_POINT_EST)

gp2kappa(gp) = 1 - gp^(σ/(σ-1))

function run_one_bound(name::Symbol, fs::Bool, δval::Real, θinit)
    @printf("\n----- %s bound: gamma'_focal %s, delta=%g (warm-started) -----\n", name, fs ? "MINIMIZED" : "MAXIMIZED", δval); flush(stdout)
    t0 = time()
    gp, θstar, st, bθ, b_gp, bwarm, cache = outer_solve_nested_cached(fs, θinit; use_exact_grad = true, δ = δval)
    κ = gp2kappa(gp)
    _, Rθ, _, _, _, okθ = seq_gravcol(θstar; δ = δval)
    CS.summarize(cache; label = "$name bound cache stats")
    wall = time() - t0
    @printf("  KNITRO:        gamma'_%s = %.6f -> kappa = %.6f  (status %d)  exact R_mean(θ*) = %.3e  gravity-feasible=%s  wall %.1fs\n",
            name, gp, κ, st, Rθ, okθ, wall)
    if bθ === nothing
        @printf("  best-feasible: NONE FOUND\n")
        return (κ = κ, gp = gp, R = Rθ, ok = okθ, best_κ = NaN, best_gp = NaN, best_θ = nothing, best_ok = false,
                θstar = θstar, cache = cache, nStatus = st, wall = wall)
    else
        _, Rb, _, _, _, okb = seq_gravcol(bθ; δ = δval, warm = bwarm)
        bκ = gp2kappa(b_gp)
        @printf("  best-feasible: gamma'_%s = %.6f -> kappa = %.6f  exact R_mean = %.3e  gravity-feasible=%s\n", name, b_gp, bκ, Rb, okb)
        return (κ = κ, gp = gp, R = Rθ, ok = okθ, best_κ = bκ, best_gp = b_gp, best_θ = bθ, best_ok = okb,
                θstar = θstar, cache = cache, nStatus = st, wall = wall)
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

@printf("\n=== [PRODUCTION: profiled/sequential, cached, free-only ForwardDiff] D=%d W=%d ρ=%g MU_FIXED(removed)=%s DELTA_GRID=%s ===\n",
        D, W, ρ, FREEZE_MU, DELTA_GRID)
@printf("point estimate kappa = %.6f\n", KAPPA_POINT_EST)

for (name, fs) in ((:lower, false), (:upper, true))
    BOUND_ARG in ("both", String(name)) || continue
    θcur = copy(θr0)   # warm-start chain within this bound direction only
    for δval in DELTA_GRID
        path = result_path(name, δval)
        existing = load_if_done(path)
        if existing !== nothing
            @printf("\n----- %s bound, delta=%g -- ALREADY DONE, skipping (resume) -----\n", name, δval)
            θcur = existing["theta_star"]
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
            "wall" => r.wall,
            "unique_free_x" => length(Set(rr.x_hash for rr in r.cache.trace)),
            "inner_solves" => r.cache.n_inner_solve, "grad_computations" => r.cache.n_grad_compute,
            "warm_started_inner" => r.cache.n_warm_started, "cold_inner" => r.cache.n_cold,
            "t_inner" => r.cache.t_inner, "t_grad" => r.cache.t_grad,
            "starting_point_source" => δval == DELTA_GRID[1] ? "theta_r0 (initial)" : "warm-started from prior delta",
            "kappa_point_estimate" => KAPPA_POINT_EST,
            "done" => true))
        θcur = r.θstar
    end
end
println("PRODUCTION_RUN DONE  D=$D")
