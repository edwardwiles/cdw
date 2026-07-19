# ============================================================================
# Continuation 10, Phase 7: SHORT (maxit-capped, NOT full convergence) joint
# constrained continuation from the known upper candidate (gamma'=0.955701,
# delta=1, large-kappa branch), one run per selected scramble -- per task
# spec: "one SHORT continuation (a few dozen outer iterations at most)... do
# NOT launch a full outer-loop optimization for every scramble." Reuses
# exactly the same joint_polish formulation as c9_phase8_d20_pilot.jl
# (maximize/minimize gamma' subject to Delta_dual<=delta, warm-started from
# the candidate's own A-block), trimmed to this task's needs and capped at
# maxit=30 (not a wall-clock budget) so every replicate does the SAME amount
# of optimization work regardless of machine load.
#
# Replicates run (kept deliberately small -- this is the expensive half of
# the comparison): 1 pseudorandom baseline, 2 independent Halton scrambles,
# 1 Sobol replicate. Reports candidate kappa after the capped continuation,
# for a like-for-like comparison across draw types.
# ============================================================================
include(joinpath(@__DIR__, "qmc_context_real_d20.jl"))
include(joinpath(@__DIR__, "qmc_draws.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using KNITRO, Sobol, Random, Statistics, LinearAlgebra, Dates, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const RUN_ID = "c10_phase7_short_continuation_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_phase7_short_continuation.jl starting ", now(), " commit=", COMMIT, " nthreads=", Threads.nthreads())

const W_REAL = 80000
const FEASIBLE_CODES = (0, -100, -101, -103)
const MAXIT_CAP = 30   # "a few dozen outer iterations at most" -- capped by iteration COUNT, not wall time
kappa_of(gp, σ) = 1 - gp^(σ / (σ - 1))
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function load_w_csv(path)
    vals = Float64[]
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        push!(vals, parse(Float64, line))
    end
    return vals
end
w_upper = load_w_csv(joinpath(@__DIR__, "qmc_fixed_points", "upper_candidate_w.csv"))
gp_start = w_upper[1]; zfree_start = w_upper[2:end]
logprint("Warm-start from upper candidate: gp=", gp_start, " (find_smallest=true, corrected 'upper'/large-kappa convention)")

# ---- trimmed joint_polish: maxit-capped, not wall-clock-capped ----
function joint_polish_capped(label::String, ctx, pe; gp0::Float64, zfree0::Vector{Float64}, maxit::Int = MAXIT_CAP)
    D2 = ctx.D^2
    w0 = vcat(gp0, zfree0)
    r0, _ = evaluate_fullA_fast(x_free_from_w(w0, pe), ctx; cache = nothing, warm = false, moment_representation = :compressed)
    logprint("[", label, "] start: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)
    r0.inner_status in FEASIBLE_CODES || error("joint_polish_capped($label): start point not inner-feasible")

    z_halfwidth = 30.0
    w_lo = vcat(ctx.bounds.γp_lo, zfree0 .- z_halfwidth)
    w_hi = vcat(ctx.bounds.γp_hi, zfree0 .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"))
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit)
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", 1800.0)   # generous wall backstop; maxit is the real cap
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0); n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy()
    t_start = time()
    find_smallest = true   # matches this candidate's original (corrected-convention) direction

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
        if !(r.inner_status in FEASIBLE_CODES)
            r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :compressed)
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            throw(DomainError(w[1], "joint_polish_capped($label): infeasible point, rejecting"))
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        feasible = Δ <= ctx.δ + 1e-6
        base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        if feasible && (best_feasible[] === nothing || (find_smallest ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[])
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            r_g, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :compressed)
            end
            r_g.inner_status in FEASIBLE_CODES || throw(DomainError(w[1], "joint_polish_capped($label): cb_G! recompute failed"))
            base = BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        invalidated, reason = maybe_invalidate!(policy, w)
        gfull, meta = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache)
        meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        n_grad_calls[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        evalResult.jac .= gfull
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    κ = NaN
    if b !== nothing
        κ = kappa_of(b.gp, ctx.σ)
    end
    logprint("[", label, "] DONE: status=", nStatus_code, " wall=", round(wall_ext,digits=1), "s n_eval=", n_eval[],
             " n_grad_calls=", n_grad_calls[], " best_feasible_gp=", b === nothing ? NaN : b.gp, " kappa=", κ)
    return (label = label, knitro_status = nStatus_code, wall_ext = wall_ext, n_eval = n_eval[],
            n_grad_calls = n_grad_calls[], best_feasible = b, kappa = κ)
end

# ---- Run the capped continuation for a small set of replicates ----
replicates = [
    (:pseudorandom, 1, "pseudorandom_r1"),
    (:halton, 1, "halton_r1"),
    (:halton, 2, "halton_r2"),
    (:sobol, 1, "sobol_r1"),
]
draw_fn(kind::Symbol) = kind == :pseudorandom ? pseudorandom_U : kind == :halton ? halton_U : sobol_U

results_cont = NamedTuple[]
for (kind, s, label) in replicates
    seed = 1000 * s + (kind == :pseudorandom ? 1 : kind == :halton ? 2 : 3)
    logprint("\n", "="^90); logprint("REPLICATE: ", label, " seed=", seed); logprint("="^90)
    U = draw_fn(kind)(W_REAL, 20; seed = seed)
    ctx = d20_real_setup_qmc(W = W_REAL, U_injected = U, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    res = joint_polish_capped(label, ctx, pe; gp0 = gp_start, zfree0 = zfree_start, maxit = MAXIT_CAP)
    push!(results_cont, res)
end

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    for r in results_cont
        println(io, r.label, ": status=", r.knitro_status, " wall=", r.wall_ext, " n_eval=", r.n_eval,
                " kappa=", r.kappa, " best_feasible=", r.best_feasible)
    end
end
logprint("\nOUTDIR = ", OUTDIR)
close(LOGIO)
