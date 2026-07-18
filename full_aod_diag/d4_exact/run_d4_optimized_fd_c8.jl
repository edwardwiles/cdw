# ============================================================================
# Continuation 8, workstream B (algorithm frontier rerun). Copy+extend of
# run_d4_optimized_fd.jl (untouched, not edited in place -- per this
# workstream's own file-ownership discipline, avoiding conflicts with the two
# other continuation-8 workstreams running in parallel this session on the
# same base commit). Adds:
#
#   1. D4X_MOMENT_REP in {dense (default), compressed} -- threaded into every
#      evaluate_fullA_fast call (the F-eval / Delta_of_w path) via the
#      existing `moment_representation` kwarg (docs/compressed_live_integration_report.md).
#      Only affects gradient methods that go through evaluate_fullA_fast
#      (delta_fd, lfix_composite, lfix_composite_fast, hybrid); the NEW
#      smoothed_ad method never touches evaluate_fullA_fast at all (it has
#      its own, entirely separate, smoothed inner-dual pipeline -- see below).
#   2. D4X_GRADIENT_METHOD gains a 5th option, "smoothed_ad": the "genuinely
#      CONSISTENT smoothed AD" gradient from smoothed_consistent.jl
#      (docs/fullA_smoothed_consistent_experiment.md), reused verbatim (not
#      reinvented) -- ForwardDiff.gradient of `smoothed_fixed_dual_L` (the
#      envelope-theorem fixed-dual construction, "Method 1" in that doc, the
#      one actually used for its own outer solve) at a freshly-solved
#      smoothed inner-dual base, matched against a re-solved smoothed VALUE
#      via `smoothed_optimized_Delta` -- NOT a hard value with a smoothed
#      gradient bolted on. SIMPLIFICATION, documented not hidden: the
#      original experiment ran a 5-STAGE rho homotopy (run_smoothed_homotopy.jl);
#      this frontier needs a single wall-clock-budgeted KNITRO solve per
#      config per checkpoint (matching every other config here), so this
#      driver uses a SINGLE FIXED rho for the whole solve (picked once via
#      the exact same "coarsest empirically-feasible rho at w0" scan
#      run_smoothed_homotopy.jl already validated, then a decade finer),
#      not the full homotopy schedule. This is a deliberate, explicit
#      reinterpretation for wall-clock comparability -- flagged, not silent.
#
# INCLUDE-ORDER NOTE (load-bearing, found empirically this session): the
# compressed_live.jl/oracle_fast.jl chain and the smoothed_consistent.jl
# chain BOTH transitively re-`include` context.jl/oracle.jl/gravity_elimination.jl
# (each is designed be laoded standalone). Loading BOTH chains in the SAME
# Julia process re-executes `module CounterfactualSensitivity`'s own
# definition a second time, which leaves Main with two distinct module
# instances and produces an unrecoverable `UndefVarError`/export-ambiguity
# for any name each module exports (confirmed directly: reconstruct_full
# MethodError first, then a `PsiObjectiveBundleDelta` export-ambiguity
# error on the SECOND redefinition). Since this harness already runs each
# config as its own subprocess (run_phase4_frontier.jl's own "clean
# KNITRO/obj state" design, reused as-is by c8_frontier_run.jl), the fix is
# simply: never load both chains in the same process. GRADIENT_METHOD is
# read from ENV before any `include` happens, so this file loads EXACTLY
# ONE of the two chains per process, matching how it's actually invoked.
# ============================================================================
using Dates

const HERE = @__DIR__
const COMMIT = try strip(read(`git rev-parse --short HEAD`, String)) catch; "uncommitted" end
const DIRECTION = length(ARGS) >= 1 ? ARGS[1] : "upper"
const GRADIENT_METHOD = Symbol(get(ENV, "D4X_GRADIENT_METHOD", "delta_fd"))   # :delta_fd | :lfix_composite | :lfix_composite_fast | :hybrid | :smoothed_ad
const HESSOPT_TAG = get(ENV, "D4X_HESSOPT", "auto")                          # "auto" | "sr1" | "lbfgs" | "productfd"
const MOMENT_REP = Symbol(get(ENV, "D4X_MOMENT_REP", "dense"))               # :dense | :compressed
GRADIENT_METHOD in (:delta_fd, :lfix_composite, :lfix_composite_fast, :hybrid, :smoothed_ad) ||
    error("D4X_GRADIENT_METHOD must be delta_fd|lfix_composite|lfix_composite_fast|hybrid|smoothed_ad, got $GRADIENT_METHOD")
MOMENT_REP in (:dense, :compressed) || error("D4X_MOMENT_REP must be dense|compressed, got $MOMENT_REP")
(MOMENT_REP == :compressed && GRADIENT_METHOD == :smoothed_ad) &&
    error("D4X_MOMENT_REP=compressed is meaningless for smoothed_ad (that method never calls evaluate_fullA_fast) -- refusing rather than silently ignoring")
const SMOOTHED = GRADIENT_METHOD == :smoothed_ad

# ---- chain selection (see header note) ----
if SMOOTHED
    include(joinpath(HERE, "smoothed_consistent.jl"))   # pulls in context.jl, winners.jl, oracle.jl, gravity_elimination.jl
else
    include(joinpath(HERE, "context.jl"))
    include(joinpath(HERE, "winners.jl"))
    include(joinpath(HERE, "oracle.jl"))
    include(joinpath(HERE, "gravity_elimination.jl"))
    include(joinpath(HERE, "three_way_derivatives.jl"))
    include(joinpath(HERE, "lfix_incremental.jl"))
    include(joinpath(HERE, "compressed_moments.jl"))
    include(joinpath(HERE, "compressed_cc_inner.jl"))
    include(joinpath(HERE, "oracle_fast.jl"))
    include(joinpath(HERE, "compressed_live.jl"))
    include(joinpath(HERE, "composite_gradient_fast.jl"))   # includes composite_gradient.jl (+three_way_derivatives.jl/lfix_incremental.jl/derivative_methods.jl) and winner_certificate.jl (+instrumentation.jl/winners_v2.jl/winners.jl) -- all SAFE re-includes (no context.jl/oracle.jl/gravity_elimination.jl among them, so no module-redefinition risk per the header note)
end
using KNITRO, Printf, LinearAlgebra

const RUN_ID = "optfdc8_$(DIRECTION)_$(GRADIENT_METHOD)_$(HESSOPT_TAG)_$(MOMENT_REP)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const FIXED_H = 0.01
const MAXIT = parse(Int, get(ENV, "D4X_MAXIT", "15"))
const OPT_FILE = HESSOPT_TAG == "auto" ? "csw_outer_fcga_no_maxit$(MAXIT).opt" : "csw_outer_wallclock_$(HESSOPT_TAG).opt"
isfile(joinpath(HERE, OPT_FILE)) || error("run_d4_optimized_fd_c8.jl: OPT_FILE=$(OPT_FILE) does not exist")
const MAXTIME_REAL = haskey(ENV, "D4X_MAXTIME_REAL") ? parse(Float64, ENV["D4X_MAXTIME_REAL"]) : nothing
const REFRESH_EVERY = parse(Int, get(ENV, "D4X_REFRESH_EVERY", "5"))
const GAP_TOL = parse(Float64, get(ENV, "D4X_GAP_TOL", "0.05"))
const FIND_SMALLEST = DIRECTION == "upper"

ctx = d4_exact_setup(find_smallest = FIND_SMALLEST,
    outer_loop_opt = joinpath(HERE, OPT_FILE))
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

# ---- IDENTICAL common starting point across every config, exactly as run_d4_optimized_fd.jl/
#      run_phase4_frontier.jl already construct it (d4_exact_setup's own deterministic ctx.θ0_up). ----
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0, zfree0)

function x_free_from_w(w::AbstractVector)
    gp = w[1]; zfree = w[2:end]
    z = pivot_expand(zfree, pe)
    Aod_theta = exp.(z)
    return vcat(gp, vec(Aod_theta))
end

w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
callback_log = NamedTuple[]
n_eval = Ref(0)

function record!(w, Δ, inner_status, gravity_value_, kind)
    n_eval[] += 1
    feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
    if feasible && (best_feasible[] === nothing || (FIND_SMALLEST ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
        best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ)
    end
    push!(callback_log, (idx = n_eval[], kind = kind, gp = w[1], Delta = Δ, feasible = feasible,
                          inner_status = inner_status, gravity_value = gravity_value_))
end

# ============================================================================
# PATH A: dense/compressed lfix-family (configs 1-4). Byte-for-byte the same
# structure as run_d4_optimized_fd.jl, with evaluate_fullA swapped for
# evaluate_fullA_fast(...; moment_representation=MOMENT_REP) (validated
# equivalent to evaluate_fullA at :dense -- oracle_fast.jl's own header) so
# the SAME code path serves both dense and compressed configs.
# ============================================================================
if !SMOOTHED
    function Delta_of_w(w::AbstractVector)
        r, meta = evaluate_fullA_fast(x_free_from_w(w), ctx; cache = nothing, warm = true, moment_representation = MOMENT_REP)
        return r.Delta_dual, r
    end

    const last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)

    function eval_F(w::Vector{Float64})
        Δ, r = Delta_of_w(w)
        record!(w, Δ, r.inner_status, r.gravity_value, "F")
        if GRADIENT_METHOD == :lfix_composite_fast
            if r.inner_status in (0, -100, -101, -103)
                base = BaseDualState(x_free_from_w(w), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
                last_F_state[] = (w = copy(w), base = base)
            else
                last_F_state[] = nothing
            end
        end
        return Δ, r
    end

    function eval_grad_central_fd(w::Vector{Float64}, h::Float64 = FIXED_H)
        n = length(w)
        g = zeros(n)
        for i in 1:n
            wp = copy(w); wp[i] += h
            wm = copy(w); wm[i] -= h
            Δp, rp = Delta_of_w(wp); record!(wp, Δp, rp.inner_status, rp.gravity_value, "G+")
            Δm, rm = Delta_of_w(wm); record!(wm, Δm, rm.inner_status, rm.gravity_value, "G-")
            if isfinite(Δp) && isfinite(Δm)
                g[i] = (Δp - Δm) / (2h)
            elseif isfinite(Δp)
                Δ0, r0 = Delta_of_w(w); record!(w, Δ0, r0.inner_status, r0.gravity_value, "G0")
                g[i] = (Δp - Δ0) / h
                println("  [grad fallback] coord $i: backward probe non-finite, using forward one-sided FD")
            elseif isfinite(Δm)
                Δ0, r0 = Delta_of_w(w); record!(w, Δ0, r0.inner_status, r0.gravity_value, "G0")
                g[i] = (Δ0 - Δm) / h
                println("  [grad fallback] coord $i: forward probe non-finite, using backward one-sided FD")
            else
                g[i] = 0.0
                println("  [grad fallback] coord $i: BOTH probes non-finite, reporting g[i]=0.0")
            end
        end
        return g
    end

    const hybrid_policy = HybridGradientPolicy(; refresh_every = REFRESH_EVERY, gap_tol = GAP_TOL)
    const grad_log = NamedTuple[]

    function eval_grad_dispatch(w::Vector{Float64})
        n0 = CS.INNER_SOLVE_COUNT[]
        t0 = time()
        if GRADIENT_METHOD == :delta_fd
            g = eval_grad_central_fd(w)
            push!(grad_log, (source = :delta_fd, reason = :none, wall = time() - t0, n_inner = CS.INNER_SOLVE_COUNT[] - n0))
            return g
        elseif GRADIENT_METHOD == :lfix_composite
            xf = x_free_from_w(w)
            g, meta = composite_gradient_at(xf, ctx, pe)
            push!(grad_log, (source = :cheap, reason = :none, wall = time() - t0, n_inner = CS.INNER_SOLVE_COUNT[] - n0))
            return g
        elseif GRADIENT_METHOD == :lfix_composite_fast
            xf = x_free_from_w(w)
            shared = last_F_state[]
            base = (shared !== nothing && shared.w == w) ? shared.base : nothing
            g, meta = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :adaptive)
            push!(grad_log, (source = base === nothing ? :cheap_unshared : :cheap_shared, reason = :none,
                              wall = time() - t0, n_inner = CS.INNER_SOLVE_COUNT[] - n0))
            return g
        else   # :hybrid
            xf = x_free_from_w(w)
            g, meta = decide_gradient!(hybrid_policy, xf, ctx, pe, eval_grad_central_fd)
            push!(grad_log, (source = meta.gradient_source, reason = meta.refresh_reason, wall = time() - t0,
                              n_inner = CS.INNER_SOLVE_COUNT[] - n0))
            return g
        end
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(HERE, OPT_FILE))
    if MAXTIME_REAL !== nothing
        KNITRO.KN_set_param_by_name(kc, "maxtime_real", MAXTIME_REAL)
        KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
        println("wall-clock budget override: maxtime_real=$(MAXTIME_REAL)s, maxit=1e6")
    end
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        Δ, r = eval_F(w)
        evalResult.obj[1] = FIND_SMALLEST ? w[1] : -w[1]
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = FIND_SMALLEST ? 1.0 : -1.0
        evalResult.jac .= eval_grad_dispatch(w)
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    println(">>> D=4 c8 frontier run, direction=$DIRECTION, gradient_method=$GRADIENT_METHOD, hessopt_tag=$HESSOPT_TAG, moment_rep=$MOMENT_REP, maxit=$MAXIT, maxtime_real=$(MAXTIME_REAL===nothing ? "(file default)" : MAXTIME_REAL)")
    flush(stdout)
    t0 = time()
    open(joinpath(OUTDIR, "knitro.log"), "w") do io
        redirect_stdout(io) do
            KNITRO.KN_solve(kc)
        end
    end
    wall = time() - t0
    nStatus, objv, w_min, lambda_ = KNITRO.KN_get_solution(kc)
    opt_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, opt_err)
    feas_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc, feas_err)
    outer_iters = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, outer_iters)
    KNITRO.KN_free(kc)

    println("KNITRO terminal: status=$nStatus  gamma'_focal=$(w_min[1])  opt_err=$(opt_err[])  feas_err=$(feas_err[])  outer_iters=$(outer_iters[])  wall=$(round(wall,digits=1))s")
    println("total Delta(w) evaluations: ", n_eval[])

    n_grad_calls = length(grad_log)
    n_refresh = count(r -> r.source == :delta_fd || r.source == :expensive, grad_log)
    n_cheap = n_grad_calls - n_refresh
    total_grad_wall = sum(r.wall for r in grad_log; init = 0.0)
    total_grad_inner_solves = sum(r.n_inner for r in grad_log; init = 0)
    compressed_fallback = MOMENT_REP == :compressed ? COMPRESSED_FALLBACK_COUNT[] : 0

    global W_MIN_RESULT = (w_min = collect(w_min), nStatus = nStatus, opt_err = opt_err[], feas_err = feas_err[],
                            outer_iters = outer_iters[], wall = wall, n_eval = n_eval[],
                            n_grad_calls = n_grad_calls, n_cheap = n_cheap, total_grad_wall = total_grad_wall,
                            total_grad_inner_solves = total_grad_inner_solves, compressed_fallback = compressed_fallback)
end

# ============================================================================
# PATH B: smoothed_ad (config 5). Single fixed-rho wall-clock-budgeted KNITRO
# solve using smoothed_consistent.jl's matched value+gradient construction
# (Method 1: ForwardDiff.gradient of smoothed_fixed_dual_L). Rho picked once
# via the SAME "coarsest empirically-feasible at w0" scan
# run_smoothed_homotopy.jl validated (docs/fullA_smoothed_consistent_experiment.md
# sec 4), then a decade finer -- reused methodology, not reinvented.
# ============================================================================
if SMOOTHED
    println(">>> D=4 c8 frontier run (SMOOTHED_AD path), direction=$DIRECTION, hessopt_tag=$HESSOPT_TAG, maxit=$MAXIT, maxtime_real=$(MAXTIME_REAL===nothing ? "(file default)" : MAXTIME_REAL)")

    θ_full0 = CS.reconstruct_full(x_free_from_w(w0), ctx.m)
    _, _, gap0 = compute_winners_fast(θ_full0, ctx)
    gap_finite = filter(isfinite, vec(gap0))
    qs = (0.75, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
    gapq = Dict(q => (isempty(gap_finite) ? NaN : Statistics.quantile(gap_finite, q)) for q in qs)
    candidate_coarse = [gapq[q] for q in qs]
    xf0 = x_free_from_w(w0)
    coarsest_feasible = nothing
    println("Rho feasibility scan at w0 (shared common start point):")
    for rho in candidate_coarse
        obj_probe = smoothed_obj_for(ctx, rho_to_tuner(rho))
        Δp, _, nsp = smoothed_optimized_Delta(xf0, ctx, obj_probe)
        inner_ok = nsp in (0, -100, -101, -103)
        println("  rho=$rho  inner_status=$nsp  inner_solve_ok=$inner_ok  Delta=$Δp")
        if inner_ok && coarsest_feasible === nothing
            global coarsest_feasible = rho
        end
    end
    RHO_FIXED = if haskey(ENV, "D4X_SMOOTHED_RHO")
        parse(Float64, ENV["D4X_SMOOTHED_RHO"])
    elseif coarsest_feasible !== nothing
        min(coarsest_feasible, minimum(gap_finite) / 10)
    else
        minimum(gap_finite) / 10
    end
    println("Chosen FIXED rho for this solve: ", RHO_FIXED, "  (tuner=", rho_to_tuner(RHO_FIXED), ")")
    tuner = rho_to_tuner(RHO_FIXED)
    obj_s = smoothed_obj_for(ctx, tuner)

    "Returns `nothing` on inner-solve failure (expected event, not fatal -- see smoothed_homotopy's own eval_F guard)."
    function base_at(w::Vector{Float64})
        xf = x_free_from_w(w)
        try
            return solve_smoothed_base_state(xf, ctx, obj_s, tuner)
        catch e
            e isa ErrorException || rethrow()
            return nothing
        end
    end

    function eval_F(w::Vector{Float64})
        xf = x_free_from_w(w)
        Δ, inner_x, nStatus = smoothed_optimized_Delta(xf, ctx, obj_s)
        record!(w, Δ, nStatus, NaN, "F")
        return Δ, nStatus
    end

    const grad_log = NamedTuple[]
    function eval_grad_dispatch(w::Vector{Float64})
        t0 = time()
        base = base_at(w)
        if base === nothing
            println("  [grad fallback] smoothed base solve failed at current w -- probing small perturbations")
            for eps in (1e-3, 1e-2, 5e-2)
                base = base_at(w .+ eps)
                base !== nothing && break
            end
            if base === nothing
                push!(grad_log, (source = :smoothed_ad_fallback_zero, reason = :base_solve_failed, wall = time() - t0, n_inner = 0))
                return zeros(length(w))
            end
        end
        g = ForwardDiff.gradient(ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base), w)
        push!(grad_log, (source = :smoothed_ad, reason = :none, wall = time() - t0, n_inner = 1))
        return g
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(HERE, OPT_FILE))
    if MAXTIME_REAL !== nothing
        KNITRO.KN_set_param_by_name(kc, "maxtime_real", MAXTIME_REAL)
        KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
        println("wall-clock budget override: maxtime_real=$(MAXTIME_REAL)s, maxit=1e6")
    end
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        Δ, nStatus = eval_F(w)
        evalResult.obj[1] = FIND_SMALLEST ? w[1] : -w[1]
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = FIND_SMALLEST ? 1.0 : -1.0
        evalResult.jac .= eval_grad_dispatch(w)
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    t0 = time()
    open(joinpath(OUTDIR, "knitro.log"), "w") do io
        redirect_stdout(io) do
            KNITRO.KN_solve(kc)
        end
    end
    wall = time() - t0
    nStatus, objv, w_min, lambda_ = KNITRO.KN_get_solution(kc)
    opt_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, opt_err)
    feas_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc, feas_err)
    outer_iters = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, outer_iters)
    KNITRO.KN_free(kc)

    println("KNITRO terminal (smoothed problem): status=$nStatus  gamma'_focal=$(w_min[1])  opt_err=$(opt_err[])  feas_err=$(feas_err[])  outer_iters=$(outer_iters[])  wall=$(round(wall,digits=1))s")
    println("total smoothed Delta(w) evaluations: ", n_eval[])

    n_grad_calls = length(grad_log)
    total_grad_wall = sum(r.wall for r in grad_log; init = 0.0)
    total_grad_inner_solves = sum(r.n_inner for r in grad_log; init = 0)

    global W_MIN_RESULT = (w_min = collect(w_min), nStatus = nStatus, opt_err = opt_err[], feas_err = feas_err[],
                            outer_iters = outer_iters[], wall = wall, n_eval = n_eval[],
                            n_grad_calls = n_grad_calls, n_cheap = 0, total_grad_wall = total_grad_wall,
                            total_grad_inner_solves = total_grad_inner_solves, compressed_fallback = 0,
                            rho_fixed = RHO_FIXED)
end

# ============================================================================
# COMMON: exact-hard fresh recheck (task's explicit requirement -- EVERY
# config's candidate, including smoothed_ad's, is re-evaluated through the
# EXACT HARD oracle for the reported kappa, never a smoothed/compressed
# surrogate value). Also does the mandated "final cold DENSE recheck via
# evaluate_fullA_fast(...; moment_representation=:dense, warm=false)" as an
# independent second check, reported alongside evaluate_fullA's own cold
# recheck (should agree to the oracle_fast.jl-documented ~1e-11-or-tighter
# tolerance regardless of which mode/method produced the candidate).
# ============================================================================
function full_recheck(w, label)
    xf = x_free_from_w(w)
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)
    κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
    println("[$label] gamma'_focal=$(r.gamma_focal_prime)  kappa=$κ  Delta_dual=$(r.Delta_dual)  Delta-delta=$(r.Delta_minus_delta)")
    println("         gravity_value=$(r.gravity_value)  max_abs_moment_kkt_resid=$(r.max_abs_moment_kkt_resid)  mean_m_resid=$(r.mean_m_resid)  inner_status=$(r.inner_status)")
    return (label = label, gamma_focal_prime = r.gamma_focal_prime, kappa = κ, Delta_dual = r.Delta_dual,
            Delta_minus_delta = r.Delta_minus_delta, gravity_value = r.gravity_value,
            max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, mean_m_resid = r.mean_m_resid,
            inner_status = r.inner_status, w = collect(w))
end

"""
    external_stationarity_check_inline(w; h=0.01) -> NamedTuple

Inlined copy of stationarity_check.jl::external_stationarity_check's exact
methodology (NOT re-derived) -- copied rather than `include`d to avoid this
driver's own include-order constraints (stationarity_check.jl itself
re-`include`s context.jl/oracle.jl/gravity_elimination.jl, which is exactly
the double-include hazard documented at this file's top). Only reused here
in the PATH A (non-smoothed) case, where `evaluate_fullA` is unambiguously
available and un-redefined.
"""
function external_stationarity_check_inline(w::AbstractVector; h::Float64 = 0.01, bound_tol::Float64 = 1e-4)
    n = length(w)
    Delta_of_w_hard(ww) = evaluate_fullA(x_free_from_w(ww), ctx; cache = nothing, warm = true).Delta_dual
    Δ0 = Delta_of_w_hard(w)
    grad_f = zeros(n); grad_f[1] = FIND_SMALLEST ? 1.0 : -1.0
    grad_Delta = zeros(n)
    n_nonfinite = 0
    for i in 1:n
        wp = copy(w); wp[i] += h; wm = copy(w); wm[i] -= h
        Δp = Delta_of_w_hard(wp); Δm = Delta_of_w_hard(wm)
        if isfinite(Δp) && isfinite(Δm)
            grad_Delta[i] = (Δp - Δm) / (2h)
        else
            n_nonfinite += 1
            grad_Delta[i] = 0.0
        end
    end
    active_lo = w .- w_lo .< bound_tol
    active_hi = w_hi .- w .< bound_tol
    n_active_bounds = count(active_lo) + count(active_hi)
    eta = -dot(grad_f, grad_Delta) / max(dot(grad_Delta, grad_Delta), 1e-300)
    residual_vec = grad_f .+ eta .* grad_Delta
    residual_norm = norm(residual_vec)
    comp_slack = eta * (Δ0 - ctx.δ)
    return (Delta = Δ0, Delta_minus_delta = Δ0 - ctx.δ, eta = eta, eta_nonneg = eta >= -1e-8,
            residual_norm = residual_norm, residual_relative = residual_norm / max(norm(grad_f), 1e-12),
            complementary_slackness = comp_slack, n_active_bounds = n_active_bounds, n_nonfinite_probes = n_nonfinite)
end

println("\n" * "="^78); println("EXACT FRESH FEASIBILITY RECHECK (hard oracle, cold)"); println("="^78)
terminal_check = full_recheck(W_MIN_RESULT.w_min, "raw_terminal")
best_check = best_feasible[] === nothing ? nothing : full_recheck(best_feasible[].w, "best_feasible_tracked")

# mandated independent second check: cold DENSE evaluate_fullA_fast recheck of the SAME candidate.
# Only meaningful for the PATH A (non-smoothed) configs -- evaluate_fullA_fast/oracle_fast.jl is not
# loaded on the smoothed_ad path at all (see this file's header include-order note: loading BOTH
# chains in one process breaks module identity), and evaluate_fullA (oracle.jl, ALWAYS loaded, used
# by full_recheck above for every config including smoothed_ad) already IS the authoritative exact-hard
# check -- this second check exists specifically to validate the dense-vs-compressed equivalence claim,
# which is a non-issue for smoothed_ad (it never touches moment_representation). Explicitly N/A there,
# not silently skipped.
cold_dense_check = if !SMOOTHED && best_feasible[] !== nothing
    xf = x_free_from_w(best_feasible[].w)
    r2, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :dense)
    κ2 = 1 - r2.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
    println("[cold_dense_fast_recheck] kappa=$κ2  Delta_dual=$(r2.Delta_dual)  matches evaluate_fullA cold recheck to |Δκ|=$(abs(κ2 - best_check.kappa))")
    (kappa = κ2, Delta_dual = r2.Delta_dual, diff_vs_evaluate_fullA = abs(κ2 - best_check.kappa))
elseif SMOOTHED
    println("[cold_dense_fast_recheck] N/A for smoothed_ad (evaluate_fullA_fast not loaded on this path -- full_recheck's evaluate_fullA cold check above is the authoritative exact-hard check for this config)")
    nothing
else
    nothing
end

stationarity = (!SMOOTHED && best_feasible[] !== nothing) ? external_stationarity_check_inline(best_feasible[].w) : nothing
if stationarity !== nothing
    println("\nSTATIONARITY (external KKT check, hard oracle): eta=$(stationarity.eta) (nonneg=$(stationarity.eta_nonneg))  residual_norm=$(stationarity.residual_norm)  residual_relative=$(stationarity.residual_relative)  comp_slack=$(stationarity.complementary_slackness)  n_active_bounds=$(stationarity.n_active_bounds)  n_nonfinite_probes=$(stationarity.n_nonfinite_probes)")
end

status_label = if best_check !== nothing && best_check.max_abs_moment_kkt_resid < 1e-4 && best_check.mean_m_resid < 1e-4 && abs(best_check.gravity_value) < 1e-6 && best_check.Delta_minus_delta <= 1e-4
    "BEST_FEASIBLE_STALLED_OR_VERIFIED"
else
    "INFEASIBLE_OR_NUMERICALLY_UNRESOLVED"
end
println("\nSTATUS LABEL: ", status_label)
println("\ngradient calls: $(W_MIN_RESULT.n_grad_calls) total ($(W_MIN_RESULT.n_grad_calls - W_MIN_RESULT.n_cheap) expensive/delta_fd, $(W_MIN_RESULT.n_cheap) cheap)  total gradient wall time=$(round(W_MIN_RESULT.total_grad_wall,digits=2))s  total inner solves consumed by gradients=$(W_MIN_RESULT.total_grad_inner_solves)  compressed_fallback_count=$(W_MIN_RESULT.compressed_fallback)")

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "direction=", DIRECTION, " find_smallest=", FIND_SMALLEST, " gradient_method=", GRADIENT_METHOD,
                " hessopt_tag=", HESSOPT_TAG, " moment_representation=", MOMENT_REP, " maxit=", MAXIT,
                " maxtime_real=", MAXTIME_REAL, " h=", FIXED_H, SMOOTHED ? " rho_fixed=$(W_MIN_RESULT.rho_fixed)" : "")
    println(io, "knitro_status=", W_MIN_RESULT.nStatus, " opt_err=", W_MIN_RESULT.opt_err, " feas_err=", W_MIN_RESULT.feas_err,
                " outer_iters=", W_MIN_RESULT.outer_iters, " wall_seconds=", W_MIN_RESULT.wall, " n_eval=", W_MIN_RESULT.n_eval)
    println(io, "gradient_calls=", W_MIN_RESULT.n_grad_calls, " n_cheap=", W_MIN_RESULT.n_cheap,
                " total_gradient_wall_seconds=", W_MIN_RESULT.total_grad_wall,
                " total_inner_solves_in_gradients=", W_MIN_RESULT.total_grad_inner_solves,
                " compressed_fallback_count=", W_MIN_RESULT.compressed_fallback)
    println(io, "status_label=", status_label)
    println(io, "terminal: ", terminal_check)
    println(io, "best_feasible: ", best_check)
    println(io, "cold_dense_fast_recheck: ", cold_dense_check)
    println(io, "stationarity: ", stationarity)
end
open(joinpath(OUTDIR, "callback_trace.csv"), "w") do io
    println(io, "idx,kind,gamma_focal_prime,Delta,feasible,inner_status,gravity_value")
    for r in callback_log
        println(io, r.idx, ",", r.kind, ",", r.gp, ",", r.Delta, ",", r.feasible, ",", r.inner_status, ",", r.gravity_value)
    end
end
println("\nWrote ", OUTDIR)
