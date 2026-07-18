# ============================================================================
# Task §18: short D=4 full-A solver run using OPTIMIZED-VALUE finite
# differences throughout, in the pivot-eliminated reduced coordinate system
# (task §10: gravity is exactly satisfied by construction, dropped as an
# explicit KNITRO constraint entirely -- only the divergence-budget
# inequality remains). Uses eval_fcga=no (task sec 16 finding: eval_fcga=yes
# silently downgrades the requested hessopt=4 to L-BFGS) so the outer solve
# genuinely gets the requested Hessian mode.
#
# Reduced coordinate w = (gamma'_focal, z_free[1:D^2-1]) in R^{D^2} (D=4: 16
# dims), z_free = log(Aod_theta) with the largest-|gravity-coefficient| entry
# eliminated (gravity_elimination.jl::PivotGravityElim). Delta(w) computed by
# evaluate_fullA's Delta_dual (fully re-solved inner CC dual at every
# gradient evaluation -- expensive but exact, per task's "optimized-value"
# method). Gradient: central FD, fixed h (not yet the full adaptive rule --
# task sec 13's adaptive_h_candidate is available but not wired in here,
# documented as a follow-up).
#
# Run: julia --project=. full_aod_diag/d4_exact/run_d4_optimized_fd.jl [upper|lower]
#
# Continuation 4, Phase 3 (live driver extension -- scaffold unchanged, only
# cb_G!'s gradient BODY and the KNITRO option loading gained new switches):
#   D4X_GRADIENT_METHOD in {delta_fd (default, ORIGINAL behavior, unchanged),
#     lfix_composite (always the cheap composite_gradient.jl gradient, no
#     refresh), hybrid (derivative_methods.jl::should_refresh-driven mix,
#     via composite_gradient.jl::HybridGradientPolicy)}.
#   D4X_HESSOPT in {auto (default, ORIGINAL fcga_no+maxit-based file,
#     unchanged), sr1, lbfgs, productfd} -- selects one of the new
#     csw_outer_wallclock_*.opt files (maxit=100000, maxtime_real=1e8 by
#     default in the FILE -- the actual wall-clock budget below overrides it
#     at runtime via KN_set_param_by_name, not by proliferating one opt file
#     per budget).
#   D4X_MAXTIME_REAL (seconds, optional): if set, overrides maxtime_real (and
#     bumps maxit to a large number) so wall-clock time is the genuinely
#     binding termination criterion -- required for Phase 4's wall-clock-
#     matched algorithm frontier.
#   D4X_REFRESH_EVERY / D4X_GAP_TOL: HybridGradientPolicy tuning (hybrid only).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
using KNITRO, Printf

const COMMIT = "9e03706"
const DIRECTION = length(ARGS) >= 1 ? ARGS[1] : "upper"
const GRADIENT_METHOD = Symbol(get(ENV, "D4X_GRADIENT_METHOD", "delta_fd"))   # :delta_fd | :lfix_composite | :hybrid
const HESSOPT_TAG = get(ENV, "D4X_HESSOPT", "auto")                          # "auto" | "sr1" | "lbfgs" | "productfd"
GRADIENT_METHOD in (:delta_fd, :lfix_composite, :hybrid) || error("D4X_GRADIENT_METHOD must be delta_fd|lfix_composite|hybrid, got $GRADIENT_METHOD")
const RUN_ID = "optfd_$(DIRECTION)_$(GRADIENT_METHOD)_$(HESSOPT_TAG)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const FIXED_H = 0.01   # per h_sweep.jl (results/fullA_d4/1bdb1cc/h_sweep.csv), well below h=0.1's demonstrated failure -- still used by the :delta_fd method and by :hybrid's expensive-refresh gradient (unchanged, for comparability with the historical control)
const MAXIT = parse(Int, get(ENV, "D4X_MAXIT", "15"))   # deliberately SHORT (15) by default per task sec 18 framing; ignored when D4X_MAXTIME_REAL is set (wall-clock becomes binding instead)
const OPT_FILE = HESSOPT_TAG == "auto" ? "csw_outer_fcga_no_maxit$(MAXIT).opt" : "csw_outer_wallclock_$(HESSOPT_TAG).opt"  # must exist -- see e.g. csw_outer_fcga_no_maxit15.opt / csw_outer_wallclock_sr1.opt
isfile(joinpath(@__DIR__, OPT_FILE)) || error("run_d4_optimized_fd.jl: OPT_FILE=$(OPT_FILE) does not exist -- KN_load_param_file prints an error but does NOT throw and silently continues with KNITRO defaults, which previously produced a misleading run (D4X_MAXIT=6 with no matching file ran to completion with wrong options and nobody noticed until the log was read); check explicitly rather than trusting KNITRO to fail loudly.")
const MAXTIME_REAL = haskey(ENV, "D4X_MAXTIME_REAL") ? parse(Float64, ENV["D4X_MAXTIME_REAL"]) : nothing
const REFRESH_EVERY = parse(Int, get(ENV, "D4X_REFRESH_EVERY", "5"))
const GAP_TOL = parse(Float64, get(ENV, "D4X_GAP_TOL", "0.05"))
const FIND_SMALLEST = DIRECTION == "upper"   # calibrated post-hoc against which gives larger kappa; see final printout

ctx = d4_exact_setup(find_smallest = FIND_SMALLEST,
    outer_loop_opt = joinpath(@__DIR__, OPT_FILE))  # eval_fcga=no (real BFGS)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

# ---- reduced <-> full-x_free maps ----
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0, zfree0)   # reduced coordinate, length D^2 (1 + (D^2-1))

function x_free_from_w(w::AbstractVector)
    gp = w[1]; zfree = w[2:end]
    z = pivot_expand(zfree, pe)
    Aod_theta = exp.(z)
    return vcat(gp, vec(Aod_theta))   # matches free_idx order: [gamma'_focal; Aod_theta column-major]
end

function Delta_of_w(w::AbstractVector)
    xf = x_free_from_w(w)
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
    return r.Delta_dual, r
end

# ---- bounds: gamma'_focal per theoretical bound; z_free generously wide (task sec 10's own
#      documented caveat: NOT a rigorous transformed-bound derivation) ----
w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

# ---- best-feasible incumbent tracking (task's explicit requirement: never trust the raw terminal iterate) ----
best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
callback_log = NamedTuple[]
n_eval = Ref(0)

function record!(w, Δ, r, kind)
    n_eval[] += 1
    feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
    if feasible && (best_feasible[] === nothing || (FIND_SMALLEST ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
        best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, gravity_resid = abs(r.gravity_value))
    end
    push!(callback_log, (idx = n_eval[], kind = kind, gp = w[1], Delta = Δ, feasible = feasible,
                          inner_status = r.inner_status, gravity_value = r.gravity_value))
end

function eval_F(w::Vector{Float64})
    Δ, r = Delta_of_w(w)
    record!(w, Δ, r, "F")
    return Δ, r
end

"""
    eval_grad_central_fd(w, h) -> Vector

Central FD, with a one-sided fallback when the inner solve fails at one side
of the probe (task sec 17's own guidance: an infeasible/failed inner solve
during a perturbation must not be silently ignored). Caught for real: the
first version of this function let a NaN from a failed inner solve at ONE
probe point propagate straight into the KNITRO Jacobian callback, which
KNITRO treats as a fatal "Evaluation error" (status -502) and aborts the
ENTIRE outer solve, not just that one gradient -- confirmed from
`ERROR: Jacobian element jac[4]... is undefined at the current point` in the
lower-direction run's knitro.log (results/fullA_d4/9e03706/optfd_lower_.../knitro.log),
reproducible identically across two independent attempts (same outer_iters=4
both times). Fix: fall back to a one-sided difference using whichever side is
finite; if BOTH sides fail, report 0.0 for that component and flag it in the
callback log (a crude but non-fatal fallback -- a genuinely adaptive h per
task sec 13 would be the correct long-term fix, not implemented here).
"""
function eval_grad_central_fd(w::Vector{Float64}, h::Float64 = FIXED_H)
    n = length(w)
    g = zeros(n)
    for i in 1:n
        wp = copy(w); wp[i] += h
        wm = copy(w); wm[i] -= h
        Δp, rp = Delta_of_w(wp); record!(wp, Δp, rp, "G+")
        Δm, rm = Delta_of_w(wm); record!(wm, Δm, rm, "G-")
        Δ0 = nothing
        if isfinite(Δp) && isfinite(Δm)
            g[i] = (Δp - Δm) / (2h)
        elseif isfinite(Δp)
            Δ0, r0 = Delta_of_w(w); record!(w, Δ0, r0, "G0")
            g[i] = (Δp - Δ0) / h
            println("  [grad fallback] coord $i: backward probe non-finite, using forward one-sided FD")
        elseif isfinite(Δm)
            Δ0, r0 = Delta_of_w(w); record!(w, Δ0, r0, "G0")
            g[i] = (Δ0 - Δm) / h
            println("  [grad fallback] coord $i: forward probe non-finite, using backward one-sided FD")
        else
            g[i] = 0.0
            println("  [grad fallback] coord $i: BOTH probes non-finite, reporting g[i]=0.0 (flagged, not a real gradient)")
        end
    end
    return g
end

# ---- gradient-method dispatch (Phase 3 extension) --------------------------
const hybrid_policy = HybridGradientPolicy(; refresh_every = REFRESH_EVERY, gap_tol = GAP_TOL)
const grad_log = NamedTuple[]
const n_inner_solves_by_gradcall = Int[]   # CS.INNER_SOLVE_COUNT[] diff per gradient callback

"""
    eval_grad_dispatch(w) -> Vector

Dispatches on GRADIENT_METHOD:
  :delta_fd        -- unchanged original behavior (2*D2 full inner re-solves)
  :lfix_composite  -- ALWAYS the cheap composite gradient (1 inner solve total,
                       to build the base+cache at w; the A-block/gamma pieces
                       add zero further inner solves)
  :hybrid          -- HybridGradientPolicy decides per call (composite_gradient.jl)
Every branch logs (gradient_source, reason, wall_seconds, n_inner_solves_consumed)
to `grad_log` for the Phase 4 cost comparison.
"""
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
    else   # :hybrid
        xf = x_free_from_w(w)
        g, meta = decide_gradient!(hybrid_policy, xf, ctx, pe, eval_grad_central_fd)
        push!(grad_log, (source = meta.gradient_source, reason = meta.refresh_reason, wall = time() - t0,
                          n_inner = CS.INNER_SOLVE_COUNT[] - n0))
        return g
    end
end

# ---- raw KNITRO NLP: minimize/maximize gamma'_focal s.t. Delta(w) <= delta ----
kc = KNITRO.KN_new()
KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, OPT_FILE))
if MAXTIME_REAL !== nothing
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", MAXTIME_REAL)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    println("wall-clock budget override: maxtime_real=$(MAXTIME_REAL)s, maxit=1e6 (wall time is the binding stop criterion)")
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

println(">>> Running D=4 short solve, direction=$DIRECTION (find_smallest=$FIND_SMALLEST), gradient_method=$GRADIENT_METHOD, hessopt_tag=$HESSOPT_TAG, maxit=$MAXIT, maxtime_real=$(MAXTIME_REAL===nothing ? "(file default)" : MAXTIME_REAL), h=$FIXED_H")
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
println("total Delta(w) evaluations (incl. FD gradient probes): ", n_eval[])

# ---- exact fresh feasibility recheck of BOTH the raw terminal point and the tracked best-feasible ----
# NOTE (bug found + fixed while reviewing the first two runs' output): the feasibility gate below
# MUST use max_abs_moment_kkt_resid (the LFD-WEIGHTED moment residual mean(m*.*G_j), which the inner
# CC dual actually drives to zero -- validated in test_inner_diagnostics.jl), NOT the raw UNWEIGHTED
# max_abs_moment_resid (mean(G_j) under the base measure) -- the latter is expected to be large away
# from the base point (that is the whole point of reweighting) and is NOT a feasibility criterion.
# An earlier version of this gate used max_abs_moment_resid and would have wrongly rejected genuinely
# feasible points.
function full_recheck(w, label)
    xf = x_free_from_w(w)
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)   # COLD re-solve
    κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
    println("[$label] gamma'_focal=$(r.gamma_focal_prime)  kappa=$κ  Delta_dual=$(r.Delta_dual)  Delta-delta=$(r.Delta_minus_delta)")
    println("         gravity_value=$(r.gravity_value)  max_abs_moment_kkt_resid=$(r.max_abs_moment_kkt_resid)  mean_m_resid=$(r.mean_m_resid)  inner_status=$(r.inner_status)")
    println("         w = ", w)   # full reduced-coordinate vector, for exact reproducibility
    return (label = label, gamma_focal_prime = r.gamma_focal_prime, kappa = κ, Delta_dual = r.Delta_dual,
            Delta_minus_delta = r.Delta_minus_delta, gravity_value = r.gravity_value,
            max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, mean_m_resid = r.mean_m_resid,
            inner_status = r.inner_status, w = collect(w))
end

println("\n" * "="^78); println("EXACT FRESH FEASIBILITY RECHECK"); println("="^78)
terminal_check = full_recheck(w_min, "raw_terminal")
best_check = best_feasible[] === nothing ? nothing : full_recheck(best_feasible[].w, "best_feasible_tracked")

status_label = if best_check !== nothing && best_check.max_abs_moment_kkt_resid < 1e-4 && best_check.mean_m_resid < 1e-4 && abs(best_check.gravity_value) < 1e-6 && best_check.Delta_minus_delta <= 1e-4
    "BEST_FEASIBLE_STALLED_OR_VERIFIED (external KKT stationarity NOT checked yet -- sec 22 not done this run)"
else
    "INFEASIBLE_OR_NUMERICALLY_UNRESOLVED"
end
println("\nSTATUS LABEL: ", status_label)

n_grad_calls = length(grad_log)
n_refresh = count(r -> r.source == :delta_fd || r.source == :expensive, grad_log)
n_cheap = n_grad_calls - n_refresh
total_grad_wall = sum(r.wall for r in grad_log; init = 0.0)
total_grad_inner_solves = sum(r.n_inner for r in grad_log; init = 0)
println("\ngradient calls: $n_grad_calls total ($n_refresh expensive/delta_fd, $n_cheap cheap)  ",
        "total gradient wall time=$(round(total_grad_wall,digits=2))s  total inner solves consumed by gradients=$total_grad_inner_solves")

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "direction=", DIRECTION, " find_smallest=", FIND_SMALLEST, " gradient_method=", GRADIENT_METHOD,
                " hessopt_tag=", HESSOPT_TAG, " maxit=", MAXIT, " maxtime_real=", MAXTIME_REAL, " h=", FIXED_H)
    println(io, "knitro_status=", nStatus, " opt_err=", opt_err[], " feas_err=", feas_err[],
                " outer_iters=", outer_iters[], " wall_seconds=", wall, " n_eval=", n_eval[])
    println(io, "gradient_calls=", n_grad_calls, " n_refresh_or_delta_fd=", n_refresh, " n_cheap=", n_cheap,
                " total_gradient_wall_seconds=", total_grad_wall, " total_inner_solves_in_gradients=", total_grad_inner_solves)
    println(io, "status_label=", status_label)
    println(io, "terminal: ", terminal_check)
    println(io, "best_feasible: ", best_check)
end
open(joinpath(OUTDIR, "callback_trace.csv"), "w") do io
    println(io, "idx,kind,gamma_focal_prime,Delta,feasible,inner_status,gravity_value")
    for r in callback_log
        println(io, r.idx, ",", r.kind, ",", r.gp, ",", r.Delta, ",", r.feasible, ",", r.inner_status, ",", r.gravity_value)
    end
end
open(joinpath(OUTDIR, "grad_log.csv"), "w") do io
    println(io, "call_idx,source,reason,wall_seconds,n_inner_solves")
    for (i, r) in enumerate(grad_log)
        println(io, i, ",", r.source, ",", r.reason, ",", r.wall, ",", r.n_inner)
    end
end
println("\nWrote ", OUTDIR)
