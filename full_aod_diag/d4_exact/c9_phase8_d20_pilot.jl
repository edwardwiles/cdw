# ============================================================================
# Continuation 9, Phase 8: short supervised D=20/W=80,000 pilot -- the FIRST
# full-A_od outer-loop optimization ever attempted at D=20 on real data.
#
# Two branches, each: (A) fixed-g profile minimization over the A-block only
# (unconstrained NLP, objective=Delta_dual, box bounds only -- same
# formulation as gamma_profile.jl's profile_delta_at_gamma, ported to D=20
# real data), then (B) a short joint (gamma', A) constrained polish (same
# formulation as c9_phase7_d10_upper_gate.jl's run_pilot_prod: maximize/
# minimize gamma' subject to Delta_dual<=delta), warm-started from (A)'s
# terminal A-block.
#
#   Branch 1 (pilot 1+3a): "upper branch" per docs/fullA_D20_W80k_microbenchmark.md
#     Point 3 -- g = gp0*1.01 = 0.997640, find_smallest=false (per task spec).
#   Branch 2 (pilot 2+3b): "lower branch" per that doc's Point 4 -- g =
#     gp0*0.99 = 0.977884, find_smallest=true (per task spec).
#
# Production architecture throughout (matching Phase 7's D=10 gate, PASSED):
# compressed moment representation (evaluate_fullA_fast(...;
# moment_representation=:compressed)), h_mode=:cached bandwidth policy
# (BandwidthCachePolicy), multi_method=:top3/validate_dense=false (composite_
# gradient_at_fast's own defaults), SR1 outer Hessian (csw_outer_wallclock_sr1.opt,
# hessopt=3) as default per Phase 7's recommendation. Dense-Hessian-exact inner
# CC dual solve unchanged (ek_inner.opt) -- Phase 3C's own D=20 finding was
# that this beats all three dense-Hessian-free alternatives.
#
# Safety: this script asserts (by grepping the actual source, not trusting a
# comment) that both the needs_outer_moment_jacobian=false memory-safety
# default and the bandwidth-cache ReentrantLock thread-safety fix are present
# in this worktree before doing anything else, then runs a cheap memory probe
# (VmHWM<5GB gate) before the first full-budget run -- same discipline as
# Phase 6/7.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using KNITRO, Printf, Dates, Random, Statistics
using LinearAlgebra: norm, dot

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const RUN_ID = "c9_phase8_d20_pilot_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase8_d20_pilot.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

# ---- Source-level safety checks (grep, not trust) ----
ctx_src = read(joinpath(@__DIR__, "context_real_d20.jl"), String)
@assert occursin("needs_outer_moment_jacobian::Bool = false", ctx_src) "SAFETY: needs_outer_moment_jacobian default is not false in context_real_d20.jl -- ABORTING"
cgf_src = read(joinpath(@__DIR__, "composite_gradient_fast.jl"), String)
@assert occursin("ReentrantLock", cgf_src) "SAFETY: ReentrantLock thread-safety fix not found in composite_gradient_fast.jl -- ABORTING"
logprint("Safety checks passed: needs_outer_moment_jacobian=false default present; ReentrantLock thread-safety fix present.")

function vmhwm_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end
vmhwm_gb() = vmhwm_kb() / 1e6

const FEASIBLE_CODES = (0, -100, -101, -103)
const PROFILE_MAXTIME = parse(Float64, get(ENV, "D20_PROFILE_MAXTIME", "900.0"))
const POLISH_MAXTIME  = parse(Float64, get(ENV, "D20_POLISH_MAXTIME", "450.0"))
const W_REAL = 80000

# ============================================================================
# PART 0: memory sanity probe (cheap) before any full-budget run
# ============================================================================
logprint("\n", "="^90); logprint("PART 0: memory sanity probe"); logprint("="^90)
t0 = time()
ctx_probe = d20_real_setup(W = W_REAL, find_smallest = true)
logprint("d20_real_setup probe wall=", round(time() - t0, digits = 1), "s  VmHWM=", round(vmhwm_gb(), digits = 2), " GB")
@assert vmhwm_gb() < 5.0 "SAFETY: VmHWM exceeded 5GB after context setup alone -- ABORTING before any optimization run"
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D; D2 = D^2
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0_natural = pivot_reduce(reshape(z0, D, D), pe_probe)
gp0 = ctx_probe.θ0_up[3+D]
logprint("D=", D, " gp0(calibration)=", gp0, " bounds=", ctx_probe.bounds)
r_dense, _ = evaluate_fullA_fast(x_free_from_w(vcat(gp0, zfree0_natural), pe_probe), ctx_probe; cache = nothing, warm = false, moment_representation = :dense)
r_comp, _ = evaluate_fullA_fast(x_free_from_w(vcat(gp0, zfree0_natural), pe_probe), ctx_probe; cache = nothing, warm = false, moment_representation = :compressed)
logprint("sanity: dense Delta=", r_dense.Delta_dual, " compressed Delta=", r_comp.Delta_dual, " diff=", abs(r_dense.Delta_dual - r_comp.Delta_dual))
@assert abs(r_dense.Delta_dual - r_comp.Delta_dual) < 1e-8
logprint("PART 0 VmHWM=", round(vmhwm_gb(), digits = 2), " GB")

# ============================================================================
# Branch offsets, per docs/fullA_D20_W80k_microbenchmark.md Point 3/Point 4.
# ============================================================================
const G_UPPER = gp0 * 1.01   # ~0.997640, Point 3
const G_LOWER = gp0 * 0.99   # ~0.977884, Point 4
logprint("G_UPPER=", G_UPPER, "  G_LOWER=", G_LOWER)

kappa_of(gp, σ) = 1 - gp^(σ / (σ - 1))

# ============================================================================
# PROFILE: fixed g, minimize Delta_dual over the A-block only (unconstrained
# NLP, box bounds only). Production architecture: compressed value,
# cached-bandwidth threaded gradient, SR1 outer Hessian.
# ============================================================================
function profile_minimize(label::String, g::Float64, find_smallest::Bool, zfree_start::Vector{Float64};
        maxtime_real::Float64 = PROFILE_MAXTIME, hessopt_tag::String = "sr1")
    ctx = d20_real_setup(W = W_REAL, find_smallest = find_smallest)
    pe = build_pivot_elimination(ctx)
    n = D2 - 1

    # Seed the compressed warm-start cache with a COLD (warm=false) solve before KNITRO ever
    # calls cb_F! with warm=true -- without this, the very first warm=true call in this fresh
    # ctx has no prior state to warm-start from, and was observed (this task, first attempt) to
    # return a bogus inner_status=-300 (unbounded) even though the SAME point is genuinely
    # feasible under a cold solve (confirmed by c9_phase8_findsmallest_probe.jl: inner_status=0,
    # Delta=0.2309 at this exact (g,A) combination for both find_smallest=true and false). That
    # bogus infeasibility produced Delta=1e6 + a zero-gradient fallback, which KNITRO then
    # wrongly reported as "Locally optimal solution found" at Iter 0. joint_polish already does
    # this seed (mirrors run_pilot_prod's r0 pre-check); profile_minimize was missing it.
    r_seed, _ = evaluate_fullA_fast(x_free_from_w(vcat(g, zfree_start), pe), ctx; cache = nothing, warm = false, moment_representation = :compressed)
    logprint("[", label, "] warm-cache seed (cold): inner_status=", r_seed.inner_status, " Delta=", r_seed.Delta_dual)
    r_seed.inner_status in FEASIBLE_CODES || error("profile_minimize($label): start point not inner-feasible even cold, cannot proceed")

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    # This profile formulation is bound-constrained-only (no general constraints). With
    # algorithm=auto, KNITRO picked Interior-Point/Barrier Direct, whose presolve nudges the start
    # point ("Knitro shifted start point further inside presolved bounds") and (as this task found)
    # can evaluate its very first trial point far from the given start (observed ||step||~160 in a
    # 399-dim box of half-width 8, i.e. essentially at the box corner) -- a large but LEGITIMATE
    # trial step for an interior-point method to consider, which happened to be genuinely
    # infeasible here. The REAL bug (fixed above/below in cb_F!/cb_G!) was this driver silently
    # accepting that failure as if it were a valid evaluation instead of rejecting it via a thrown
    # DomainError; with that fixed, either algorithm should behave correctly. Active-Set Conjugate
    # Gradient (3) is kept as the default here since it does not carry barrier/slack-variable
    # machinery for what is a plain box-constrained problem, but this is a preference, not a
    # required workaround, now that infeasible trial points are rejected properly.
    KNITRO.KN_set_param_by_name(kc, "algorithm", 3)
    xIndices = KNITRO.KN_add_vars(kc, n)
    # Root-cause fix (found live, this task, the source of EVERY earlier failure mode in this
    # driver): the box bounds [-8,8] were copied from the SYNTHETIC D=4/D=10 drivers
    # (gamma_profile.jl, c9_phase7_d10_upper_gate.jl), where natural-theta A_od is O(1) so z=log(A)
    # sits near 0 and [-8,8] is generous. The REAL D=20 economy's natural-theta A_od is NOT O(1)
    # (confirmed via c9_phase8_zfree_check.jl: Aod_theta_natural in [708, 4.86e11], so
    # z0=log(Aod_theta_natural) in [6.56, 26.9], pivot-reduced zfree0 has norm 315). The old [-8,8]
    # box therefore CLIPPED the given start point itself (KNITRO's honorbnds presolve silently
    # projects an out-of-bounds primal init onto the nearest bound) -- every symptom chased earlier
    # today (KNITRO evaluating at a box corner instead of the given x0, "Could not evaluate
    # objective... trying perturbed initial points", genuine -300 infeasibility even cold) was a
    # downstream consequence of this one bug, not independent problems. Fix: center a generous
    # (half-width 30, comfortably larger than zfree_start's own ~20-unit spread) box on the ACTUAL
    # start point instead of an assumed-near-zero one.
    z_halfwidth = 30.0
    KNITRO.KN_set_var_lobnds_all(kc, zfree_start .- z_halfwidth)
    KNITRO.KN_set_var_upbnds_all(kc, zfree_start .+ z_halfwidth)
    KNITRO.KN_set_var_primal_init_values_all(kc, zfree_start)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0); n_grad_calls = Ref(0)
    grad_wall_total = Ref(0.0)
    policy = BandwidthCachePolicy()
    trace = NamedTuple[]
    t_start = time()

    n_cold_retries = Ref(0)
    n_rejected = Ref(0)
    # Root-cause fix (found live, this task): KNITRO.jl's own callback wrapper
    # (`_try_catch_handler` in C_wrapper.jl) already catches ANY exception thrown inside an eval
    # callback and converts it to a proper KNITRO evaluation-error return code (KN_RC_EVAL_ERR for
    # DomainError specifically, KN_RC_CALLBACK_ERR otherwise) -- telling KNITRO "this point could
    # not be evaluated, reject it and backtrack/shrink the trust region," which is the CORRECT,
    # robust way to signal infeasibility. This driver's earlier approach (catch the failure
    # ourselves, substitute Δ=1e6 and/or a zero gradient, `return 0` as if successful) instead told
    # KNITRO the point WAS successfully evaluated with a huge objective and (via the paired
    # zero-gradient G fallback) zero gradient -- which trivially satisfies first-order optimality
    # and made KNITRO immediately declare bogus convergence at whatever point it happened to try
    # first (observed: Iter 0, "the initial point is a stationary point", 1-2 evals, objective
    # printed as exactly 1e6, "time spent in evaluations = 0.00000" -- all consistent with KNITRO
    # never receiving a real evaluation at all). Now: on infeasibility (after the warm-then-cold
    # retry below), THROW a DomainError and let it propagate through KNITRO.jl's own wrapper.
    n_dbg_calls = Ref(0)
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w, pe)
        r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
        if n_dbg_calls[] < 8
            n_dbg_calls[] += 1
            logprint("  [", label, " F DEBUG ", n_dbg_calls[], "] norm(zfree-zfree_start)=", norm(zfree - zfree_start),
                     " zfree[1:3]=", zfree[1:3], " warm_status=", r.inner_status, " warm_Delta=", r.Delta_dual)
        end
        # Robustness fix: a warm=true call can spuriously return a bad inner_status (observed:
        # -300 unbounded) at a point that is genuinely dual-feasible under a cold solve -- e.g.
        # right after KNITRO's barrier presolve nudges bounded variables slightly inside their box
        # bounds. Confirmed via c9_phase8_findsmallest_probe.jl: the same (g,A) combination that
        # produced -300/NaN here is inner_status=0/Delta=0.2309 under evaluate_fullA (cold).
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :compressed)
            if n_dbg_calls[] <= 8
                logprint("  [", label, " F DEBUG ", n_dbg_calls[], "] cold retry: status=", r.inner_status, " Delta=", r.Delta_dual)
            end
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            throw(DomainError(w[1], "profile_minimize($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting"))
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        if best[] === nothing || Δ < best[].Delta_dual
            best[] = (zfree = copy(zfree), Delta_dual = Δ, gravity_value = r.gravity_value,
                      max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                      t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, Delta_dual = Δ, inner_status = r.inner_status,
                       delta_feasible = Δ <= ctx.δ + 1e-6))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            logprint("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s Delta=", Δ, " status=", r.inner_status,
                     " cold_retries=", n_cold_retries[], " rejected=", n_rejected[])
        end
        return 0
    end
    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        # Found live (this task): exact `==` equality between F's cached w and G's incoming w can
        # legitimately FAIL by floating-point noise even when KNITRO is asking for the gradient at
        # "the same point" it just evaluated the objective at -- observed to crash the whole solve
        # with "User routine for grad_callback returned -500" once a hard `error()` replaced the old
        # silent zero-gradient fallback (worse: KNITRO treats a gradient-callback failure as FATAL,
        # not a rejectable trial point like cb_F!'s DomainError). Self-heal instead: if the cache
        # doesn't match, just re-derive a fresh base state at the CURRENT w via one extra warm solve
        # -- correct regardless of why the cache missed, and cheap relative to the gradient itself.
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            n_g_recompute[] += 1
            r_g, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
            if n_g_recompute[] <= 8
                logprint("  [", label, " G DEBUG ", n_g_recompute[], "] norm(zfree-zfree_start)=", norm(zfree - zfree_start),
                         " zfree[1:3]=", zfree[1:3], " shared_nothing=", shared === nothing, " warm_status=", r_g.inner_status, " warm_Delta=", r_g.Delta_dual)
            end
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :compressed)
                if n_g_recompute[] <= 8
                    logprint("  [", label, " G DEBUG ", n_g_recompute[], "] cold retry: status=", r_g.inner_status, " Delta=", r_g.Delta_dual)
                end
            end
            r_g.inner_status in FEASIBLE_CODES || throw(DomainError(w[1], "profile_minimize($label): cb_G! could not recompute a feasible base state"))
            base = BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        t0g = time()
        invalidated, reason = maybe_invalidate!(policy, w)
        gfull, meta = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true,
                                                   h_mode = :cached, bandwidth_cache = policy.cache)
        meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        grad_wall_total[] += time() - t0g; n_grad_calls[] += 1
        evalResult.objGrad .= gfull[2:end]
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best[]
    logprint("[", label, "] PROFILE DONE: status=", nStatus_code, " wall_ext=", round(wall_ext, digits = 1),
             "s n_eval=", n_eval[], " n_grad_calls=", n_grad_calls[], " grad_wall_total=", round(grad_wall_total[], digits = 1),
             "s cache_hits=", policy.n_hits, " cache_misses=", policy.n_misses, " invalidations=", policy.n_invalidations)
    if b !== nothing
        logprint("  best: Delta=", b.Delta_dual, " gravity=", b.gravity_value, " kkt=", b.max_abs_moment_kkt_resid,
                 " found_at_eval=", b.n_eval, " t=", round(b.t_elapsed, digits = 1), "s")
    end
    return (label = label, g = g, find_smallest = find_smallest, ctx = ctx, pe = pe,
            knitro_status = nStatus_code, wall_ext = wall_ext, n_eval = n_eval[], n_grad_calls = n_grad_calls[],
            grad_wall_total = grad_wall_total[], cache_hits = policy.n_hits, cache_misses = policy.n_misses,
            zfree_terminal = collect(xsol), best = b, trace = trace)
end

# ============================================================================
# POLISH: joint (gamma', A) constrained KNITRO solve, warm-started from the
# profile's terminal A-block, same convention as run_pilot_prod
# (c9_phase7_d10_upper_gate.jl): objective = find_smallest ? w[1] : -w[1],
# constraint Delta_dual<=delta.
# ============================================================================
function joint_polish(label::String, find_smallest::Bool, g_start::Float64, zfree_start::Vector{Float64};
        maxtime_real::Float64 = POLISH_MAXTIME, hessopt_tag::String = "sr1")
    ctx = d20_real_setup(W = W_REAL, find_smallest = find_smallest)
    pe = build_pivot_elimination(ctx)

    w0 = vcat(g_start, zfree_start)
    r0, _ = evaluate_fullA_fast(x_free_from_w(w0, pe), ctx; cache = nothing, warm = false, moment_representation = :compressed)
    logprint("[", label, "] polish start point: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual, " g_start=", g_start)
    r0.inner_status in FEASIBLE_CODES || error("joint_polish($label): start point not inner-feasible, cannot proceed")

    # Same box-bounds fix as profile_minimize -- see that function's comment for the full
    # root-cause explanation (real D=20 natural-theta z is NOT near zero; a fixed [-8,8] box
    # copied from the synthetic D=4/D=10 drivers clipped the given start point itself).
    z_halfwidth = 30.0
    w_lo = vcat(ctx.bounds.γp_lo, zfree_start .- z_halfwidth)
    w_hi = vcat(ctx.bounds.γp_hi, zfree_start .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0); n_grad_calls = Ref(0)
    grad_wall_total = Ref(0.0)
    policy = BandwidthCachePolicy()
    trace = NamedTuple[]
    t_start = time()

    n_cold_retries = Ref(0)
    n_rejected = Ref(0)
    # Root-cause fix (found live in profile_minimize, applied here too): a genuine inner-solve
    # evaluation FAILURE (inner_status not feasible even after the warm-then-cold retry) is
    # rejected via a thrown DomainError, letting KNITRO.jl's own callback wrapper convert it into a
    # proper KN_RC_EVAL_ERR and backtrack -- NOT a value this driver invents and hands to KNITRO as
    # if it were a real evaluation. This is DIFFERENT from the constraint being merely VIOLATED
    # (Δ finite but > δ): that is normal, expected territory for a constrained solve to explore and
    # is passed through to KNITRO honestly (real Δ, real gradient) so it can navigate back to
    # feasibility -- only a true evaluation failure gets rejected.
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :compressed)
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            throw(DomainError(w[1], "joint_polish($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting"))
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        feasible = Δ <= ctx.δ + 1e-6
        base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        if feasible && (best_feasible[] === nothing || (find_smallest ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, gravity = r.gravity_value,
                                kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                                t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = w[1], Delta_dual = Δ, inner_status = r.inner_status, feasible = feasible))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            logprint("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s gp=", w[1], " Delta=", Δ, " status=", r.inner_status,
                     " cold_retries=", n_cold_retries[], " rejected=", n_rejected[])
        end
        return 0
    end
    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        # Self-heal on cache miss instead of erroring/zero-fallback -- see profile_minimize's
        # cb_G! comment for the full explanation (exact `==` equality between F's cached w and G's
        # incoming w can legitimately fail by floating-point noise; a hard error here is FATAL to
        # the whole KNITRO solve, unlike cb_F!'s rejectable DomainError).
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            n_g_recompute[] += 1
            r_g, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = false, moment_representation = :compressed)
            end
            r_g.inner_status in FEASIBLE_CODES || throw(DomainError(w[1], "joint_polish($label): cb_G! could not recompute a feasible base state"))
            base = BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        t0g = time()
        invalidated, reason = maybe_invalidate!(policy, w)
        gfull, meta = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true,
                                                   h_mode = :cached, bandwidth_cache = policy.cache)
        meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        grad_wall_total[] += time() - t0g; n_grad_calls[] += 1
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
    σ = ctx.σ
    logprint("[", label, "] POLISH DONE: status=", nStatus_code, " wall_ext=", round(wall_ext, digits = 1),
             "s n_eval=", n_eval[], " n_grad_calls=", n_grad_calls[], " grad_wall_total=", round(grad_wall_total[], digits = 1), "s")
    local r_recheck = nothing
    κ = NaN
    if b !== nothing
        κ = kappa_of(b.gp, σ)
        r_recheck = evaluate_fullA(x_free_from_w(b.w, pe), ctx; cache = nothing, warm = false)
        logprint("  best_feasible: gp=", b.gp, " kappa=", κ, " Delta=", b.Delta, " Delta-delta=", b.Delta - ctx.δ,
                 " gravity=", b.gravity, " kkt=", b.kkt, " found_at_eval=", b.n_eval, " t=", round(b.t_elapsed, digits = 1), "s")
        logprint("  COLD RECHECK (dense evaluate_fullA): inner_status=", r_recheck.inner_status, " Delta=", r_recheck.Delta_dual,
                 " (matches: ", abs(r_recheck.Delta_dual - b.Delta) < 1e-6, ")")
    else
        logprint("  NO feasible point found this polish run")
    end
    return (label = label, find_smallest = find_smallest, ctx = ctx, pe = pe,
            knitro_status = nStatus_code, wall_ext = wall_ext, n_eval = n_eval[], n_grad_calls = n_grad_calls[],
            grad_wall_total = grad_wall_total[], cache_hits = policy.n_hits, cache_misses = policy.n_misses,
            best_feasible = b, kappa = κ, r_recheck = r_recheck, trace = trace)
end

# ============================================================================
# Directional secant diagnostics (5 random directions, lightweight -- per
# task's explicit "5, not 20" instruction, reusing W80k doc sec3F / Phase7
# directional_check pattern).
# ============================================================================
function directional_check(ctx, pe, w_point::Vector{Float64}, g0::Vector{Float64}; n_dir::Int = 5, h_dir::Float64 = 0.02, seed::Int = 271828)
    Random.seed!(seed)
    rows = NamedTuple[]
    for i in 1:n_dir
        dir = randn(length(w_point) - 1); dir ./= norm(dir)
        wp = copy(w_point); wp[2:end] .+= h_dir .* dir
        wm = copy(w_point); wm[2:end] .-= h_dir .* dir
        xfp = x_free_from_w(wp, pe); xfm = x_free_from_w(wm, pe)
        rp = evaluate_fullA(xfp, ctx; warm = true)
        rm = evaluate_fullA(xfm, ctx; warm = true)
        ok = rp.inner_status in FEASIBLE_CODES && rm.inner_status in FEASIBLE_CODES
        secant = ok ? (rp.Delta_dual - rm.Delta_dual) / (2h_dir) : NaN
        pred = dot(@view(g0[2:end]), dir)
        abserr = ok ? abs(pred - secant) : NaN
        logprint("  dir ", i, ": status+=", rp.inner_status, " status-=", rm.inner_status, " secant=", secant, " pred=", pred, " abs_err=", abserr)
        push!(rows, (dir = i, status_plus = rp.inner_status, status_minus = rm.inner_status,
                      secant = secant, pred = pred, abs_err = abserr, finite_sane = ok && isfinite(secant)))
    end
    return rows
end

function report_gravity(ctx, pe, w::Vector{Float64})
    xf = x_free_from_w(w, pe)
    r = evaluate_fullA(xf, ctx; warm = true)
    return r.gravity_value
end

# ============================================================================
# MAIN
# ============================================================================
all_summary = Dict{String,Any}()

logprint("\n", "="^90); logprint("BRANCH 1 (upper): PROFILE at g=", G_UPPER, " find_smallest=false"); logprint("="^90)
t_b1 = time()
profile1 = profile_minimize("upper_profile", G_UPPER, false, zfree0_natural; maxtime_real = PROFILE_MAXTIME)
logprint("VmHWM after profile1 = ", round(vmhwm_gb(), digits = 2), " GB")

logprint("\n", "="^90); logprint("BRANCH 1 (upper): POLISH warm-started from profile1"); logprint("="^90)
zfree1_seed = profile1.best !== nothing ? profile1.best.zfree : profile1.zfree_terminal
polish1 = joint_polish("upper_polish", false, G_UPPER, zfree1_seed; maxtime_real = POLISH_MAXTIME)
logprint("VmHWM after polish1 = ", round(vmhwm_gb(), digits = 2), " GB")
logprint("BRANCH 1 total wall = ", round(time() - t_b1, digits = 1), "s")

logprint("\n", "="^90); logprint("BRANCH 2 (lower): PROFILE at g=", G_LOWER, " find_smallest=true"); logprint("="^90)
t_b2 = time()
profile2 = profile_minimize("lower_profile", G_LOWER, true, zfree0_natural; maxtime_real = PROFILE_MAXTIME)
logprint("VmHWM after profile2 = ", round(vmhwm_gb(), digits = 2), " GB")

logprint("\n", "="^90); logprint("BRANCH 2 (lower): POLISH warm-started from profile2"); logprint("="^90)
zfree2_seed = profile2.best !== nothing ? profile2.best.zfree : profile2.zfree_terminal
polish2 = joint_polish("lower_polish", true, G_LOWER, zfree2_seed; maxtime_real = POLISH_MAXTIME)
logprint("VmHWM after polish2 = ", round(vmhwm_gb(), digits = 2), " GB")
logprint("BRANCH 2 total wall = ", round(time() - t_b2, digits = 1), "s")

# ---- Directional diagnostics at each of the 4 final points ----
logprint("\n", "="^90); logprint("DIRECTIONAL SECANT DIAGNOSTICS (5 dirs each, h=0.02)"); logprint("="^90)

function final_w_of_profile(res)
    b = res.best
    zfree = b !== nothing ? b.zfree : res.zfree_terminal
    return vcat(res.g, zfree)
end

diag_profile1 = let
    logprint("\n-- profile1 (upper, fixed-g) --")
    w = final_w_of_profile(profile1)
    xf = x_free_from_w(w, profile1.pe)
    g0, _ = composite_gradient_at_fast(xf, profile1.ctx, profile1.pe; threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    directional_check(profile1.ctx, profile1.pe, w, g0)
end

diag_profile2 = let
    logprint("\n-- profile2 (lower, fixed-g) --")
    w = final_w_of_profile(profile2)
    xf = x_free_from_w(w, profile2.pe)
    g0, _ = composite_gradient_at_fast(xf, profile2.ctx, profile2.pe; threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    directional_check(profile2.ctx, profile2.pe, w, g0)
end

diag_polish1 = let
    logprint("\n-- polish1 (upper, joint) --")
    if polish1.best_feasible !== nothing
        w = polish1.best_feasible.w
        xf = x_free_from_w(w, polish1.pe)
        g0, _ = composite_gradient_at_fast(xf, polish1.ctx, polish1.pe; threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        directional_check(polish1.ctx, polish1.pe, w, g0)
    else
        logprint("  no feasible point, skipping")
        NamedTuple[]
    end
end

diag_polish2 = let
    logprint("\n-- polish2 (lower, joint) --")
    if polish2.best_feasible !== nothing
        w = polish2.best_feasible.w
        xf = x_free_from_w(w, polish2.pe)
        g0, _ = composite_gradient_at_fast(xf, polish2.ctx, polish2.pe; threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        directional_check(polish2.ctx, polish2.pe, w, g0)
    else
        logprint("  no feasible point, skipping")
        NamedTuple[]
    end
end

# ---- Write CSV trajectories + summary ----
function write_csv_rows(path, rows)
    isempty(rows) && return
    keys_ = collect(propertynames(rows[1]))
    open(path, "w") do io
        println(io, join(string.(keys_), ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in keys_], ","))
        end
    end
end

write_csv_rows(joinpath(OUTDIR, "profile1_trace.csv"), profile1.trace)
write_csv_rows(joinpath(OUTDIR, "profile2_trace.csv"), profile2.trace)
write_csv_rows(joinpath(OUTDIR, "polish1_trace.csv"), polish1.trace)
write_csv_rows(joinpath(OUTDIR, "polish2_trace.csv"), polish2.trace)
write_csv_rows(joinpath(OUTDIR, "diag_profile1.csv"), diag_profile1)
write_csv_rows(joinpath(OUTDIR, "diag_profile2.csv"), diag_profile2)
write_csv_rows(joinpath(OUTDIR, "diag_polish1.csv"), diag_polish1)
write_csv_rows(joinpath(OUTDIR, "diag_polish2.csv"), diag_polish2)

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "=== profile1 (upper) ===")
    println(io, "status=", profile1.knitro_status, " wall_ext=", profile1.wall_ext, " n_eval=", profile1.n_eval,
        " n_grad_calls=", profile1.n_grad_calls, " grad_wall_total=", profile1.grad_wall_total,
        " cache_hits=", profile1.cache_hits, " cache_misses=", profile1.cache_misses, " best=", profile1.best)
    println(io, "=== polish1 (upper) ===")
    println(io, "status=", polish1.knitro_status, " wall_ext=", polish1.wall_ext, " n_eval=", polish1.n_eval,
        " n_grad_calls=", polish1.n_grad_calls, " grad_wall_total=", polish1.grad_wall_total,
        " kappa=", polish1.kappa, " best_feasible=", polish1.best_feasible)
    println(io, "=== profile2 (lower) ===")
    println(io, "status=", profile2.knitro_status, " wall_ext=", profile2.wall_ext, " n_eval=", profile2.n_eval,
        " n_grad_calls=", profile2.n_grad_calls, " grad_wall_total=", profile2.grad_wall_total,
        " cache_hits=", profile2.cache_hits, " cache_misses=", profile2.cache_misses, " best=", profile2.best)
    println(io, "=== polish2 (lower) ===")
    println(io, "status=", polish2.knitro_status, " wall_ext=", polish2.wall_ext, " n_eval=", polish2.n_eval,
        " n_grad_calls=", polish2.n_grad_calls, " grad_wall_total=", polish2.grad_wall_total,
        " kappa=", polish2.kappa, " best_feasible=", polish2.best_feasible)
end
logprint("\nWrote summary to ", joinpath(OUTDIR, "summary.txt"))
logprint("FINAL VmHWM = ", round(vmhwm_gb(), digits = 2), " GB")
logprint("OUTDIR = ", OUTDIR)
close(LOGIO)
