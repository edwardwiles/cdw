# ============================================================================
# Continuation 9, Phase 7: rerun the D=10 upper gate with the full production
# architecture, longer budget.
#
# Continuation 8's `run_d6_pilot.jl` (unmodified, reused as-is per this task's
# instruction) found D=10's upper direction budget-stalled at 120s (KNITRO
# status -401), using: the PLAIN dense oracle (`evaluate_fullA`) in the F
# callback, `h_mode=:adaptive` (bisection bandwidth search every call, no
# caching) in the G callback, and `csw_outer_wallclock_sr1.opt` only.
#
# This driver is a THIN wrapper around the exact same `run_d6_pilot.jl`
# machinery -- same context (`d_exact_setup_scaled`, W=8000), same
# `x_free_from_w`/pivot-reduction setup, same KNITRO wiring pattern -- with
# three production-architecture swaps, each independently validated earlier
# in Continuation 9 (this task does not re-derive them, only re-applies them
# at D=10, per the brief's "re-verify it also works and helps at D=10, don't
# just assume the D=20 number transfers" instruction -- see the sanity probe
# below):
#   1. F callback: `evaluate_fullA_fast(...; moment_representation=:compressed)`
#      instead of the plain `evaluate_fullA` (Phase 3.1).
#   2. G callback: `composite_gradient_at_fast(...; h_mode=:cached,
#      bandwidth_cache=policy.cache)` wrapped in a `BandwidthCachePolicy`
#      (Phase 5's recommended production lever), instead of `h_mode=:adaptive`
#      every call. `multi_method=:top3` (Phase 4's coordinate-specialized
#      update) and `validate_dense=false` (Phase 3.2's new default) are
#      already `composite_gradient_at_fast`'s own defaults -- not passed
#      explicitly, just confirmed present.
#   3. Both `csw_outer_wallclock_sr1.opt` (hessopt=3) and
#      `csw_outer_wallclock_lbfgs.opt` (hessopt=6) tried, per the task's
#      explicit "try both Hessian modes" instruction.
# The inner CC dual solve itself is UNCHANGED from Continuation 8 (still
# `ek_inner.opt`, dense-Hessian exact) -- Phase 3C's own conclusion was that
# this dense-Hessian compressed baseline beats all three alternatives it
# built, so it remains the default here, not re-litigated.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
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

const D_PILOT = parse(Int, get(ENV, "D10_D", "10"))
const MAXTIME_REAL = parse(Float64, get(ENV, "D10_MAXTIME_REAL", "600.0"))
const COMMIT = strip(read(`git rev-parse --short HEAD`, String))

function vmhwm_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end

# ----------------------------------------------------------------------------
# Safety: quick sanity probe, per this investigation's standing discipline,
# before the full 600s x 2 gate run. Cheap: one compressed value call + one
# cached-bandwidth gradient call at the D=10 natural-theta start point.
# ----------------------------------------------------------------------------
function sanity_probe()
    ctx = d_exact_setup_scaled(D = D_PILOT, W = 8000, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    z0 = log.(Aod_theta_natural)
    zfree0 = pivot_reduce(reshape(z0, D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0, zfree0)
    xf0 = x_free_from_w(w0)

    r_dense, _ = evaluate_fullA_fast(xf0, ctx; cache = nothing, warm = false, moment_representation = :dense)
    r_comp, _ = evaluate_fullA_fast(xf0, ctx; cache = nothing, warm = false, moment_representation = :compressed)
    @printf("[sanity] D=%d dense Delta=%.10f compressed Delta=%.10f diff=%.3e\n",
            D, r_dense.Delta_dual, r_comp.Delta_dual, abs(r_dense.Delta_dual - r_comp.Delta_dual))
    @assert abs(r_dense.Delta_dual - r_comp.Delta_dual) < 1e-8 "sanity_probe: dense/compressed Delta_dual mismatch"

    base = BaseDualState(collect(xf0), r_comp.θ_full, r_comp.zeta, r_comp.lambda, copy(ctx.obj.arg1), r_comp.inner_status)
    policy = BandwidthCachePolicy()
    maybe_invalidate!(policy, w0)
    g1, meta1 = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache)
    record_hits!(policy, meta1.cache_hits[2:end])
    @printf("[sanity] gradient call OK, |g|=%.4f cache_hits=%d/%d\n", norm(g1), count(meta1.cache_hits[2:end]), D2 - 1)
    @printf("[sanity] VmHWM = %.2f GB\n", vmhwm_kb() / 1e6)
    flush(stdout)
    return nothing
end

println("="^90); println("SANITY PROBE (memory + dense-vs-compressed equivalence, before the full gate run)"); println("="^90)
sanity_probe()

# ----------------------------------------------------------------------------
# The gate run itself: production-architecture version of run_d6_pilot.jl's
# run_pilot(), upper direction only (this task's scope), for a given opt file.
# ----------------------------------------------------------------------------
function run_pilot_prod(direction::String, opt_file::String, hess_label::String; maxtime_real::Float64 = MAXTIME_REAL)
    find_smallest = direction == "upper"
    ctx = d_exact_setup_scaled(D = D_PILOT, W = 8000, find_smallest = find_smallest)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    z0 = log.(Aod_theta_natural)
    zfree0 = pivot_reduce(reshape(z0, D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0, zfree0)
    r0, _ = evaluate_fullA_fast(x_free_from_w(w0), ctx; cache = nothing, warm = false, moment_representation = :compressed)
    println("D=$D [$hess_label] start point: inner_status=$(r0.inner_status) Delta=$(r0.Delta_dual) gp0=$gp0")
    r0.inner_status in (0, -100, -101, -103) || error("run_pilot_prod: start point is NOT inner-feasible, cannot proceed")

    w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
    w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, opt_file)
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
    n_eval = Ref(0)
    grad_wall_total = Ref(0.0)
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy()   # fresh per run, one per (direction, hess_label)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w)
        r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = true, moment_representation = :compressed)
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        n_eval[] += 1
        feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
        if r.inner_status in (0, -100, -101, -103)
            base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
            last_F_state[] = (w = copy(w), base = base)
        else
            last_F_state[] = nothing
        end
        if feasible && (best_feasible[] === nothing || (find_smallest ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, gravity = r.gravity_value, kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        t0 = time()
        local gfull
        try
            if base === nothing
                error("no cached base state at this w (F not called first at this exact point)")
            end
            invalidated, reason = maybe_invalidate!(policy, w)
            gfull, meta = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true,
                                                       h_mode = :cached, bandwidth_cache = policy.cache)
            # Bug found in this task's first run (D=10 upper, lbfgs mode, 2/241 grad calls): on a
            # TiedWinnerError, composite_gradient_at_fast falls back to full_rebuild_gradient_fallback
            # internally and returns a VALID gradient, but that fallback's meta NamedTuple has no
            # `cache_hits` field (composite_gradient_fast.jl line ~139: `merge(meta_fb, (tie_fallback
            # = true, tie_error = e))`, no cache_hits key) -- record_hits! blindly indexing
            # meta.cache_hits threw a FieldError, which the catch-all below then wrongly treated as a
            # gradient-computation failure and discarded the already-correct `gfull`, downgrading it to
            # an all-zero A-block. Fixed by only recording cache-hit bookkeeping on the non-tie-fallback
            # path -- the tie-fallback gradient itself was always fine, only this diagnostic bookkeeping
            # call was broken.
            meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        catch e
            println("  [D=$D prod pilot grad fallback] base-state/cache failure at gp=$(w[1]) ($e) -- returning zero A-block, exact gamma component only")
            gfull = zeros(D2)
        end
        grad_wall_total[] += time() - t0; n_grad_calls[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        evalResult.jac .= gfull
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    t0 = time()
    KNITRO.KN_solve(kc)
    wall = time() - t0
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    println("D=$D $direction [$hess_label]: knitro_status=$nStatus_code wall=$(round(wall,digits=1))s n_eval=$(n_eval[]) n_grad_calls=$(n_grad_calls[]) grad_wall_total=$(round(grad_wall_total[],digits=2))s policy_invalidations=$(policy.n_invalidations) cache_hits=$(policy.n_hits) cache_misses=$(policy.n_misses)")
    local r_recheck = nothing
    if b !== nothing
        κ = 1 - b.gp^(ctx.σ / (ctx.σ - 1))
        # Cold reference recheck via the TRUSTED plain dense oracle (evaluate_fullA, oracle.jl) --
        # matches this investigation's standing verification discipline: never trust the same fast/
        # compressed path that produced the candidate to also verify it.
        r_recheck = evaluate_fullA(x_free_from_w(b.w), ctx; cache = nothing, warm = false)
        println("  best_feasible: gp=$(b.gp) kappa=$κ Delta=$(b.Delta) Delta-delta=$(b.Delta - ctx.δ) gravity=$(b.gravity) kkt=$(b.kkt)")
        println("  COLD RECHECK (dense evaluate_fullA): inner_status=$(r_recheck.inner_status) Delta=$(r_recheck.Delta_dual) (matches: $(abs(r_recheck.Delta_dual - b.Delta) < 1e-6))")
    else
        println("  NO feasible point found this run")
    end
    flush(stdout)
    return (direction = direction, hess_label = hess_label, D = D, opt_file = opt_file, knitro_status = nStatus_code,
            wall = wall, n_eval = n_eval[], n_grad_calls = n_grad_calls[], grad_wall_total = grad_wall_total[],
            policy_invalidations = policy.n_invalidations, cache_hits = policy.n_hits, cache_misses = policy.n_misses,
            best_feasible = b, kappa = b === nothing ? NaN : 1 - b.gp^(ctx.σ / (ctx.σ - 1)),
            ctx = ctx, pe = pe, r_recheck = r_recheck)
end

# ----------------------------------------------------------------------------
# Lightweight A-directional sanity check (3-5 random directions), reusing the
# W80k doc's sec 1F / bandwidth doc's sec 3G methodology: perturb in the
# pivot-reduced z-space, h=0.02, 2 full warm-started dense evaluate_fullA
# re-solves per direction (bypassing the incremental machinery entirely),
# compare the re-solved secant of Delta_dual against the production
# gradient's directional derivative at the SAME point.
# ----------------------------------------------------------------------------
function directional_check(res; n_dir::Int = 5, h_dir::Float64 = 0.02, seed::Int = 31415)
    b = res.best_feasible
    b === nothing && (println("  [directional_check] no feasible point, skipping"); return NamedTuple[])
    ctx = res.ctx; pe = res.pe; D = ctx.D
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    xf0 = x_free_from_w(b.w)

    base0 = BaseDualState(xf0, res.r_recheck.θ_full, res.r_recheck.zeta, res.r_recheck.lambda, copy(ctx.obj.arg1), res.r_recheck.inner_status)
    policy = BandwidthCachePolicy()
    maybe_invalidate!(policy, b.w)
    g0, meta0 = composite_gradient_at_fast(xf0, ctx, pe; base = base0, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache)

    Random.seed!(seed)
    rows = NamedTuple[]
    for i in 1:n_dir
        dir = randn(length(b.w) - 1); dir ./= norm(dir)
        wp = copy(b.w); wp[2:end] .+= h_dir .* dir
        wm = copy(b.w); wm[2:end] .-= h_dir .* dir
        xfp = x_free_from_w(wp); xfm = x_free_from_w(wm)
        rp = evaluate_fullA(xfp, ctx; warm = true)
        rm = evaluate_fullA(xfm, ctx; warm = true)
        ok = rp.inner_status in (0, -100, -101, -103) && rm.inner_status in (0, -100, -101, -103)
        secant = ok ? (rp.Delta_dual - rm.Delta_dual) / (2h_dir) : NaN
        pred = dot(@view(g0[2:end]), dir)
        abserr = ok ? abs(pred - secant) : NaN
        @printf("  dir %d: status+=%d status-=%d secant=%s pred=%s abs_err=%s\n",
                i, rp.inner_status, rm.inner_status, string(secant), string(pred), string(abserr))
        push!(rows, (dir = i, status_plus = rp.inner_status, status_minus = rm.inner_status,
                      secant = secant, pred = pred, abs_err = abserr, finite_sane = ok && isfinite(secant)))
    end
    flush(stdout)
    return rows
end

# ----------------------------------------------------------------------------
# Main: upper direction, both Hessian modes.
# ----------------------------------------------------------------------------
OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase7_d10_upper_gate_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(OUTDIR)

configs = [
    ("sr1",   joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt")),
    ("lbfgs", joinpath(@__DIR__, "csw_outer_wallclock_lbfgs.opt")),
]

all_results = NamedTuple[]
for (label, opt_file) in configs
    println("\n", "="^90); println("D=$D_PILOT UPPER GATE, Hessian mode = $label, maxtime_real=$MAXTIME_REAL s"); println("="^90)
    flush(stdout)
    res = run_pilot_prod("upper", opt_file, label; maxtime_real = MAXTIME_REAL)
    println("\n  --- directional sanity check ($label) ---")
    diag_rows = directional_check(res; n_dir = 5)
    push!(all_results, merge(res, (diag = diag_rows,)))
    println("  VmHWM = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
    flush(stdout)
end

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    for r in all_results
        println(io, "hess=", r.hess_label, " direction=", r.direction, " status=", r.knitro_status,
            " wall=", r.wall, " n_eval=", r.n_eval, " n_grad_calls=", r.n_grad_calls,
            " grad_wall_total=", r.grad_wall_total, " cache_hits=", r.cache_hits, " cache_misses=", r.cache_misses,
            " kappa=", r.kappa, " best_feasible=", r.best_feasible)
        for row in r.diag
            println(io, "    diag: ", row)
        end
    end
end
println("\nWrote ", joinpath(OUTDIR, "summary.txt"))
println("FINAL VmHWM = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
