# ============================================================================
# Continuation 5, Priority 5c: D=6 pilot. Generic-D version of
# run_d4_optimized_fd.jl's lfix_composite_fast path, starting from a
# genuinely inner-feasible point (NOT the calibration point, which is
# cold-infeasible at D=6/8/10 -- confirmed directly this session, matching
# docs/fullA_block_local_performance.md sec 7's earlier D=6 finding): the
# "natural" Aod_theta values baked into ctx.θ0_up (the pre-gammanorm
# structural values), which give inner_status=0, Delta=0.0029 (feasible,
# lots of slack) -- found by direct query, not random search (60 random
# perturbation trials around calibration all failed first).
#
# Also serves as Priority 5b's D=6 half of the "redo D-scaling profile using
# the new incremental live gradient" -- reports live gradient-call wall time
# at D=6, comparable to the D=4 numbers in docs/fullA_p2_p3_fast_gradient_and_comparison.md.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using KNITRO, Printf, Dates

const D_PILOT = parse(Int, get(ENV, "D6_D", "6"))
const MAXTIME_REAL = parse(Float64, get(ENV, "D6_MAXTIME_REAL", "60.0"))
const COMMIT = strip(read(`git rev-parse --short HEAD`, String))

function run_pilot(direction::String; maxtime_real::Float64 = MAXTIME_REAL)
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
    r0 = evaluate_fullA(x_free_from_w(w0), ctx; cache = nothing, warm = false)
    println("D=$D start point: inner_status=$(r0.inner_status) Delta=$(r0.Delta_dual) gp0=$gp0")
    r0.inner_status in (0, -100, -101, -103) || error("run_d6_pilot: start point is NOT inner-feasible, cannot proceed")

    w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
    w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"))
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

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w)
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
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
            gfull, _ = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :adaptive)
        catch e
            println("  [D=$D pilot grad fallback] base-state/cache failure at gp=$(w[1]) ($e) -- returning zero A-block, exact gamma component only")
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
    println("D=$D $direction: knitro_status=$nStatus_code wall=$(round(wall,digits=1))s n_eval=$(n_eval[]) n_grad_calls=$(n_grad_calls[]) grad_wall_total=$(round(grad_wall_total[],digits=2))s")
    if b !== nothing
        κ = 1 - b.gp^(ctx.σ / (ctx.σ - 1))
        r_recheck = evaluate_fullA(x_free_from_w(b.w), ctx; cache = nothing, warm = false)
        println("  best_feasible: gp=$(b.gp) kappa=$κ Delta=$(b.Delta) Delta-delta=$(b.Delta - ctx.δ) gravity=$(b.gravity) kkt=$(b.kkt)")
        println("  COLD RECHECK: inner_status=$(r_recheck.inner_status) Delta=$(r_recheck.Delta_dual) (matches: $(abs(r_recheck.Delta_dual - b.Delta) < 1e-6))")
    else
        println("  NO feasible point found this run")
    end
    return (direction = direction, D = D, knitro_status = nStatus_code, wall = wall, n_eval = n_eval[],
            n_grad_calls = n_grad_calls[], grad_wall_total = grad_wall_total[],
            best_feasible = b, kappa = b === nothing ? NaN : 1 - b.gp^(ctx.σ / (ctx.σ - 1)))
end

if get(ENV, "D6_RUN", "1") == "1"
    OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "d$(D_PILOT)_pilot_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    mkpath(OUTDIR)
    res_upper = run_pilot("upper")
    res_lower = run_pilot("lower")
    open(joinpath(OUTDIR, "summary.txt"), "w") do io
        for r in (res_upper, res_lower)
            println(io, r.direction, ": status=", r.knitro_status, " wall=", r.wall, " n_eval=", r.n_eval,
                " n_grad_calls=", r.n_grad_calls, " grad_wall_total=", r.grad_wall_total,
                " kappa=", r.kappa, " best_feasible=", r.best_feasible)
        end
    end
    println("\nWrote ", joinpath(OUTDIR, "summary.txt"))
end
