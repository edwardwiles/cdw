# ============================================================================
# Continuation 9 (user request, mid-Phase-8): fixed-A* baseline at REAL D=20/W=80000.
#
# Direct adaptation of c8_fixedA_pilot.jl (Continuation 8, Section 10's D=4/6/8/10
# sanity check) to the real-data D=20 context this session built. Same idea:
# A_od is held FIXED at the "natural theta" (calibration) values throughout --
# only gamma'_focal (1 free variable) is searched via KNITRO. This isolates how
# much of Phase 8's free-A pilot kappa improvement comes from genuinely
# exploring A-space vs. what a trivial 1-D gamma-only search at the SAME
# starting A would already find. Deliberately NOT reusing c8_fixedA_pilot.jl's
# D_PILOT/context_scaled.jl machinery (synthetic-only) -- this uses
# context_real_d20.jl's d20_real_setup instead, everything else unchanged.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using KNITRO, Printf, Dates

const W_PILOT = parse(Int, get(ENV, "FIXEDA_W", "80000"))
const MAXTIME_REAL = parse(Float64, get(ENV, "FIXEDA_MAXTIME_REAL", "120.0"))
const COMMIT = strip(read(`git rev-parse --short HEAD`, String))

function run_fixedA_pilot_d20(direction::String; maxtime_real::Float64 = MAXTIME_REAL)
    find_smallest = direction == "upper"
    ctx = d20_real_setup(W = W_PILOT, find_smallest = find_smallest)
    D = ctx.D; D2 = D^2

    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    gp0 = ctx.θ0_up[3+D]

    x_free_fixed_A(gp) = vcat(gp, Aod_theta_natural)

    r0 = evaluate_fullA(x_free_fixed_A(gp0), ctx; cache = nothing, warm = false)
    println("D=$D/W=$W_PILOT fixedA start point: inner_status=$(r0.inner_status) Delta=$(r0.Delta_dual) gp0=$gp0")
    r0.inner_status in (0, -100, -101, -103) || error("c9_fixedA_pilot_d20_real: start point is NOT inner-feasible, cannot proceed")

    w_lo = [ctx.bounds.γp_lo]
    w_hi = [ctx.bounds.γp_hi]

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, 1)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, [gp0])
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        gp = w[1]
        r = evaluate_fullA(x_free_fixed_A(gp), ctx; cache = nothing, warm = true)
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? gp : -gp
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        n_eval[] += 1
        feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
        if feasible && (best_feasible[] === nothing || (find_smallest ? gp < best_feasible[].gp : gp > best_feasible[].gp))
            best_feasible[] = (gp = gp, Delta = Δ, gravity = r.gravity_value, kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status)
        end
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)

    t0 = time()
    KNITRO.KN_solve(kc)
    wall = time() - t0
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    println("D=$D/W=$W_PILOT $direction FIXED-A: knitro_status=$nStatus_code wall=$(round(wall,digits=1))s n_eval=$(n_eval[])")
    if b !== nothing
        κ = 1 - b.gp^(ctx.σ / (ctx.σ - 1))
        r_recheck = evaluate_fullA(x_free_fixed_A(b.gp), ctx; cache = nothing, warm = false)
        println("  best_feasible: gp=$(b.gp) kappa=$κ Delta=$(b.Delta) Delta-delta=$(b.Delta - ctx.δ) gravity=$(b.gravity) kkt=$(b.kkt)")
        println("  COLD RECHECK: inner_status=$(r_recheck.inner_status) Delta=$(r_recheck.Delta_dual) (matches: $(abs(r_recheck.Delta_dual - b.Delta) < 1e-6))")
    else
        println("  NO feasible point found this run")
    end
    return (direction = direction, D = D, W = W_PILOT, knitro_status = nStatus_code, wall = wall, n_eval = n_eval[],
            best_feasible = b, kappa = b === nothing ? NaN : 1 - b.gp^(ctx.σ / (ctx.σ - 1)))
end

if get(ENV, "FIXEDA_RUN", "1") == "1"
    OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "d20real_fixedA_pilot_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    mkpath(OUTDIR)
    res_upper = run_fixedA_pilot_d20("upper")
    res_lower = run_fixedA_pilot_d20("lower")
    open(joinpath(OUTDIR, "summary.txt"), "w") do io
        for r in (res_upper, res_lower)
            println(io, r.direction, ": status=", r.knitro_status, " wall=", r.wall, " n_eval=", r.n_eval,
                " kappa=", r.kappa, " best_feasible=", r.best_feasible)
        end
    end
    println("\nWrote ", joinpath(OUTDIR, "summary.txt"))
end
