# ============================================================================
# Continuation 5, Priority 4: profile_Delta(g) = min_A Delta(g, A) over a
# coarse-to-fine grid of g=gamma'_focal, using the validated `lfix_composite`
# A-block gradient (h_mode=:adaptive, byte-identical to the original) for the
# per-grid-point local minimization, warm-started across the grid
# (continuation), with a periodic exact-hard Delta refresh (always -- every
# KNITRO objective callback here IS the exact hard Delta_dual, no smoothing
# anywhere in this driver).
#
# Formulation: for FIXED g, minimize Delta_dual(w) over the reduced A-block
# coordinates z_free (w[2:end], 15 dims at D=4) -- an UNCONSTRAINED NLP (no
# KNITRO constraint rows at all; box bounds only), objective = Delta_dual
# itself (not gamma), gradient = composite_gradient_at_fast's OWN A-block
# sub-vector (envelope-theorem-valid local gradient of Delta w.r.t. the
# A-block, per this investigation's established L_fix<->Delta_dual envelope
# equivalence at the base point -- reused, not re-derived).
#
# Shares the same two Priority 2 levers as run_d4_optimized_fd.jl's
# lfix_composite_fast: shared base-state between the objective and gradient
# callbacks (0 extra inner solves per gradient call), threaded A-block loop.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "live_defaults.jl"))   # enable_live_defaults! (pow cache / autarky-CF-v2 opt-ins; see docs/lowrisk_specialization_live_wiring.md)
using KNITRO, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const RUN_ID = "gamma_profile_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)

# ---- Continuation 8, workstream 4: opt-in low-risk moment-build specializations ----
# GP_ENABLE_POW_CACHE=1 (default): fixed-mu,sigma power cache, bit-identical, modest
#   allocation/GC win (see docs/fullA_pow_cache_wiring.md) -- on by default, no known
#   downside.
# GP_ENABLE_AUTARKY_CF_V2=0 (default): cached-base focal-autarky CF column
#   (docs/autarky_cf_v2_cached_base.md). OFF by default: this driver's per-eval cost is
#   a FULL evaluate_fullA call (full inner KNITRO dual solve over ALL D^2 moments, not
#   a CF-only column), so UsigmaPow is already materialized for the factual block
#   regardless -- v2's real win (skipping a from-raw-Usigma sigma-power) does not apply
#   here. See docs/lowrisk_specialization_live_wiring.md for the measured before/after.
# Both flags read once and passed straight through -- set at the process environment,
# not per-call, matching how this script is normally invoked (`ENV_VAR=... julia ...`).
const GP_ENABLE_POW_CACHE = get(ENV, "GP_ENABLE_POW_CACHE", "1") == "1"
const GP_ENABLE_AUTARKY_CF_V2 = get(ENV, "GP_ENABLE_AUTARKY_CF_V2", "0") == "1"
const GP_LIVE_DEFAULTS = enable_live_defaults!(ctx; pow_cache = GP_ENABLE_POW_CACHE, autarky_cf_v2 = GP_ENABLE_AUTARKY_CF_V2)
println("gamma_profile.jl live-defaults wiring: pow_cache=", GP_ENABLE_POW_CACHE, " autarky_cf_v2=", GP_ENABLE_AUTARKY_CF_V2)

pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
n = D2 - 1   # A-block-only reduced dimension (z_free)

x_free_from_w(w::AbstractVector) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

"""
    profile_delta_at_gamma(g, zfree_start, ctx, pe; maxtime_real, hessopt_tag) -> NamedTuple

Local minimization of Delta_dual(g, A) over the A-block (z_free) ONLY, g held
fixed. Returns the best-feasible-tracked (lowest Delta, not necessarily
<=delta -- this IS the profile function, not a feasibility search) point
found, plus the raw terminal iterate and full diagnostics.
"""
function profile_delta_at_gamma(g::Float64, zfree_start::Vector{Float64}, ctx, pe;
        maxtime_real::Float64 = 30.0, hessopt_tag::String = "sr1", threaded::Bool = true)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, fill(-8.0, n))
    KNITRO.KN_set_var_upbnds_all(kc, fill(8.0, n))
    KNITRO.KN_set_var_primal_init_values_all(kc, zfree_start)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0)
    trace = NamedTuple[]

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w)
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
        Δ = isfinite(r.Delta_dual) ? r.Delta_dual : 1e6
        evalResult.obj[1] = Δ
        n_eval[] += 1
        if r.inner_status in (0, -100, -101, -103)
            base = BaseDualState(xf, r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
            last_F_state[] = (w = copy(w), base = base)
            if best[] === nothing || Δ < best[].Delta_dual
                best[] = (zfree = copy(zfree), Delta_dual = Δ, gravity_value = r.gravity_value,
                          max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, inner_status = r.inner_status)
            end
        else
            last_F_state[] = nothing
        end
        push!(trace, (idx = n_eval[], Delta_dual = Δ, inner_status = r.inner_status))
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        # A base-state solve can fail (nStatus=-300, e.g. very near the known calibration-infeasibility
        # corner) -- solve_base_state throws in that case (mirrors evaluate_fullA's own inner_status
        # check, just via an exception instead of a sentinel). Letting that exception propagate into
        # KNITRO's C callback crashes the ENTIRE outer solve (confirmed directly: an uncaught error
        # here produces "User routine for grad_callback returned -500" and aborts, not just a bad
        # gradient for one probe) -- caught here, falls back to a zero gradient (KNITRO will reject
        # the step via its own line search, not crash), exactly the same non-fatal-fallback discipline
        # eval_grad_central_fd already uses for non-finite FD probes.
        local gfull
        try
            gfull, _ = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = threaded, h_mode = :adaptive)
        catch e
            println("  [gamma_profile grad fallback] base-state solve failed at g=$g, zfree=$zfree ($e) -- returning zero gradient")
            gfull = zeros(D2)
        end
        evalResult.objGrad .= gfull[2:end]
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)   # dense objective gradient over all n vars (default nV=-1), no constraints so no jacIndex* needed -- matches run_d4_optimized_fd.jl's own convention of omitting objGradIndexVars for a dense gradient

    t0 = time()
    nStatus = KNITRO.KN_solve(kc)
    wall = time() - t0
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best[]
    return (g = g, knitro_status = nStatus_code, wall = wall, n_eval = n_eval[],
            zfree_terminal = collect(xsol),
            best_zfree = b === nothing ? nothing : b.zfree,
            best_Delta = b === nothing ? NaN : b.Delta_dual,
            best_gravity = b === nothing ? NaN : b.gravity_value,
            best_kkt = b === nothing ? NaN : b.max_abs_moment_kkt_resid,
            trace = trace)
end

"""
    walk_grid(g_values, zfree_start, ctx, pe; maxtime_per_point) -> Vector{NamedTuple}

Runs profile_delta_at_gamma along `g_values` IN ORDER, warm-starting each point from the PREVIOUS
point's own best z_free (continuation) -- so the caller should order `g_values` so consecutive entries
are close together (e.g. walking outward from a known-feasible start), not an arbitrary grid order.
"""
function walk_grid(g_values::Vector{Float64}, zfree_start::Vector{Float64}, ctx, pe; maxtime_per_point::Float64)
    zf = zfree_start
    rows = NamedTuple[]
    for g in g_values
        res = profile_delta_at_gamma(g, zf, ctx, pe; maxtime_real = maxtime_per_point, hessopt_tag = "sr1")
        @printf("  g=%.6f: knitro_status=%d wall=%.1fs n_eval=%d best_Delta=%.6f (Delta-delta=%.3e) gravity=%.3e kkt=%.3e\n",
            g, res.knitro_status, res.wall, res.n_eval, res.best_Delta, res.best_Delta - ctx.δ, res.best_gravity, res.best_kkt)
        push!(rows, (g = g, knitro_status = res.knitro_status, wall = res.wall, n_eval = res.n_eval,
                      best_Delta = res.best_Delta, Delta_minus_delta = res.best_Delta - ctx.δ,
                      best_gravity = res.best_gravity, best_kkt = res.best_kkt))
        if res.best_zfree !== nothing
            zf = res.best_zfree   # continuation: warm-start the NEXT point from THIS point's best
        end
    end
    return rows
end

const RUN_GRID = get(ENV, "GP_RUN_GRID", "1") == "1"   # set GP_RUN_GRID=0 to only load the functions (e.g. for interactive testing via `include`)
if RUN_GRID
    # ---- coarse-to-fine grid over the theoretical gamma'_focal interval, warm-started FROM THE
    # KNOWN-FEASIBLE INCUMBENT outward in both directions (NOT from the calibration corner, which is
    # documented cold-infeasible at g close to 1 -- confirmed directly this session: a first attempt
    # starting at g=g_hi=1.0 with the calibration z_free hit exactly that infeasibility and required
    # the cb_G! try/catch fallback above to avoid a hard crash). ----
    g_lo, g_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    N_COARSE = parse(Int, get(ENV, "GP_N_COARSE", "9"))
    MAXTIME_PER_POINT = parse(Float64, get(ENV, "GP_MAXTIME_PER_POINT", "20.0"))

    const G_INCUMBENT = 0.8926359584642946   # upper_lfixcomposite_sr1_60s, Priority 0's canonical candidate
    const ZFREE_INCUMBENT = [0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]

    grid_up = collect(range(G_INCUMBENT, g_hi, length = ceil(Int, N_COARSE/2)+1))[2:end]     # walk UP from incumbent (exclusive) to g_hi
    grid_down = reverse(collect(range(g_lo, G_INCUMBENT, length = ceil(Int, N_COARSE/2)+1)))[2:end]  # walk DOWN from incumbent (exclusive) to g_lo
    println("theoretical gamma'_focal bounds: [$g_lo, $g_hi], incumbent g=$G_INCUMBENT")
    println("walking UP: ", grid_up)
    println("walking DOWN: ", grid_down)

    rows_incumbent = walk_grid([G_INCUMBENT], ZFREE_INCUMBENT, ctx, pe; maxtime_per_point = MAXTIME_PER_POINT)
    rows_up = walk_grid(grid_up, ZFREE_INCUMBENT, ctx, pe; maxtime_per_point = MAXTIME_PER_POINT)
    rows_down = walk_grid(grid_down, ZFREE_INCUMBENT, ctx, pe; maxtime_per_point = MAXTIME_PER_POINT)
    rows = vcat(rows_down |> reverse, rows_incumbent, rows_up)   # re-sort ascending in g for reporting

    open(joinpath(OUTDIR, "gamma_profile_coarse.csv"), "w") do io
        println(io, "g,knitro_status,wall,n_eval,best_Delta,Delta_minus_delta,best_gravity,best_kkt")
        for r in rows
            println(io, r.g, ",", r.knitro_status, ",", r.wall, ",", r.n_eval, ",", r.best_Delta, ",", r.Delta_minus_delta, ",", r.best_gravity, ",", r.best_kkt)
        end
    end
    println("\nWrote ", joinpath(OUTDIR, "gamma_profile_coarse.csv"))

    # ---- report the delta=1 crossing (feasibility boundary): the g interval where best_Delta<=delta ----
    feasible_g = [r.g for r in rows if isfinite(r.best_Delta) && r.best_Delta <= ctx.δ]
    if !isempty(feasible_g)
        println("Delta<=delta=", ctx.δ, " feasible for g in roughly [", minimum(feasible_g), ", ", maximum(feasible_g), "] on this COARSE grid")
        println("(kappa = 1 - g^(sigma/(sigma-1)); find_smallest wants the SMALLEST feasible g -- i.e. minimum(feasible_g) is the coarse-grid upper-direction candidate)")
    else
        println("NO coarse-grid point is Delta<=delta -- refine grid density or check bounds/start point")
    end
end
