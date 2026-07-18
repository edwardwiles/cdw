# ============================================================================
# Continuation 8, workstream 4: controlled before/after measurement of
# enable_autarky_cf_v2! specifically in gamma_profile.jl's OWN repeated-
# fixed-draw-sweep access pattern.
#
# gamma_profile.jl's per-eval cost is a FULL evaluate_fullA call (all D^2
# moments, full inner KNITRO dual solve, all 15 free A_od entries -- including
# A_dd -- moving under a KNITRO local optimization at each fixed g), NOT a
# CF-only sweep from raw Usigma. A naive before/after wrapped around
# gamma_profile.jl's own maxtime_real-limited KNITRO local solves is confounded:
# tiny per-eval timing noise changes how many quasi-Newton iterations fit in the
# wall-clock budget before termination, so the two configs' outer trajectories
# can diverge (observed directly this session -- see
# docs/lowrisk_specialization_live_wiring.md) and n_eval differs, making total
# wall-clock not apples-to-apples.
#
# This script removes that confound: it captures ONE real x_free trajectory
# from an actual gamma_profile.jl-style local optimization (same
# profile_delta_at_gamma logic, same KNITRO settings) at a fixed g, then
# REPLAYS that exact, identical sequence of x_free points against two
# independently-built ctxs (pow_cache-only vs pow_cache+autarky_cf_v2) via
# direct evaluate_fullA(...; warm=true) calls in a tight timed loop -- so both
# configs see IDENTICAL inputs and the only thing that can differ is per-call
# cost, with @elapsed/@allocated evidence (not wall-clock alone).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "live_defaults.jl"))
using KNITRO, Printf, Statistics

ctx_capture = d4_exact_setup(find_smallest = true)
enable_live_defaults!(ctx_capture; pow_cache = true, autarky_cf_v2 = false)
pe = build_pivot_elimination(ctx_capture)
D = ctx_capture.D; D2 = D^2
n = D2 - 1
x_free_from_w(w::AbstractVector) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

const G_INCUMBENT = 0.8926359584642946
const ZFREE_INCUMBENT = [0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
const G_TARGET = 0.9284239723095297   # the g that produced a clean, non-time-limited 94-eval trajectory in the real sweep this session

println("Capturing a real gamma_profile.jl-style x_free trajectory at g=$G_TARGET ...")
xf_trace = Vector{Float64}[]
kc = KNITRO.KN_new()
KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"))
KNITRO.KN_set_param_by_name(kc, "maxtime_real", 20.0)   # generous -- we want the FULL natural trajectory, not a time-cut one
KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
xIndices = KNITRO.KN_add_vars(kc, n)
KNITRO.KN_set_var_lobnds_all(kc, fill(-8.0, n))
KNITRO.KN_set_var_upbnds_all(kc, fill(8.0, n))
KNITRO.KN_set_var_primal_init_values_all(kc, ZFREE_INCUMBENT)
last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
    zfree = evalRequest.x
    w = vcat(G_TARGET, zfree)
    xf = x_free_from_w(w)
    push!(xf_trace, copy(xf))
    r = evaluate_fullA(xf, ctx_capture; cache = nothing, warm = true)
    Δ = isfinite(r.Delta_dual) ? r.Delta_dual : 1e6
    evalResult.obj[1] = Δ
    if r.inner_status in (0, -100, -101, -103)
        base = BaseDualState(xf, r.θ_full, r.zeta, r.lambda, copy(ctx_capture.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
    else
        last_F_state[] = nothing
    end
    return 0
end
function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
    zfree = evalRequest.x
    w = vcat(G_TARGET, zfree)
    xf = x_free_from_w(w)
    shared = last_F_state[]
    base = (shared !== nothing && shared.w == w) ? shared.base : nothing
    local gfull
    try
        gfull, _ = composite_gradient_at_fast(xf, ctx_capture, pe; base = base, threaded = true, h_mode = :adaptive)
    catch e
        gfull = zeros(D2)
    end
    evalResult.objGrad .= gfull[2:end]
    return 0
end
cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
KNITRO.KN_set_cb_grad(kc, cb, cb_G!)
nStatus = KNITRO.KN_solve(kc)
KNITRO.KN_free(kc)
println("Captured trajectory: n_eval=", length(xf_trace), " knitro_status=", nStatus)

@assert length(xf_trace) >= 20 "trajectory too short to be a meaningful replay benchmark"

# ---- Replay the IDENTICAL xf_trace against two independent ctxs ----
function replay_bench(label::String; autarky_cf_v2::Bool)
    ctx = d4_exact_setup(find_smallest = true)
    info = enable_live_defaults!(ctx; pow_cache = true, autarky_cf_v2 = autarky_cf_v2)
    # warm-up (JIT) -- one call, not timed
    evaluate_fullA(xf_trace[1], ctx; cache = nothing, warm = true)

    n_rep = length(xf_trace)
    times = Vector{Float64}(undef, n_rep)
    allocs = Vector{Int}(undef, n_rep)
    gctimes = Vector{Float64}(undef, n_rep)
    for i in 1:n_rep
        stats = @timed evaluate_fullA(xf_trace[i], ctx; cache = nothing, warm = true)
        times[i] = stats.time
        allocs[i] = stats.bytes
        gctimes[i] = stats.gctime
    end
    total = sum(times)
    @printf("%-32s n=%d  total=%.3fs  median=%.4fs  mean=%.4fs  alloc/call=%.1f KB  gc/call=%.4fs\n",
        label, n_rep, total, median(times), mean(times), mean(allocs)/1024, mean(gctimes))
    return (label = label, n = n_rep, total = total, median = median(times), mean = mean(times),
             alloc_kb = mean(allocs)/1024, gc = mean(gctimes), info = info)
end

println("\nReplaying identical $(length(xf_trace))-point trajectory against two fresh ctxs (both warmed, N=1 pass each -- trajectory itself provides the repetition):")
resA = replay_bench("pow_cache only (v1, current default)"; autarky_cf_v2 = false)
resB = replay_bench("pow_cache + autarky_cf_v2"; autarky_cf_v2 = true)

speedup_total = resA.total / resB.total
speedup_median = resA.median / resB.median
@printf("\nRESULT: total-time speedup (v2 vs v1) = %.3fx   median-per-call speedup = %.3fx\n", speedup_total, speedup_median)
@printf("        alloc/call: v1=%.1f KB  v2=%.1f KB  (delta=%.1f KB)\n", resA.alloc_kb, resB.alloc_kb, resA.alloc_kb - resB.alloc_kb)
