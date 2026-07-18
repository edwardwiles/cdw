# ============================================================================
# Continuation 5, Priority 1C: warmed profile of the smoothed value/gradient
# and resolution of the 3.147s-vs-live-logs contradiction flagged in the
# continuation prompt's fact #6.
#
# Root cause, confirmed below by direct measurement (not assumed): the
# archived 3.147s number (run_smoothed_homotopy.jl's own "GRADIENT BENCHMARK"
# section, `t1 = @elapsed (g1 = ForwardDiff.gradient(ww -> ..., w_bench))`)
# times the FIRST-EVER call of a closure literal defined at that exact source
# location. In Julia, each `->` literal is its own compiled type; even though
# `smoothed_fixed_dual_L`/`ForwardDiff.gradient` were already warm from the
# homotopy stages' OWN closure (a different literal, run_smoothed_homotopy.jl
# line 192), the benchmark's closure (line 312) had never been called before
# and pays a full one-time JIT specialization cost on top of the real
# per-call work -- exactly what a "one-off" (N=1, no pre-warm) timing always
# risks. This script isolates that cost directly: an UNWARMED first call vs a
# WARMED steady-state median over N=30 reps, same closure object reused both
# times (no fresh literal per rep, so no confound from a NEW type each call).
# ============================================================================
include(joinpath(@__DIR__, "smoothed_consistent.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using ForwardDiff, Statistics, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_p1c_smoothed_warmed")
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

function x_free_from_w(w::AbstractVector)
    z = pivot_expand(w[2:end], pe)
    return vcat(w[1], vec(exp.(z)))
end

# ---- representative feasible point: the current canonical incumbent (upper_lfixcomposite_sr1_60s).
# Falls back to upper_maxit15_productfd_control (the homotopy's own start point) if the smoothed
# inner dual is infeasible there at the chosen rho, per the documented coarse-rho feasibility wall
# (fullA_smoothed_consistent_experiment.md sec "Genuine finding") -- checked directly below, not assumed.
const W_LFIXCOMPOSITE = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
const W_FALLBACK = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069,
    0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011,
    1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663,
    0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]

const RHO_LIST = [0.0054, 0.0027]

function try_point(w, rho)
    xf = x_free_from_w(w)
    tuner = rho_to_tuner(rho)
    obj_s = smoothed_obj_for(ctx, tuner)
    base = try
        solve_smoothed_base_state(xf, ctx, obj_s, tuner)
    catch e
        return (ok = false, obj_s = obj_s, base = nothing, xf = xf, tuner = tuner)
    end
    return (ok = base.inner_status in (0, -100, -101, -103), obj_s = obj_s, base = base, xf = xf, tuner = tuner)
end

println("="^78); println("Selecting a feasible (point, rho) pair for each rho in $RHO_LIST"); println("="^78)
points_for_rho = Dict{Float64, NamedTuple}()
for rho in RHO_LIST
    r1 = try_point(W_LFIXCOMPOSITE, rho)
    if r1.ok
        println("  rho=$rho: upper_lfixcomposite_sr1_60s is smoothed-feasible, inner_status=$(r1.base.inner_status)")
        points_for_rho[rho] = merge(r1, (label = "upper_lfixcomposite_sr1_60s", w = W_LFIXCOMPOSITE))
    else
        r2 = try_point(W_FALLBACK, rho)
        println("  rho=$rho: upper_lfixcomposite_sr1_60s INFEASIBLE under smoothing, falling back to upper_maxit15_productfd_control, ok=$(r2.ok)")
        @assert r2.ok "profile_p1c_smoothed_warmed: neither candidate point is smoothed-feasible at rho=$rho"
        points_for_rho[rho] = merge(r2, (label = "upper_maxit15_productfd_control", w = W_FALLBACK))
    end
end

# ============================================================================
# PART 1: the contradiction, isolated directly -- unwarmed first call of a
# FRESH closure literal vs warmed steady-state median of the SAME closure
# object, both measured in THIS process (excludes any cross-run noise).
# ============================================================================
println("\n" * "="^78); println("PART 1: unwarmed-first-call vs warmed-steady-state, same closure object"); println("="^78)
contradiction_rows = NamedTuple[]
for rho in RHO_LIST
    p = points_for_rho[rho]
    obj_s, base, xf = p.obj_s, p.base, p.xf
    w = p.w
    # Build the closure ONCE (mirrors run_smoothed_homotopy.jl's literal at line 312) -- reused for
    # every timing below so the closure-compilation cost is isolated to the FIRST call only.
    f = ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base)

    t_first = @elapsed (g_first = ForwardDiff.gradient(f, w))
    N = 30
    times = zeros(N)
    for i in 1:N
        times[i] = @elapsed ForwardDiff.gradient(f, w)
    end
    t_med = median(times)
    bytes_med = median([@allocated ForwardDiff.gradient(f, w) for _ in 1:N])

    @printf("  rho=%.4f (%s): UNWARMED first call = %.4fs.  WARMED median (N=%d, same closure) = %.4fs (%.1fx faster).  median bytes=%d\n",
        rho, p.label, t_first, N, t_med, t_first / t_med, bytes_med)
    push!(contradiction_rows, (rho = rho, point = p.label, unwarmed_first_call_s = t_first,
        warmed_median_s = t_med, speedup_unwarmed_to_warmed = t_first / t_med, warmed_median_bytes = bytes_med))
end
open(joinpath(OUTDIR, "contradiction_resolution.csv"), "w") do io
    println(io, "rho,point,unwarmed_first_call_s,warmed_median_s,speedup_unwarmed_to_warmed,warmed_median_bytes")
    for r in contradiction_rows
        println(io, r.rho, ",", r.point, ",", r.unwarmed_first_call_s, ",", r.warmed_median_s, ",", r.speedup_unwarmed_to_warmed, ",", r.warmed_median_bytes)
    end
end

# ============================================================================
# PART 2: full warmed component breakdown at each rho -- smoothed moment
# construction, smoothed CC inner solve (warm+cold), smoothed value callback,
# ForwardDiff scalar-envelope gradient, materialized-Jacobian contraction
# (diagnostic only), allocations, chunk count.
# ============================================================================
println("\n" * "="^78); println("PART 2: warmed component breakdown"); println("="^78)
component_rows = NamedTuple[]
for rho in RHO_LIST
    p = points_for_rho[rho]
    obj_s, base, xf, tuner = p.obj_s, p.base, p.xf, p.tuner
    w = p.w
    θf = CS.reconstruct_full(xf, ctx.m)
    Wn = size(obj_s.U, 1); d = obj_s.d

    # -- smoothed moment construction alone (K,G build at theta0) --
    K = zeros(Wn); G = zeros(Wn, d)
    smoothed_moments!(K, G, θf, obj_s.U, obj_s; tuner = tuner)   # warm-up / JIT
    t_moments = median([@elapsed smoothed_moments!(K, G, θf, obj_s.U, obj_s; tuner = tuner) for _ in 1:20])

    # -- smoothed CC inner solve, WARM (obj_s.x already at a converged nearby point after base solve) --
    t_inner_warm = median([@elapsed CS.inner_loop_internal(obj_s, θf) for _ in 1:20])

    # -- smoothed CC inner solve, COLD (force obj_s.x reset each rep) --
    function cold_inner_once()
        obj_s.x .= NaN
        CS.inner_loop_internal(obj_s, θf)
    end
    cold_inner_once()   # warm-up JIT
    t_inner_cold = median([@elapsed cold_inner_once() for _ in 1:20])
    CS.inner_loop_internal(obj_s, θf)   # restore obj_s.x to a converged warm state before continuing

    # -- smoothed value callback (smoothed_fixed_dual_L at base point) --
    smoothed_fixed_dual_L(xf, ctx, obj_s, base)   # warm-up
    t_value = median([@elapsed smoothed_fixed_dual_L(xf, ctx, obj_s, base) for _ in 1:30])

    # -- ForwardDiff scalar-envelope gradient, warmed (same closure reused) --
    f = ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base)
    ForwardDiff.gradient(f, w)   # warm-up
    grad_times = [@elapsed ForwardDiff.gradient(f, w) for _ in 1:30]
    t_grad = median(grad_times)
    grad_bytes = median([@allocated ForwardDiff.gradient(f, w) for _ in 1:10])
    chunk = ForwardDiff.pickchunksize(length(w))

    # -- materialized-Jacobian contraction (diagnostic only, per task) --
    function G_of_w(ww::AbstractVector)
        xfw = x_free_from_w(ww)
        θfw = CS.reconstruct_full(xfw, ctx.m)
        Kd = zeros(eltype(θfw), Wn); Gd = zeros(eltype(θfw), Wn, d)
        obj_s.moments!(Kd, Gd, θfw, obj_s.U, obj_s)
        return vec(Gd)
    end
    ForwardDiff.jacobian(G_of_w, w)   # warm-up
    t_jac = median([@elapsed ForwardDiff.jacobian(G_of_w, w) for _ in 1:10])
    jac_bytes = median([@allocated ForwardDiff.jacobian(G_of_w, w) for _ in 1:10])

    @printf("  rho=%.4f: moments=%.4fs  inner_warm=%.5fs  inner_cold=%.4fs  value=%.5fs  grad(warmed,N=30)=%.4fs (chunk=%d, %d bytes)  jac_diagnostic=%.4fs (%d bytes)\n",
        rho, t_moments, t_inner_warm, t_inner_cold, t_value, t_grad, chunk, grad_bytes, t_jac, jac_bytes)
    push!(component_rows, (rho = rho, point = p.label, t_moments_build = t_moments,
        t_inner_solve_warm = t_inner_warm, t_inner_solve_cold = t_inner_cold, t_value_callback = t_value,
        t_grad_warmed_median = t_grad, grad_bytes_median = grad_bytes, forwarddiff_chunksize = chunk,
        t_jac_materialize_diagnostic = t_jac, jac_bytes_median = jac_bytes))
end
open(joinpath(OUTDIR, "component_breakdown.csv"), "w") do io
    cols = keys(component_rows[1])
    println(io, join(cols, ","))
    for r in component_rows
        println(io, join((r[c] for c in cols), ","))
    end
end

println("\nWrote all Priority 1C artifacts to ", OUTDIR)
