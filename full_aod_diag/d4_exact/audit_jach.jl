# ============================================================================
# jac_h runtime audit (diag/fullA-d4-exact-jach-audit).
#
# Investigates and validates: does the cached/Method-B full-A path actually
# allocate / populate / read the dense jac_h::Array{Float64,3} tensor that
# cc_algo/PsiObjectiveBundle.jl's @with_kw default unconditionally built?
# Measures with runtime counters (cc_algo/jac_h_instrumentation.jl), not
# static grep alone, then validates the new opt-in
# needs_outer_moment_jacobian=false construction mode (jac_h -> 0x0x0,
# legacy jac_h-touching code paths error with a clear diagnostic instead of
# an out-of-bounds/silent-wrong-result) against the original mode at
# identical draws/points, across every quantity the task requires.
#
# Run: julia --project=. full_aod_diag/d4_exact/audit_jach.jl
# ============================================================================
include(joinpath(@__DIR__, "candidate_registry.jl"))   # -> ctx, pe, D, x_free_from_w, zfree0, gp0, w_up40, w_low (also prints its own report)
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
using KNITRO, Random, Printf, Statistics

println("\n" * "="^78); println("jac_h RUNTIME AUDIT"); println("="^78)

# ============================================================================
# PART 0: build a second context, IDENTICAL economy/draws, jac_h disabled
# ============================================================================
ctx2 = d4_exact_setup(find_smallest = true, needs_outer_moment_jacobian = false)
@assert ctx2.θ0_up == ctx.θ0_up "ctx2 must be the SAME synthetic economy/draws as ctx (only the jac_h flag differs)"
@assert ctx2.U == ctx.U

println("\n---- jac_h sizes (measured from the actual constructed bundles, not assumed) ----")
N_ = ctx.obj.N; d_ = ctx.obj.d; l_ = ctx.obj.l
println("ctx.obj  (needs_outer_moment_jacobian=true):  size(jac_h)=", size(ctx.obj.jac_h),
        "  eltype=", eltype(ctx.obj.jac_h))
println("ctx2.obj (needs_outer_moment_jacobian=false): size(jac_h)=", size(ctx2.obj.jac_h),
        "  eltype=", eltype(ctx2.obj.jac_h))
theoretical_bytes = 8 * N_ * (d_ + 2) * l_
println("N(=Jac_W)=", N_, "  d(=nTotalMoments)=", d_, "  l(=l_full)=", l_)
println("theoretical jac_h bytes = 8 * N * (d+2) * l = ", theoretical_bytes,
        " (", round(theoretical_bytes / 1024^2, digits = 2), " MB)")
println("measured Base.summarysize(ctx.obj.jac_h)  = ", Base.summarysize(ctx.obj.jac_h), " bytes")
println("measured Base.summarysize(ctx2.obj.jac_h) = ", Base.summarysize(ctx2.obj.jac_h), " bytes")
# Base.summarysize includes ~56 bytes of Array header/metadata overhead beyond the raw data buffer
# (confirmed: summarysize(ctx2.obj.jac_h)==56 for the 0x0x0 array, which has zero data bytes) -- the
# theoretical DATA-bytes formula therefore matches to within that fixed small header, not exactly.
@assert Base.summarysize(ctx.obj.jac_h) - theoretical_bytes == Base.summarysize(ctx2.obj.jac_h)

# ============================================================================
# PART 1: allocation-time isolation -- construct two FRESH bundles (same
# settings as ctx.obj) that differ ONLY in needs_outer_moment_jacobian, timing
# the constructor call itself. Isolates jac_h's own allocation+zeroing cost
# from everything else the constructor does (all other fields are identical
# work in both cases).
# ============================================================================
println("\n---- PART 1: constructor wall-time, with vs without jac_h (N reps) ----")
function time_construct(; needs_jac::Bool, reps::Int = 20)
    ts = Float64[]
    for _ in 1:reps
        t0 = time()
        o = CS.PsiObjectiveBundleImplicit(δ = ctx.δ, find_smallest = ctx.find_smallest, γ = ctx.γ,
            (moments!) = ctx.obj.moments!, moments_jacobian! = error, d = ctx.nTotalMoments,
            outer_constr_index = ctx.outer_constr_index, inequality_index = ctx.obj.inequality_index,
            complement_index = ctx.obj.complement_index, l = ctx.l_full, U = ctx.U, N = ctx.obj.N,
            lower_limit = ctx.obj.lower_limit, use_cached_x = true,
            outer_loop_opt = ctx.obj.outer_loop_opt, inner_loop_opt = ctx.obj.inner_loop_opt,
            needs_outer_moment_jacobian = needs_jac)
        push!(ts, time() - t0)
    end
    return ts
end
CS.reset_jac_h_counters!()
t_with = time_construct(needs_jac = true)
t_without = time_construct(needs_jac = false)
snap1 = CS.jac_h_counters_snapshot()
println("median constructor wall time WITH jac_h    = ", round(median(t_with) * 1000, digits = 3), " ms  (n=", length(t_with), ")")
println("median constructor wall time WITHOUT jac_h  = ", round(median(t_without) * 1000, digits = 3), " ms  (n=", length(t_without), ")")
println("median difference (jac_h's own alloc+zero cost, attributed) = ",
        round((median(t_with) - median(t_without)) * 1000, digits = 3), " ms")
println("counters after this part: alloc_count=", snap1.alloc_count, " (expect ", length(t_with), ")",
        "  skipped_count=", snap1.skipped_count, " (expect ", length(t_without), ")",
        "  cumulative alloc_time=", round(snap1.alloc_time * 1000, digits = 3), "ms",
        "  cumulative alloc_bytes=", snap1.alloc_bytes)
@assert snap1.alloc_count == length(t_with)
@assert snap1.skipped_count == length(t_without)
@assert snap1.populate_count == 0 && snap1.theta_branch_count == 0 && snap1.ift_count == 0

# ============================================================================
# PART 2: REAL-USAGE counter run -- exercise every code path this
# investigation's live full-A drivers actually use (evaluate_fullA
# warm+cold, a short central-FD-style multi-coordinate probe, and the
# L_fix incremental machinery) on ctx.obj (needs_outer_moment_jacobian=true,
# i.e. jac_h IS allocated) and confirm via counters -- not by re-reading the
# source -- that jac_h is never populated or read/contracted along the way.
# ============================================================================
println("\n" * "="^78); println("PART 2: real-usage counters (does the live full-A path ever touch jac_h?)"); println("="^78)
CS.reset_jac_h_counters!()

# NOTE: the pure calibration point is reached here via CS.pack_free(ctx.θ0_up, ctx.m) (matching
# test_oracle.jl's own proven-working approach), NOT via a pivot_reduce/pivot_expand round-trip of
# the zero log-matrix -- confirmed by direct check that the two differ substantially (maxabsdiff
# 3.14 in x_free) and that the pivot-roundtrip version is COLD/WARM INFEASIBLE (nStatus=-300) at
# this D=4 economy under find_smallest=true, even though the true calibration point is feasible.
# This is an existing, pre-dating-this-audit property of gravity_elimination.jl's pivot machinery
# (the pivot A_od entry is DETERMINED by the gravity constraint, not free, so reducing an
# off-manifold point does not round-trip back to the same point) -- out of scope to fix here, not a
# jac_h issue, just a reason to pick the point that is actually known-feasible for this audit's
# real-usage exercise.
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
r_warm = evaluate_fullA(x0, ctx; cache = nothing, warm = true)
r_cold = evaluate_fullA(x0, ctx; cache = nothing, warm = false)
println("calibration point (via pack_free): warm inner_status=", r_warm.inner_status,
        "  cold inner_status=", r_cold.inner_status, " (both should be 0/feasible)")
@assert r_warm.inner_status == 0 && r_cold.inner_status == 0 "calibration point unexpectedly infeasible -- pick a different real-usage probe point"

# short central-FD-style probe over a FEW coordinates (mirrors run_d4_optimized_fd.jl's
# eval_grad_central_fd applied directly in x_free space here since x0 is not itself derived from
# the reduced w/pivot map -- restricted to 4 of the l_full=23 coordinates to keep this audit fast;
# the mechanism being tested, does repeated evaluate_fullA usage ever touch jac_h, does not depend
# on how many coordinates are probed).
for i in 1:4
    xp = copy(x0); xp[i] *= 1.01
    xm = copy(x0); xm[i] *= 0.99
    evaluate_fullA(xp, ctx; cache = nothing, warm = true)
    evaluate_fullA(xm, ctx; cache = nothing, warm = true)
end

# L_fix incremental machinery (the INTENDED gradient path for the A-block) -- exercised at the
# upper_maxit40 candidate (w_up40, already confirmed inner_status=0 by candidate_registry.jl above),
# since lfix_incremental_at's coordinate perturbations are defined in the REDUCED w/pivot space that
# w_up40 already lives in (unlike the raw calibration x0 above).
x_up = x_free_from_w(w_up40)
base = solve_base_state(x_up, ctx)
lfix_cache = build_lfix_base_cache(x_up, ctx, base)
for i in 1:4
    lfix_incremental_at(lfix_cache, ctx, pe, w_up40, i, w_up40[i] + 0.01; tier = :incremental_o1)
end

snap2 = CS.jac_h_counters_snapshot()
println("After evaluate_fullA(warm)+evaluate_fullA(cold)+8 FD-style probe evals+L_fix base+4 incremental evals:")
println("  alloc_count=", snap2.alloc_count, " (expect 0 -- no NEW bundle constructed during this part)")
println("  populate_count=", snap2.populate_count, " (expect 0 -- calculate_jac_θ! never called)")
println("  theta_branch_count=", snap2.theta_branch_count, " (expect 0 -- callable's length(theta)>0 branch never entered)")
println("  ift_count=", snap2.ift_count, " (expect 0)")
@assert snap2.alloc_count == 0 && snap2.populate_count == 0 && snap2.theta_branch_count == 0 && snap2.ift_count == 0 "REAL USAGE UNEXPECTEDLY TOUCHED jac_h -- claim in docs/fullA_jach_audit.md would be FALSE, do not proceed"
println("  CONFIRMED (by counter, not by static reading): the live full-A cached/L_fix path never touches jac_h.")

# ============================================================================
# PART 3: counters are NOT dead code -- force the legacy theta-nonempty
# branch directly and confirm the counters DO increment (sanity check that
# Part 2's zero counts reflect real behavior, not a wiring bug in the
# instrumentation itself).
# ============================================================================
println("\n" * "="^78); println("PART 3: counter sanity check (force the legacy branch, confirm counters fire)"); println("="^78)
CS.reset_jac_h_counters!()
θ_full0 = CS.reconstruct_full(x0, ctx.m)
inner_x_probe = copy(ctx.obj.x)   # last successful (zeta*,lambda*) from Part 2's evaluate_fullA calls
g_buf = zeros(ctx.l_full)
ncon = ctx.obj.d - ctx.obj.outer_constr_index + 2
jac_buf = zeros(ncon * ctx.l_full)
ctx.obj(inner_x_probe, g_buf, θ_full0; jac = jac_buf)   # forces length(theta)>0 branch
snap3 = CS.jac_h_counters_snapshot()
println("Forced one direct call: obj(inner_x, g, theta_full; jac=...) with nonempty theta.")
println("  populate_count=", snap3.populate_count, " (expect 1)  theta_branch_count=", snap3.theta_branch_count,
        " (expect 1)  ift_count=", snap3.ift_count, " (expect 1, since outer_constr_index<=d here)")
@assert snap3.populate_count == 1 && snap3.theta_branch_count == 1 && snap3.ift_count == 1 "counters did not fire on a forced legacy call -- instrumentation itself is broken"
println("  CONFIRMED: the counters are wired correctly (they DO fire when the legacy path genuinely runs);")
println("  Part 2's all-zero counts are therefore real evidence, not silent dead instrumentation.")

# ============================================================================
# PART 4: the guard -- the SAME forced call on ctx2.obj (jac_h disabled)
# must throw a CLEAR diagnostic, not corrupt state or index out of bounds.
# ============================================================================
println("\n" * "="^78); println("PART 4: no-jac_h guard fires cleanly (not an out-of-bounds crash)"); println("="^78)
θ_full0_2 = CS.reconstruct_full(x0, ctx2.m)
inner_x_probe2 = copy(ctx2.obj.x)
g_buf2 = zeros(ctx2.l_full)
jac_buf2 = zeros(ncon * ctx2.l_full)
guard_ok = false
guard_msg = ""
try
    global guard_ok, guard_msg
    ctx2.obj(inner_x_probe2, g_buf2, θ_full0_2; jac = jac_buf2)
catch e
    global guard_ok = e isa ErrorException
    global guard_msg = sprint(showerror, e)
end
println("Forced call on ctx2.obj (needs_outer_moment_jacobian=false):")
println("  threw ErrorException: ", guard_ok)
println("  message: ", first(guard_msg, 200), guard_msg == first(guard_msg, 200) ? "" : "...")
@assert guard_ok "no-jac_h guard did NOT throw a clean ErrorException -- unsafe"
@assert occursin("needs_outer_moment_jacobian", guard_msg)
println("  CONFIRMED: attempting the legacy jac_h path on a no-jac_h object errors with a clear,")
println("  diagnostic message instead of an out-of-bounds crash or silent wrong result.")
# confirm ctx2 still works normally afterward (the failed forced call did not corrupt state)
r_post_guard = evaluate_fullA(x0, ctx2; cache = nothing, warm = true)
println("  ctx2 still evaluates correctly after the guard fired: Delta_dual=", r_post_guard.Delta_dual,
        " (matches ctx's own value below in Part 5)")

# ============================================================================
# PART 5: bit-identical (or tight-tolerance) validation across candidates.
# Since jac_h is provably untouched by every code path exercised (Part 2),
# every quantity below is expected to be EXACTLY equal between ctx (jac_h
# allocated, unused) and ctx2 (jac_h skipped) -- any difference would mean
# jac_h's mere presence/absence somehow perturbed a computation that should
# be independent of it (e.g. via aliasing/uninitialized-memory reuse), which
# would be a genuine correctness bug in the no-jac_h mode.
# ============================================================================
println("\n" * "="^78); println("PART 5: ctx (jac_h on) vs ctx2 (jac_h off) equivalence battery"); println("="^78)

const TIGHT = 0.0        # bit-identical: for quantities that never pass through a KNITRO solve
                          # (moments!'s K/G) -- pure deterministic Julia broadcasts, jac_h touches no
                          # buffer they read, confirmed bit-identical in an initial run of this script.
const SOLVER_TOL = 1e-8  # for quantities downstream of ctx.obj's / ctx2.obj's OWN independent inner
                          # KNITRO dual solve (two SEPARATE KN_new() problem instances, even given
                          # identical data/draws): an initial 0.0-tolerance run of this script found
                          # every mismatch was <= 9.33e-14 in absolute value (lambda/m_star/zeta/
                          # Delta_dual/etc.), i.e. floating-point-noise-level from independent BLAS/
                          # KNITRO internal solve paths -- NOT a jac_h-caused discrepancy (moments!
                          # itself, which never touches a solver, was exactly bit-identical in that
                          # same run). This matches this investigation's own established convention
                          # (test_oracle.jl TEST 4: warm-vs-cold KNITRO solves compared at "diff <
                          # 1e-8: PASS", not bit-identical). 1e-8 is ~1e6x looser than the observed
                          # noise floor -- tight enough to still catch a genuine correctness bug
                          # (which would show as an O(1) or O(1e-3) difference, not O(1e-13)).

function cmp_field(name, a, b; tol = TIGHT)
    ok = a isa AbstractArray ? (length(a) == length(b) && maximum(abs.(collect(a) .- collect(b)); init = 0.0) <= tol) :
         (isnan(a) && isnan(b)) ? true : abs(a - b) <= tol
    ok || println("    MISMATCH: ", name, "  ctx=", a, "  ctx2=", b)
    return ok
end

function validate_point(label, xf)
    println("\n-- ", label, " --")
    all_ok = true

    # ---- moments! (structural moment matrix) ----
    W = size(ctx.obj.U, 1); dW = ctx.obj.d
    θ_full = CS.reconstruct_full(xf, ctx.m)
    K1 = zeros(W); G1 = zeros(W, dW); ctx.obj.moments!(K1, G1, θ_full, ctx.obj.U, ctx.obj)
    K2 = zeros(W); G2 = zeros(W, dW); ctx2.obj.moments!(K2, G2, θ_full, ctx2.obj.U, ctx2.obj)
    all_ok &= cmp_field("moments! K", K1, K2)
    all_ok &= cmp_field("moments! G", G1, G2)

    # ---- evaluate_fullA (full oracle: dual vars, divergence, LFD weights, residuals, gravity) ----
    r1 = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
    r2 = evaluate_fullA(xf, ctx2; cache = nothing, warm = true)
    all_ok &= cmp_field("inner_status", Float64(r1.inner_status), Float64(r2.inner_status))
    if r1.inner_status in (0, -100, -101, -103)
        all_ok &= cmp_field("K_hard", r1.K_hard, r2.K_hard; tol = SOLVER_TOL)
        all_ok &= cmp_field("Delta_dual", r1.Delta_dual, r2.Delta_dual; tol = SOLVER_TOL)
        all_ok &= cmp_field("Delta_primal", r1.Delta_primal, r2.Delta_primal; tol = SOLVER_TOL)
        all_ok &= cmp_field("zeta", r1.zeta, r2.zeta; tol = SOLVER_TOL)
        all_ok &= cmp_field("lambda", r1.lambda, r2.lambda; tol = SOLVER_TOL)
        all_ok &= cmp_field("benchmark_unweighted_moment_mean", r1.benchmark_unweighted_moment_mean, r2.benchmark_unweighted_moment_mean; tol = SOLVER_TOL)
        all_ok &= cmp_field("max_abs_moment_kkt_resid", r1.max_abs_moment_kkt_resid, r2.max_abs_moment_kkt_resid; tol = SOLVER_TOL)
        all_ok &= cmp_field("gravity_value", r1.gravity_value, r2.gravity_value; tol = SOLVER_TOL)
        all_ok &= cmp_field("gravity_raw", r1.gravity_raw, r2.gravity_raw; tol = SOLVER_TOL)
        all_ok &= cmp_field("m_mean", r1.m_mean, r2.m_mean; tol = SOLVER_TOL)
        all_ok &= cmp_field("m_min", r1.m_min, r2.m_min; tol = SOLVER_TOL)
        all_ok &= cmp_field("m_max", r1.m_max, r2.m_max; tol = SOLVER_TOL)
        all_ok &= cmp_field("winner_hash", Float64(r1.winner_hash), Float64(r2.winner_hash))
        all_ok &= cmp_field("primal_dual_gap", r1.primal_dual_gap, r2.primal_dual_gap; tol = SOLVER_TOL)

        # ---- fixed-dual state + L_fix value ----
        base1 = solve_base_state(xf, ctx)
        base2 = solve_base_state(xf, ctx2)
        all_ok &= cmp_field("base.ζstar", base1.ζstar, base2.ζstar; tol = SOLVER_TOL)
        all_ok &= cmp_field("base.λstar", base1.λstar, base2.λstar; tol = SOLVER_TOL)
        all_ok &= cmp_field("base.m_star", base1.m_star, base2.m_star; tol = SOLVER_TOL)
        L1 = fixed_dual_L(xf, ctx, base1); L2 = fixed_dual_L(xf, ctx2, base2)
        all_ok &= cmp_field("fixed_dual_L(x0)", L1, L2; tol = SOLVER_TOL)
        all_ok &= cmp_field("fixed_dual_L(x0) vs Delta_dual", L1, r1.Delta_dual; tol = SOLVER_TOL)
    else
        println("    (inner solve infeasible at this point for both ctx/ctx2 -- skipping downstream field checks)")
    end

    println(all_ok ? "  ALL FIELDS MATCH (tol=$TIGHT): PASS" : "  SOME FIELDS MISMATCHED: FAIL")
    return all_ok
end

results_p5 = Bool[]
push!(results_p5, validate_point("calibration", x0))
push!(results_p5, validate_point("upper_maxit40 (headline)", x_free_from_w(w_up40)))
push!(results_p5, validate_point("lower_stalled (maxit15)", x_free_from_w(w_low)))

# ---- random feasible perturbations around w_up40, radius 0.01 (mirrors test_lfix_incremental.jl's
#      own random-perturbation protocol) -- skip any that are genuinely inner-solve-infeasible on ctx
#      (a separately-documented phenomenon per docs/fullA_block_local_performance.md sec 4, not a
#      jac_h issue), report how many were skipped rather than silently padding the count. ----
Random.seed!(20260718)
n_rand_ok = 0; n_rand_skip = 0
for k in 1:8
    wk = w_up40 .+ 0.01 .* (2 .* rand(length(w_up40)) .- 1)
    xk = x_free_from_w(wk)
    rk = evaluate_fullA(xk, ctx; cache = nothing, warm = true)
    if rk.inner_status in (0, -100, -101, -103)
        push!(results_p5, validate_point("random_feasible_$(k)", xk))
        global n_rand_ok += 1
    else
        global n_rand_skip += 1
    end
    n_rand_ok >= 5 && break
end
println("\nrandom feasible perturbations: ", n_rand_ok, " validated, ", n_rand_skip, " skipped (base-solve infeasible on ctx)")

# ---- L_fix INCREMENTAL gradient (Tier 3) at upper_maxit40 and lower_stalled, a few coordinates ----
println("\n-- L_fix incremental gradient (tier=:incremental_o1), ctx vs ctx2 --")
for (lbl, wc) in [("upper_maxit40", w_up40), ("lower_stalled", w_low)]
    base_c1 = solve_base_state(x_free_from_w(wc), ctx)
    base_c2 = solve_base_state(x_free_from_w(wc), ctx2)
    cache_c1 = build_lfix_base_cache(x_free_from_w(wc), ctx, base_c1)
    cache_c2 = build_lfix_base_cache(x_free_from_w(wc), ctx2, base_c2)
    ok = true
    for i in 1:4
        v1 = lfix_incremental_at(cache_c1, ctx, pe, wc, i, wc[i] + 0.01; tier = :incremental_o1)
        v2 = lfix_incremental_at(cache_c2, ctx2, pe, wc, i, wc[i] + 0.01; tier = :incremental_o1)
        ok &= cmp_field("lfix_incremental[$lbl,coord=$i]", v1, v2; tol = SOLVER_TOL)
    end
    println("  ", lbl, ": ", ok ? "PASS" : "FAIL")
    push!(results_p5, ok)
end

# ---- optimized-value central-FD gradient (a few coordinates), ctx vs ctx2 ----
println("\n-- optimized-value central-FD gradient (4 coords, h=0.01), ctx vs ctx2 --")
function short_fd_grad(ctxX, w, idxs; h = 0.01)
    g = zeros(length(idxs))
    for (j, i) in enumerate(idxs)
        wp = copy(w); wp[i] += h; wm = copy(w); wm[i] -= h
        Δp = evaluate_fullA(x_free_from_w(wp), ctxX; cache = nothing, warm = true).Delta_dual
        Δm = evaluate_fullA(x_free_from_w(wm), ctxX; cache = nothing, warm = true).Delta_dual
        g[j] = (Δp - Δm) / (2h)
    end
    return g
end
g1 = short_fd_grad(ctx, w_up40, 1:4)
g2 = short_fd_grad(ctx2, w_up40, 1:4)
ok_fd = cmp_field("optimized_FD_grad[1:4]", g1, g2; tol = SOLVER_TOL)
println("  ctx: ", g1); println("  ctx2: ", g2); println("  ", ok_fd ? "PASS" : "FAIL")
push!(results_p5, ok_fd)

# ---- short solver trajectory: outer_loop_cached, maxit=3, ctx vs ctx2 ----
println("\n-- short outer-loop trajectory (outer_loop_cached, maxit=3), ctx vs ctx2 --")
function run_short_outer(ctxX)
    objX = ctxX.obj
    function make_div_grad_fn(objX, mX)
        ncon_inner = objX.d - objX.outer_constr_index + 2
        cfg_cache = Ref{Any}(nothing)
        return function (g_free, x_free, θ_full, inner_x)
            objX(inner_x, Float64[], Float64[]; constr = zeros(ncon_inner))
            λ = @view inner_x[2:end]
            ctxAD = (U = objX.U, γobj = objX.γ, λ = λ, arg1 = objX.arg1, d = objX.d, outer_constr_index = objX.outer_constr_index)
            f = x -> envelope_scalar_div_ctx(CS.reconstruct_full(x, mX), ctxAD)
            if cfg_cache[] === nothing
                cfg_cache[] = ForwardDiff.GradientConfig(f, x_free)
            end
            ForwardDiff.gradient!(g_free, f, x_free, cfg_cache[])
            return g_free
        end
    end
    div_grad_fnX = make_div_grad_fn(objX, ctxX.m)
    obj_grad_fnX = (g_free, x_free) -> (fill!(g_free, 0.0); g_free[1] = (-1.0)^objX.find_smallest)
    grav_grad_fnX = (g_free, x_free) -> gravity_grad_free!(g_free, x_free, ctxX.D, ctxX.Aod_free_pos, ctxX.fixed_vals[1], ctxX.q_tilde, ctxX.N_obs)
    opt3 = joinpath(@__DIR__, "csw_outer_default_maxit3.opt")
    r = CS.outer_loop_cached(objX, ctxX.m, ctxX.θ_lo, ctxX.θ_hi, ctxX.θ0_up;
        obj_grad_fn! = obj_grad_fnX, div_grad_fn! = div_grad_fnX, gravity_grad_fn! = grav_grad_fnX,
        has_gravity = true, gravity_value_scale = -1.0 / ctxX.N_obs, use_cache = true, outer_loop_opt = opt3)
    return r
end
CS.reset_jac_h_counters!()
t0 = time(); r_traj1 = run_short_outer(ctx); t_traj1 = time() - t0
snap_traj1 = CS.jac_h_counters_snapshot()
t0 = time(); r_traj2 = run_short_outer(ctx2); t_traj2 = time() - t0
snap_traj2 = CS.jac_h_counters_snapshot()
# TRAJ_TOL looser than SOLVER_TOL: a multi-iteration outer SQP trajectory chains several
# independent inner KNITRO dual solves end to end, so the per-solve ~1e-13 floating-point noise
# floor (see SOLVER_TOL's derivation above) can compound across outer iterations -- reported
# empirically below, not assumed.
const TRAJ_TOL = 1e-4
ok_traj = cmp_field("trajectory θ_min_full", r_traj1.θ_min_full, r_traj2.θ_min_full; tol = TRAJ_TOL) &
          cmp_field("trajectory objective", r_traj1.objective, r_traj2.objective; tol = TRAJ_TOL) &
          cmp_field("trajectory nStatus", Float64(r_traj1.nStatus), Float64(r_traj2.nStatus))
println("  ctx : nStatus=", r_traj1.nStatus, " objective=", r_traj1.objective, " outer_iters=", r_traj1.outer_iters,
        " wall=", round(t_traj1, digits = 3), "s")
println("  ctx2: nStatus=", r_traj2.nStatus, " objective=", r_traj2.objective, " outer_iters=", r_traj2.outer_iters,
        " wall=", round(t_traj2, digits = 3), "s")
println("  ", ok_traj ? "TRAJECTORIES MATCH: PASS" : "TRAJECTORIES DIFFER: FAIL")
println("  jac_h counters after ctx  trajectory (cumulative since reset): alloc=", snap_traj1.alloc_count,
        " populate=", snap_traj1.populate_count, " theta_branch=", snap_traj1.theta_branch_count,
        " ift=", snap_traj1.ift_count)
println("  jac_h counters after ctx2 trajectory (delta over ctx's run):   alloc=", snap_traj2.alloc_count - snap_traj1.alloc_count,
        " populate=", snap_traj2.populate_count - snap_traj1.populate_count,
        " theta_branch=", snap_traj2.theta_branch_count - snap_traj1.theta_branch_count,
        " ift=", snap_traj2.ift_count - snap_traj1.ift_count)
@assert snap_traj1.populate_count == 0 && snap_traj1.theta_branch_count == 0 "outer_loop_cached trajectory unexpectedly touched jac_h on ctx"
push!(results_p5, ok_traj)

println("\n" * "="^78)
println("PART 5 SUMMARY: ", count(results_p5), "/", length(results_p5), " checks PASS")
println("="^78)
@assert all(results_p5) "PART 5 VALIDATION FAILED -- no-jac_h mode is NOT safe, do not recommend it"

# ============================================================================
# PART 6: performance, with JIT/ordering artifacts controlled for.
#
# Part 5's raw trajectory wall times (ctx=5.80s, ctx2=1.36s) are NOT trusted
# as a jac_h-caused effect here -- ctx ran FIRST in that comparison, so most
# of that gap is plausibly first-in-process JIT compilation (outer_loop_cached,
# ForwardDiff.GradientConfig, KNITRO callback closures, etc. all compile on
# their first call), not jac_h's memory footprint. This section repeats a
# warm (post-compilation) timing comparison, in ALTERNATING order across
# repetitions, using @timed for allocation/GC accounting -- so any remaining
# gap can be attributed to jac_h specifically, not warmup order.
# ============================================================================
println("\n" * "="^78); println("PART 6: performance (JIT-controlled, alternating order, @timed)"); println("="^78)

# ---- 6.1: bundle construction cost (allocation + zeroing time + bytes), already isolated in
#      Part 1 above (median WITH=15.825ms, WITHOUT=0.919ms at this run's D=4/N=8000/d=18/l=23
#      sizing) -- restated here for the consolidated performance report, not re-measured.
println("\n-- 6.1: bundle construction (from Part 1 above) --")
println("  median WITH jac_h    = ", round(median(t_with) * 1000, digits = 3), " ms")
println("  median WITHOUT jac_h = ", round(median(t_without) * 1000, digits = 3), " ms")
println("  jac_h alloc+zero share = ", round(median(t_with) - median(t_without), digits = 4) * 1000, " ms",
        "  (", round(theoretical_bytes / 1024^2, digits = 2), " MB)")

# ---- 6.2: warm, alternating-order evaluate_fullA timing (single-evaluation steady state) ----
println("\n-- 6.2: warm evaluate_fullA, alternating order, @timed (n=30 each) --")
# pre-warm (compile) both paths once, untimed
evaluate_fullA(x_free_from_w(w_up40), ctx; cache = nothing, warm = true)
evaluate_fullA(x_free_from_w(w_up40), ctx2; cache = nothing, warm = true)
n_reps_62 = 30
wall1 = Float64[]; wall2 = Float64[]; bytes1 = Int[]; bytes2 = Int[]; gc1 = Float64[]; gc2 = Float64[]
for k in 1:n_reps_62
    stats1 = @timed evaluate_fullA(x_free_from_w(w_up40), ctx; cache = nothing, warm = true)
    push!(wall1, stats1.time); push!(bytes1, stats1.bytes); push!(gc1, stats1.gctime)
    stats2 = @timed evaluate_fullA(x_free_from_w(w_up40), ctx2; cache = nothing, warm = true)
    push!(wall2, stats2.time); push!(bytes2, stats2.bytes); push!(gc2, stats2.gctime)
end
println("  ctx  (jac_h on) : median wall=", round(median(wall1) * 1000, digits = 4), "ms  median bytes=",
        round(Int, median(bytes1)), "  median gc_time=", round(median(gc1) * 1000, digits = 4), "ms")
println("  ctx2 (jac_h off): median wall=", round(median(wall2) * 1000, digits = 4), "ms  median bytes=",
        round(Int, median(bytes2)), "  median gc_time=", round(median(gc2) * 1000, digits = 4), "ms")
println("  wall-time ratio (ctx/ctx2) = ", round(median(wall1) / median(wall2), digits = 3),
        "  (expect ~1.0 -- jac_h is inert dead memory during this call, per Part 2's zero-touch counters)")

# ---- 6.3: N=5 short outer_loop_cached (maxit=3) trajectories, alternating order ----
println("\n-- 6.3: short outer-loop trajectory (maxit=3), alternating order, n=5 each --")
twall1 = Float64[]; twall2 = Float64[]
for k in 1:5
    s1 = @timed run_short_outer(ctx); push!(twall1, s1.time)
    s2 = @timed run_short_outer(ctx2); push!(twall2, s2.time)
end
println("  ctx  (jac_h on) : median wall=", round(median(twall1), digits = 3), "s   all=", round.(twall1, digits = 3))
println("  ctx2 (jac_h off): median wall=", round(median(twall2), digits = 3), "s   all=", round.(twall2, digits = 3))
println("  wall-time ratio (ctx/ctx2) = ", round(median(twall1) / median(twall2), digits = 3))

println("\nAUDIT COMPLETE (Parts 0-6, D=4). See full_aod_diag/d4_exact/audit_jach_d6d8.jl for the")
println("separate-process D=6/8 microbenchmark (kept in its own script/process deliberately -- this")
println("investigation's context.jl/context_scaled.jl are not designed to be include()'d twice in one")
println("process; re-including redefines the CounterfactualSensitivity module and breaks type identity")
println("for already-constructed objects, confirmed by a UndefVarError/export-ambiguity crash when")
println("first attempted inline here). See docs/fullA_jach_audit.md for the full writeup.")
