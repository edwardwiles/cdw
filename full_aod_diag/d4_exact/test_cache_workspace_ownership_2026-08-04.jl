# Task §5 (profiled-outer-ab-completion-2026-08-04): ProfiledLFixCache / shared-workspace
# ownership gate. Proves, by construction, that the per-thread workspace pool + copy-out fix in
# profiled_lfix_incremental_2026-08-01.jl (`SHARED_PROFILED_LFIX_WS_POOL`,
# `build_price_winner_base_cache`) removes the aliasing hazard the prior
# profiled-outer-ab-readiness-2026-08-04 session discovered but did not fix: two
# `ProfiledLFixCache` objects with overlapping lifetimes previously shared ALIASED mutable arrays
# (`logCC0`/`mulU`/`winner0`/...) backed by a single module-level `Ref{LFixFactorizedWorkspace}`,
# so building cache B silently mutated cache A's data out from under it.
#
# Real D4 KNITRO context, no mocks. D4 chosen for speed -- the higher-level "xA -> xB -> xA
# freshness" check at real D20/W=20,000 already exists and independently re-passes against this
# same fix (test_threaded_gradient_gate_2026-08-04.jl, re-run this session: "xA->xB->xA freshness
# max abs diff (gA1 vs gA2) = 0.0"); this file adds the lower-level checks that test does NOT
# cover (it only compares returned gradient vectors, never holds two live ProfiledLFixCache
# objects at once).
D4X = @__DIR__
include(joinpath(D4X, "context.jl"))
include(joinpath(dirname(dirname(D4X)), "cc_algo", "active_layout.jl"))
include(joinpath(D4X, "compressed_moments.jl"))
include(joinpath(D4X, "oracle.jl"))
include(joinpath(D4X, "oracle_fast.jl"))
include(joinpath(D4X, "operator_psi_bundle.jl"))
include(joinpath(D4X, "compressed_live.jl"))
include(joinpath(D4X, "operator_verification.jl"))
include(joinpath(D4X, "winner_certificate.jl"))
include(joinpath(D4X, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(D4X, "gravity_elimination.jl"))
include(joinpath(D4X, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(D4X, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(D4X, "recover_full_a_2026-07-31.jl"))
include(joinpath(D4X, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(D4X, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(D4X, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(D4X, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(D4X, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(D4X, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(D4X, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(D4X, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(D4X, "profiled_outer_gradient_fd_2026-08-01.jl"))
include(joinpath(D4X, "three_way_derivatives.jl"))
include(joinpath(D4X, "lfix_incremental.jl"))
include(joinpath(D4X, "gradient_workspace.jl"))
include(joinpath(D4X, "lfix_factorized_workspace.jl"))
include(joinpath(D4X, "profiled_lfix_incremental_2026-08-01.jl"))
include(joinpath(D4X, "profiled_outer_gradient_layout_contract_2026-08-01.jl"))
include(joinpath(D4X, "profiled_shared_economic_gradient_engine_2026-08-01.jl"))
include(joinpath(D4X, "profiled_family_adapters_2026-08-01.jl"))
using Random, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
lp("Threads.nthreads() = ", Threads.nthreads())

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_free = outer_dim_profiled(pe) - 1

evA = evaluate_profiled_point(w_calib, ctx, spec, pe)
ufctxA = build_unrestricted_family_ctx(ctx, spec, pe, evA)

results = Dict{Symbol,Bool}()

# ---------------------------------------------------------------------------
# 1. Two-cache overlapping-lifetime test: build cache A, snapshot its arrays, build cache B at a
# DIFFERENT point (same D/Ddest/W so the workspace pool slot is genuinely reused, not rebuilt),
# then confirm cache A's arrays are UNCHANGED -- i.e. cache A owns independent data, it does not
# alias the (reused) persistent workspace.
# ---------------------------------------------------------------------------
lp("\n=== 1. Two-cache overlapping-lifetime test ===")
cacheA = build_shared_profiled_lfix_cache(w_calib, ufctxA, ctx, evA)
snap_logCC0 = copy(cacheA.logCC0); snap_mulU = copy(cacheA.mulU)
snap_winner0 = copy(cacheA.winner0); snap_winner_price0 = copy(cacheA.winner_price0)
snap_q0 = copy(cacheA.q0)

Random.seed!(4204)
w_B = copy(w_calib); w_B[2:end] .+= 0.05 .* randn(n_free)
evB = evaluate_profiled_point(w_B, ctx, spec, pe)
ufctxB = build_unrestricted_family_ctx(ctx, spec, pe, evB)
cacheB = build_shared_profiled_lfix_cache(w_B, ufctxB, ctx, evB)

diff_logCC0 = maximum(abs.(cacheA.logCC0 .- snap_logCC0))
diff_mulU = maximum(abs.(cacheA.mulU .- snap_mulU))
diff_winner0 = maximum(abs.(cacheA.winner0 .- snap_winner0))
diff_winner_price0 = maximum(abs.(cacheA.winner_price0 .- snap_winner_price0))
diff_q0 = maximum(abs.(cacheA.q0 .- snap_q0))
lp("  cacheA.logCC0 drift after building cacheB       = ", diff_logCC0)
lp("  cacheA.mulU drift after building cacheB         = ", diff_mulU)
lp("  cacheA.winner0 drift after building cacheB      = ", diff_winner0)
lp("  cacheA.winner_price0 drift after building cacheB = ", diff_winner_price0)
lp("  cacheA.q0 drift after building cacheB            = ", diff_q0)
lp("  cacheA vs cacheB genuinely different (sanity, should be >0) = ", maximum(abs.(cacheA.q0 .- cacheB.q0)))
results[:two_cache_overlapping_lifetime] = (diff_logCC0 == 0.0) && (diff_mulU == 0.0) &&
    (diff_winner0 == 0.0) && (diff_winner_price0 == 0.0) && (diff_q0 == 0.0) &&
    (maximum(abs.(cacheA.q0 .- cacheB.q0)) > 0.0)

# ---------------------------------------------------------------------------
# 2. xA -> xB -> xA (cache-level, not just gradient-level): rebuild at A again after B, confirm the
# newly-built cache A2 matches the ORIGINAL snapshot exactly (workspace pool slot correctly
# refilled from scratch each build, no leftover state from B).
# ---------------------------------------------------------------------------
lp("\n=== 2. xA -> xB -> xA cache-rebuild test ===")
cacheA2 = build_shared_profiled_lfix_cache(w_calib, ufctxA, ctx, evA)
diff_A2_logCC0 = maximum(abs.(cacheA2.logCC0 .- snap_logCC0))
diff_A2_q0 = maximum(abs.(cacheA2.q0 .- snap_q0))
lp("  cacheA2 (rebuilt at A after B) vs original A snapshot: logCC0 diff = ", diff_A2_logCC0, "  q0 diff = ", diff_A2_q0)
results[:xA_xB_xA_cache_rebuild] = (diff_A2_logCC0 == 0.0) && (diff_A2_q0 == 0.0)

# ---------------------------------------------------------------------------
# 3. Parallel-thread cache test: build a cache concurrently on every thread at a distinct
# perturbed point, then verify EVERY thread's cache matches an independently-computed serial
# reference at that same point -- proves the per-thread workspace pool prevents cross-thread
# corruption when multiple gradient calls' cache-builds genuinely overlap in time.
# ---------------------------------------------------------------------------
lp("\n=== 3. Parallel-thread cache test ===")
nT = Threads.nthreads()
nT > 1 || error("test_cache_workspace_ownership: needs Threads.nthreads()>1 (got 1) -- relaunch with -t 10")
n_pts = max(nT * 3, 12)
Random.seed!(777)
pts = Vector{Vector{Float64}}(undef, n_pts)
for i in 1:n_pts
    wi = copy(w_calib); wi[2:end] .+= 0.02 .* randn(n_free) .* (1 + 0.1 * i)
    pts[i] = wi
end
# Serial reference q0 for every point, computed BEFORE the threaded section (ground truth).
ref_q0 = Vector{Vector{Float64}}(undef, n_pts)
for i in 1:n_pts
    evi = evaluate_profiled_point(pts[i], ctx, spec, pe)
    fctxi = build_unrestricted_family_ctx(ctx, spec, pe, evi)
    ref_q0[i] = build_shared_profiled_lfix_cache(pts[i], fctxi, ctx, evi).q0
end
par_q0 = Vector{Vector{Float64}}(undef, n_pts)
par_tid = Vector{Int}(undef, n_pts)
errs = Vector{Union{Nothing,Exception}}(nothing, n_pts)
Threads.@threads for i in 1:n_pts
    try
        par_tid[i] = Threads.threadid()
        evi = evaluate_profiled_point(pts[i], ctx, spec, pe)
        fctxi = build_unrestricted_family_ctx(ctx, spec, pe, evi)
        par_q0[i] = build_shared_profiled_lfix_cache(pts[i], fctxi, ctx, evi).q0
    catch e
        errs[i] = e
    end
end
n_errors = count(!isnothing, errs)
max_par_diff = n_errors == 0 ? maximum(maximum(abs.(par_q0[i] .- ref_q0[i])) for i in 1:n_pts) : Inf
n_distinct_threads = length(unique(par_tid))
lp("  points=", n_pts, "  distinct threads used=", n_distinct_threads, "  errors=", n_errors)
lp("  max |parallel q0 - serial-reference q0| across all points = ", max_par_diff)
results[:parallel_thread_cache] = (n_errors == 0) && (max_par_diff == 0.0)

# ---------------------------------------------------------------------------
# 4. Different-(D,Ddest,W) rejection test: force the calling thread's workspace pool slot to hold
# a (D,Ddest,W) built for one shape, then request a genuinely different W -- confirm the guard in
# ensure_shared_profiled_lfix_ws! rebuilds fresh (correct new shape) rather than silently returning
# the old, wrong-shaped workspace.
# ---------------------------------------------------------------------------
lp("\n=== 4. Different-(D,Ddest,W) rejection test ===")
D_here = ctx.D; Ddest_here = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
ws1 = ensure_shared_profiled_lfix_ws!(D_here, Ddest_here, 37)
ws1_id = objectid(ws1)
ws2 = ensure_shared_profiled_lfix_ws!(D_here, Ddest_here, 41)
ws2_id = objectid(ws2)
ws3 = ensure_shared_profiled_lfix_ws!(D_here, Ddest_here, 37)
shape_ok = (ws1.W == 37) && (ws2.W == 41) && (ws3.W == 37)
rebuilt_on_change = (ws1_id != ws2_id)
lp("  ws1.W=", ws1.W, "  ws2.W=", ws2.W, "  ws3.W=", ws3.W, "  rebuilt_on_W_change=", rebuilt_on_change)
results[:different_shape_rejection] = shape_ok && rebuilt_on_change

lp("\nALL_RESULTS: ", results)
all_pass = all(values(results))
lp("\nCACHE_WORKSPACE_OWNERSHIP_GATE: ", all_pass ? "PASS" : "FAIL")
all_pass || error("test_cache_workspace_ownership_2026-08-04: one or more checks failed -- see ALL_RESULTS above")
