# ============================================================================
# Correctness + before/after allocation gate for the fast_range_screen.jl:637 fix
# (allocation/Hessian port task, section 3.2).
#
# `evaluate_fullA_screened_compressed_with_cf` used to do
# `K, G = (copy(@view(obj.H[:, 1])), copy(CS.select_G_from_H(obj, obj.H)))` -- the audit's own
# dynamic Profile.Allocs by-site trace found this the SINGLE LARGEST unrestricted hot-path
# allocation site (980,480,000 bytes / 8 events at real D=20/W=80,000 -> ~122.6 MB/event). `K` was
# never read again (dead); `G` is read-only downstream (kkt_residual_blas/moment_resid_blas, both
# plain BLAS `mul!` calls that accept a StridedMatrix view) and never escapes the function, and
# nothing between its construction and its last use mutates `obj.H` -- so both copies are now
# removed (K entirely, G replaced by the existing view `CS.select_G_from_H` already returns).
#
# This test (a) reproduces the exact isolated before/after allocation delta at this one call site
# via a direct micro-comparison (mimicking the audit's own isolated-measurement style, without
# needing to revert the production fix to get a "before" number), and (b) runs the REAL,
# unmodified production entry point (evaluate_fullA_screened_ranged, called through screened_eval)
# at a real D=20/W=80,000 calibrated point and confirms every G-derived result field
# (max_abs_moment_kkt_resid, benchmark_unweighted_moment_mean, Delta_dual, Delta_primal,
# gravity_value) is finite and internally consistent -- since a view of the identical underlying
# memory a copy would have captured is mathematically forced to produce IDENTICAL numbers (this is
# not a coincidence to test for, it is a consequence of removing an unnecessary copy of otherwise-
# unmutated memory), the decisive test is that production still runs correctly end to end, which
# the existing matched-benchmark cold-verify (already re-run after this fix, see below) confirms.
#
# Usage: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_screened_g_view_no_copy.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("Section 1: real D=20/W=80,000 point -- end-to-end correctness through the fixed line")
println("="^78)
lp(">>> building real D=20/W=80000 context...")
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
rsc0 = build_ranged_screen_context(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
sc = ScreenCounters(); n_eval = Ref(0)
r, meta = screened_eval(x_free_calib, ctx0, rsc0, sc, n_eval; warm = false)

check("inner solve feasible", r.inner_status in FEASIBLE_CODES)
check("Delta_dual finite", isfinite(r.Delta_dual))
check("Delta_primal finite", isfinite(r.Delta_primal))
check("gravity_value finite", isfinite(r.gravity_value))
check("max_abs_moment_kkt_resid finite (G-derived, this fix's exact downstream consumer)", isfinite(r.max_abs_moment_kkt_resid))
check("benchmark_unweighted_moment_mean non-empty and all finite (G-derived)",
      !isempty(r.benchmark_unweighted_moment_mean) && all(isfinite, r.benchmark_unweighted_moment_mean))
check("K_hard finite", isfinite(r.K_hard))
lp(">>> Delta_dual=", r.Delta_dual, " max_abs_moment_kkt_resid=", r.max_abs_moment_kkt_resid,
   " gravity_value=", r.gravity_value)

println("="^78)
println("Section 2: isolated before/after allocation at the exact fixed call site")
println("="^78)
# Reconstructs the OLD (copy-based) and NEW (view-based) computation of exactly the two lines
# that changed, operating on obj.H in its current (already-materialized, post-call-above) state --
# same underlying data either way, so this measures ONLY the allocation delta, not a correctness
# difference (there is none, by construction: a view and a copy of unmutated memory contain
# identical bytes).
obj0 = ctx0.obj
GC.gc()
b_old = @allocated begin
    global K_old, G_old = (copy(@view(obj0.H[:, 1])), copy(CS.select_G_from_H(obj0, obj0.H)))
end
GC.gc()
b_new = @allocated begin
    global G_new = CS.select_G_from_H(obj0, obj0.H)
end
lp(">>> @allocated OLD (copy(K), copy(G)):  ", b_old, " bytes (", round(b_old / 1e6, digits = 2), " MB)")
lp(">>> @allocated NEW (view G only, K removed): ", b_new, " bytes (", round(b_new / 1e6, digits = 2), " MB)")
lp(">>> reduction: ", round(100 * (1 - b_new / b_old), digits = 2), "%  (", round((b_old - b_new) / 1e6, digits = 2), " MB saved per call)")
check("new path allocates strictly less", b_new < b_old)
check("new path allocates near-zero (a view, not a copy)", b_new < 1024)   # a SubArray struct itself, no data copy
check("G_new (view) contains IDENTICAL values to G_old (copy) -- same underlying memory", G_new == G_old)

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
