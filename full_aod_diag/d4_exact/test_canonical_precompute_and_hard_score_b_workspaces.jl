# ============================================================================
# Correctness + before/after allocation gate for section 3.3's two fixes:
#   - attach_hard_score_b_cache / hard_score_B caching (infeasibility_screen.jl)
#   - attach_canonical_price_precompute_workspace / canonical_price_precompute's
#     mulU/UPow/UσPow persistent buffers (winner_certificate.jl)
#
# Usage: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_canonical_precompute_and_hard_score_b_workspaces.jl
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
println("Section 1: real D=20/W=80,000 context")
println("="^78)
lp(">>> building real D=20/W=80000 context...")
# 2026-08-11: `inner_lower_limit` became a REQUIRED kwarg (no default) in the 2026-08-06
# lower-limit/hotpath task, which left this script throwing UndefKeywordError before its first
# check -- one of the ~200 scripts CLAUDE.md records as intentionally broken by that hardening.
# Supplying the production value explicitly is compliance with that rule, not a workaround; the
# value is irrelevant to what this script tests (workspace caching), but it must be stated.
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row,
                      inner_lower_limit = -10.0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
θfull_calib = CS.reconstruct_full(x_free_calib, ctx0.m)

println("="^78)
println("Section 2: hard_score_B caching -- bit-identical + allocation")
println("="^78)
B_orig = hard_score_B(ctx0)
ctx_hs = attach_hard_score_b_cache(ctx0)
check("attach adds hard_score_B_cache field", hasproperty(ctx_hs, :hard_score_B_cache))
B_cached = hard_score_B(ctx_hs)
check("cached hard_score_B bit-identical to original", B_orig == B_cached)
GC.gc()
b_before = @allocated hard_score_B(ctx0)
GC.gc()
b_after = @allocated hard_score_B(ctx_hs)
lp(">>> @allocated hard_score_B (no cache): ", b_before, " bytes; (cached): ", b_after, " bytes")
check("cached path allocates strictly less", b_after < b_before)
check("cached path allocates ~zero", b_after < 1024)
ctx_hs2 = attach_hard_score_b_cache(ctx_hs)
check("re-attaching is a no-op (same cache object)", ctx_hs2.hard_score_B_cache === ctx_hs.hard_score_B_cache)

println("="^78)
println("Section 3: canonical_price_precompute workspace -- bit-identical + allocation")
println("="^78)
pp_orig = canonical_price_precompute(θfull_calib, ctx0)
ctx_cp = attach_canonical_price_precompute_workspace(ctx0)
check("attach adds canonical_price_ws field", hasproperty(ctx_cp, :canonical_price_ws))
pp_new = canonical_price_precompute(θfull_calib, ctx_cp)
# NOTE: pp_new.mulU/UPow/UσPow ALIAS ctx_cp.canonical_price_ws's own mutable buffers (same
# aliasing discipline as CompressedFactualWorkspace/build_compressed_factual! -- see that file's
# docstring). A subsequent canonical_price_precompute call using the SAME ctx_cp overwrites them
# in place, so any comparison against a LATER call must copy first, exactly like this test does.
mulU_1 = copy(pp_new.mulU); UPow_1 = copy(pp_new.UPow); UσPow_1 = copy(pp_new.UσPow)
check("mulU bit-identical", pp_orig.mulU == mulU_1)
check("UPow bit-identical", pp_orig.UPow == UPow_1)
check("UσPow bit-identical", pp_orig.UσPow == UσPow_1)
check("constCons/logCC/AodPow still bit-identical (untouched by this fix)",
      pp_orig.constCons == pp_new.constCons && pp_orig.logCC == pp_new.logCC && pp_orig.AodPow == pp_new.AodPow)

# a second, different theta through the SAME workspace -- confirms mu-dependent recompute (not a
# stale cached value) and no cross-call leakage, comparing against the COPIED 1st-point arrays
# above (not pp_new directly, which is now stale/overwritten -- this IS the aliasing hazard the
# workspace's docstring warns callers about, deliberately exercised here rather than avoided).
θfull_near = copy(θfull_calib); θfull_near[1] *= 1.001   # perturb mu
pp_orig2 = canonical_price_precompute(θfull_near, ctx0)
pp_new2 = canonical_price_precompute(θfull_near, ctx_cp)   # SAME ctx_cp/workspace, 2nd theta -- overwrites pp_new's buffers in place
check("2nd (perturbed-mu) point: mulU bit-identical, same workspace", pp_orig2.mulU == pp_new2.mulU)
check("2nd point differs from 1st's COPIED values (genuine mu-dependent recompute, not stale)", pp_new2.mulU != mulU_1)
check("aliasing hazard confirmed as documented: pp_new.mulU (uncopied) now EQUALS pp_new2.mulU (same backing buffer)",
      pp_new.mulU == pp_new2.mulU)

GC.gc()
b_cp_before = @allocated canonical_price_precompute(θfull_calib, ctx0)
GC.gc()
b_cp_after = @allocated canonical_price_precompute(θfull_calib, ctx_cp)
lp(">>> @allocated canonical_price_precompute (no ws): ", b_cp_before, " bytes (", round(b_cp_before/1e6, digits=2), " MB)")
lp(">>> @allocated canonical_price_precompute (workspace): ", b_cp_after, " bytes (", round(b_cp_after/1e6, digits=2), " MB)")
lp(">>> reduction: ", round(100*(1 - b_cp_after/b_cp_before), digits=2), "%  (", round((b_cp_before-b_cp_after)/1e6, digits=2), " MB saved per call)")
check("workspace path allocates strictly less", b_cp_after < b_cp_before)

println("="^78)
println("Section 4: end-to-end real screened_eval still runs correctly with both attached")
println("="^78)
pe0 = build_pivot_elimination(ctx0)
rsc0 = build_ranged_screen_context(ctx0)
ctx_full = attach_hard_score_b_cache(attach_canonical_price_precompute_workspace(attach_compressed_factual_workspace(ctx0, ctx0.D, ctx0.D_dest, 80_000)))
sc = ScreenCounters(); n_eval = Ref(0)
r, meta = screened_eval(x_free_calib, ctx_full, rsc0, sc, n_eval; warm = false)
check("inner solve feasible with all three workspaces attached", r.inner_status in FEASIBLE_CODES)
check("Delta_dual finite", isfinite(r.Delta_dual))
lp(">>> Delta_dual=", r.Delta_dual)

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
