# ============================================================================
# Correctness gate for build_compressed_factual!/CompressedFactualWorkspace/
# attach_compressed_factual_workspace (allocation/Hessian port task, section 3.1).
#
# Ports and extends the audit's own bit-identical/allocation gate
# (diag/production-wallclock-allocation-audit-2026-07-25's test_compressed_factual_buffer_reuse.jl)
# with the additional D=4 square + D=20/small-W rectangular (non-last destination omission)
# coverage the port task requires, plus a check on attach_compressed_factual_workspace's own
# no-op-reuse-on-matching-shape / rebuild-on-shape-change contract (the mechanics `reuse=`/resumed
# checkpoints rely on).
#
# Usage: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_compressed_factual_buffer_reuse.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using LinearAlgebra, Random

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

function compare_cf(cf1::CompressedFactual, cf2::CompressedFactual, label::String)
    ok = true
    ok &= cf1.D == cf2.D && cf1.D_dest == cf2.D_dest && cf1.W == cf2.W && cf1.oci == cf2.oci
    ok &= cf1.winner == cf2.winner
    ok &= cf1.wval == cf2.wval
    ok &= cf1.Pmat == cf2.Pmat
    ok &= cf1.denom == cf2.denom
    ok &= cf1.gdiv == cf2.gdiv
    ok &= cf1.nrm == cf2.nrm
    ok &= cf1.PMM == cf2.PMM
    ok &= cf1.usePMM == cf2.usePMM
    ok &= cf1.SW == cf2.SW
    ok &= cf1.gammafac == cf2.gammafac
    ok &= cf1.cf_raw == cf2.cf_raw
    ok &= cf1.cf_col == cf2.cf_col
    check(label * ": bit-identical", ok)
    return ok
end

println("="^78)
println("Section 1: D=4 square (d4_exact_setup) -- bit-identical + two points, one workspace")
println("="^78)
ctx4 = d4_exact_setup()
pe4 = build_pivot_elimination(ctx4)
D4 = ctx4.D; Ddest4 = hasproperty(ctx4, :D_dest) ? ctx4.D_dest : ctx4.D
xf4_calib = ctx4.θ0_up
cf4_orig = build_compressed_factual(xf4_calib, ctx4; check_ties = true)
ws4 = build_compressed_factual_workspace(D4, Ddest4, size(ctx4.U, 1))
cf4_new = build_compressed_factual!(ws4, xf4_calib, ctx4; check_ties = true)
compare_cf(cf4_orig, cf4_new, "D4 square calibration point")

rng4 = MersenneTwister(20260725)
x_free4 = ctx4.θ0_up[ctx4.free_idx] .* (1.0 .+ 0.01 .* randn(rng4, length(ctx4.free_idx)))
xf4_near = CS.reconstruct_full(x_free4, ctx4.m)
cf4_orig2 = build_compressed_factual(xf4_near, ctx4; check_ties = true)
cf4_new2 = build_compressed_factual!(ws4, xf4_near, ctx4; check_ties = true)   # SAME ws4 -- second point through one workspace
compare_cf(cf4_orig2, cf4_new2, "D4 square, 2nd (perturbed) point, same workspace")

println("="^78)
println("Section 2: D=20/W=200, destination_sample=:exclude_row -- rectangular, non-last (ROW=20) omitted")
println("="^78)
ctx_r = d20_real_setup(W = 200, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe_r = build_pivot_elimination(ctx_r)
Dr = ctx_r.D; Ddest_r = ctx_r.D_dest
check("row_idx==20 (ROW is the omitted destination, not the last free column trivially)", ctx_r.row_idx == 20)
check("Ddest == D-1 (true rectangular reduction)", Ddest_r == Dr - 1)
xfr_calib = CS.reconstruct_full(ctx_r.θ0_up[ctx_r.free_idx], ctx_r.m)
cfr_orig = build_compressed_factual(xfr_calib, ctx_r; check_ties = true)
wsr = build_compressed_factual_workspace(Dr, Ddest_r, size(ctx_r.U, 1))
cfr_new = build_compressed_factual!(wsr, xfr_calib, ctx_r; check_ties = true)
compare_cf(cfr_orig, cfr_new, "D20/W200 rectangular (exclude_row) calibration point")

println("="^78)
println("Section 3: attach_compressed_factual_workspace mechanics (reuse=/resume= contract)")
println("="^78)
ctx_a = attach_compressed_factual_workspace(ctx_r, Dr, Ddest_r, 200)
check("attach adds cf_workspace field", hasproperty(ctx_a, :cf_workspace))
check("attached workspace has correct shape", ctx_a.cf_workspace.D == Dr && ctx_a.cf_workspace.Ddest == Ddest_r && ctx_a.cf_workspace.W == 200)
ctx_a2 = attach_compressed_factual_workspace(ctx_a, Dr, Ddest_r, 200)
check("re-attaching with SAME shape reuses the SAME workspace object (no rebuild)", ctx_a2.cf_workspace === ctx_a.cf_workspace)
ctx_a3 = attach_compressed_factual_workspace(ctx_a, Dr, Ddest_r, 999)
check("re-attaching with a DIFFERENT W builds a NEW workspace (genuine shape change)", ctx_a3.cf_workspace !== ctx_a.cf_workspace)
check("new workspace has the new shape", ctx_a3.cf_workspace.W == 999)
# set_context_delta! is the exact mechanism run_polish_checkpointed's reuse= path calls -- confirm
# a NEW delta preserves the SAME cf_workspace object by reference (the guarantee this fix relies on).
ctx_a4 = set_context_delta!(ctx_a, 2.0)
check("set_context_delta! preserves cf_workspace by reference across a delta change", ctx_a4.cf_workspace === ctx_a.cf_workspace)
check("set_context_delta! actually changed delta", ctx_a4.δ == 2.0)

println("="^78)
println("Section 4: real D=20/W=80,000 point -- bit-identical + before/after allocation")
println("="^78)
lp(">>> building real D=20/W=80000 context (this is the slow step, ~1-2 min)...")
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
xf_calib = CS.reconstruct_full(x_free_calib, ctx0.m)

cf_orig = build_compressed_factual(xf_calib, ctx0; check_ties = true)
ws = build_compressed_factual_workspace(D, Ddest, size(ctx0.U, 1))
cf_new = build_compressed_factual!(ws, xf_calib, ctx0; check_ties = true)
compare_cf(cf_orig, cf_new, "D20/W80000 calibration point")

rng = MersenneTwister(20260725)
zfree_near = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0) .+ 0.05 .* randn(rng, D * Ddest - 1)
x_free_near = x_free_from_w(vcat(x_free_calib[1], zfree_near), pe0)
xf_near = CS.reconstruct_full(x_free_near, ctx0.m)
cf_orig2 = build_compressed_factual(xf_near, ctx0; check_ties = true)
cf_new2 = build_compressed_factual!(ws, xf_near, ctx0; check_ties = true)   # SAME ws -- 2nd point, one workspace
compare_cf(cf_orig2, cf_new2, "D20/W80000 near-calibration point (2nd call, same ws)")

GC.gc()
b_before = @allocated build_compressed_factual(xf_calib, ctx0; check_ties = true)
GC.gc()
b_after = @allocated build_compressed_factual!(ws, xf_calib, ctx0; check_ties = true)
lp(">>> @allocated build_compressed_factual (ORIGINAL, fresh alloc):  ", b_before, " bytes (", round(b_before / 1e6, digits = 2), " MB)")
lp(">>> @allocated build_compressed_factual! (NEW, reused buffer):    ", b_after, " bytes (", round(b_after / 1e6, digits = 2), " MB)")
lp(">>> reduction: ", round(100 * (1 - b_after / b_before), digits = 2), "%  (", round((b_before - b_after) / 1e6, digits = 2), " MB saved per call)")
check("workspace variant allocates strictly less than the original", b_after < b_before)

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
