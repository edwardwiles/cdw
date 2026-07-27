# ============================================================================
# Shared economic moment-state builder task (2026-07-27) -- D=20/W=80,000 real-data extension of
# test_shared_economic_moment_state_builder_2026-07-27.jl's Part 3 gate (the REAL fixed hot path,
# evaluate_fullA_fast(...; moment_representation=:compressed) with an attached cf_workspace, vs
# dense, at TWO distinct outer points through the SAME workspace). Real KNITRO, real D=20/W=80,000
# data -- run in background, expect several minutes.
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_shared_economic_moment_state_builder_d20_2026-07-27.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name); flush(stdout)
    cond || push!(FAILURES, name)
end

lp(">>> building real D=20/W=80000 context (destination_sample=:exclude_row, ROW=20 omitted, non-last)...")
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
lp(">>> D=", D, " Ddest=", Ddest, " W=", size(ctx0.U, 1), " row_idx=", ctx0.row_idx)

n_alloc_before_attach = ECONOMIC_WORKSPACE_ALLOCATIONS[]
ctx_ws = attach_compressed_factual_workspace(ctx0, D, Ddest, size(ctx0.U, 1))
check("attach_compressed_factual_workspace: exactly 1 new ECONOMIC_WORKSPACE_ALLOCATIONS", ECONOMIC_WORKSPACE_ALLOCATIONS[] == n_alloc_before_attach + 1)

x_free_calib = ctx_ws.θ0_up[ctx_ws.free_idx]
rng = MersenneTwister(20260727)
zfree_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
zfree_near = zfree_calib .+ 0.03 .* randn(rng, D * Ddest - 1)
x_free_near = x_free_from_w(vcat(x_free_calib[1], zfree_near), pe0)

reset_economic_moment_state_counters!()

function compare_point(label, x_free)
    t0 = time()
    ra, _ = evaluate_fullA_fast(x_free, ctx_ws; cache = nothing, use_cache = false, warm = true, moment_representation = :dense)
    t_dense = time() - t0
    t1 = time()
    rb, _ = evaluate_fullA_fast(x_free, ctx_ws; cache = nothing, use_cache = false, warm = true, moment_representation = :compressed)
    t_compr = time() - t1
    dK = abs(ra.K_hard - rb.K_hard)
    dDelta = abs(ra.Delta_dual - rb.Delta_dual)
    dlam = isempty(ra.lambda) ? 0.0 : maximum(abs.(ra.lambda .- rb.lambda))
    ok = ra.inner_status == rb.inner_status && dK < 1e-6 && dDelta < 1e-6 && dlam < 1e-6
    lp(rpad(label, 55), "inner_status(dense/compr)=", ra.inner_status, "/", rb.inner_status,
       "  dK=", @sprintf("%.3e", dK), "  dDelta=", @sprintf("%.3e", dDelta), "  dlambda=", @sprintf("%.3e", dlam),
       "  t_dense=", round(t_dense, digits=1), "s  t_compr=", round(t_compr, digits=1), "s")
    check(label, ok)
    return rb
end

lp(">>> Point 1: calibration...")
compare_point("D20/W80000 calibration point (workspace attached)", x_free_calib)
check("calib: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0", ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[] == 0)

lp(">>> Point 2: near-calibration (perturbed, SAME workspace)...")
compare_point("D20/W80000 near-calibration point (SAME workspace -- stale-value check)", x_free_near)
check("near-calib: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0", ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[] == 0)
check("2 distinct D20 outer points -> ECONOMIC_WORKSPACE_REFILLS >= 2", ECONOMIC_WORKSPACE_REFILLS[] >= 2)
check("D20: ECONOMIC_WORKSPACE_ALLOCATIONS == 0 new allocations in measured window", ECONOMIC_WORKSPACE_ALLOCATIONS[] == 0)
check("D20: ECONOMIC_WORKSPACE_RESIZES == 0", ECONOMIC_WORKSPACE_RESIZES[] == 0)

println("="^90)
if isempty(FAILURES)
    println("ALL D20/W80000 TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
