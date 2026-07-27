# ============================================================================
# Shared economic moment-state builder task (2026-07-27) -- correctness + counter gate.
#
# Targets the specific gap this task fixed: `compressed_live.jl::inner_loop_internal_compressed`
# (the UNRESTRICTED family's real per-inner-solve moment build under
# `moment_representation=:compressed`, the production driver's default mode) previously called the
# always-allocating `build_compressed_factual` directly, ignoring any `ctx.cf_workspace` the
# production driver had already attached. This gate verifies:
#   (1) build_economic_moment_state! is bit-identical to the allocating reference AND to the
#       pre-existing build_compressed_factual! path, at D=4 square and D=4 rectangular (exclude_row
#       via row_idx=D pseudo-ROW, mirroring d_exact_setup_scaled's own convention);
#   (2) evaluate_fullA_fast(...; moment_representation=:compressed) -- i.e. the REAL fixed hot path
#       -- agrees with the dense reference at TWO DISTINCT outer points through the SAME attached
#       workspace (stale-value detection, per addendum section 7's "at least two distinct outer
#       points" requirement);
#   (3) the runtime counters behave as required: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS stays 0
#       once a workspace is attached, INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS/ECONOMIC_WORKSPACE_
#       REFILLS increase, and DUPLICATE_ECONOMIC_STATE_BUILDS increments on a genuine repeat call at
#       the identical point;
#   (4) without an attached workspace, the allocating fallback still fires unchanged (no regression
#       for any pre-existing caller that never attaches one).
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_shared_economic_moment_state_builder_2026-07-27.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
isdefined(Main, :d_exact_setup_scaled) || include(joinpath(@__DIR__, "context_scaled.jl"))
using Random, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

function compare_cf(cf1::CompressedFactual, cf2::CompressedFactual, label::String)
    ok = cf1.D == cf2.D && cf1.D_dest == cf2.D_dest && cf1.W == cf2.W && cf1.oci == cf2.oci &&
         cf1.winner == cf2.winner && cf1.wval == cf2.wval && cf1.Pmat == cf2.Pmat &&
         cf1.denom == cf2.denom && cf1.gdiv == cf2.gdiv && cf1.nrm == cf2.nrm &&
         cf1.PMM == cf2.PMM && cf1.usePMM == cf2.usePMM && cf1.SW == cf2.SW &&
         cf1.gammafac == cf2.gammafac && cf1.cf_raw == cf2.cf_raw && cf1.cf_col == cf2.cf_col
    check(label * ": bit-identical", ok)
    return ok
end

println("="^90)
println("Part 1: build_economic_moment_state! vs allocating reference / build_compressed_factual!, D=4 square")
println("="^90)
ctx4 = d4_exact_setup(find_smallest = true)
D4 = ctx4.D; Ddest4 = hasproperty(ctx4, :D_dest) ? ctx4.D_dest : ctx4.D
xf4 = ctx4.θ0_up
cf_ref = build_compressed_factual(xf4, ctx4; check_ties = true)
cf_bems_nows = build_economic_moment_state!(xf4, ctx4; check_ties = true)   # ctx4 has NO cf_workspace -> fallback branch
compare_cf(cf_ref, cf_bems_nows, "D4 square: build_economic_moment_state! (no ws attached) vs allocating reference")

ctx4_ws = attach_compressed_factual_workspace(ctx4, D4, Ddest4, size(ctx4.U, 1))
cf_bems_ws = build_economic_moment_state!(xf4, ctx4_ws; check_ties = true)   # in-place branch
compare_cf(cf_ref, cf_bems_ws, "D4 square: build_economic_moment_state! (ws attached) vs allocating reference")
check("D4 square: build_economic_moment_state! returns cf whose winner/wval alias ctx4_ws.cf_workspace's buffers", cf_bems_ws.winner === ctx4_ws.cf_workspace.winner)

println("="^90)
println("Part 2: D=4 rectangular (row_idx=D pseudo-ROW, D_dest=D-1)")
println("="^90)
ctxr = d_exact_setup_scaled(D = 4, W = 8000, row_idx = 4)
Dr = ctxr.D; Ddestr = ctxr.D_dest
check("D4 rectangular: D_dest == D-1", Ddestr == Dr - 1)
xfr = ctxr.θ0_up
cfr_ref = build_compressed_factual(xfr, ctxr; check_ties = true)
ctxr_ws = attach_compressed_factual_workspace(ctxr, Dr, Ddestr, size(ctxr.U, 1))
cfr_bems = build_economic_moment_state!(xfr, ctxr_ws; check_ties = true)
compare_cf(cfr_ref, cfr_bems, "D4 rectangular: build_economic_moment_state! (ws attached) vs allocating reference")

println("="^90)
println("Part 3: REAL fixed hot path (evaluate_fullA_fast, moment_representation=:compressed) vs dense,")
println("        two distinct outer points through ONE attached workspace + counter behavior")
println("="^90)
ctx = d4_exact_setup(find_smallest = true)
D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
n_alloc_before_attach = ECONOMIC_WORKSPACE_ALLOCATIONS[]
ctx_ws = attach_compressed_factual_workspace(ctx, D, Ddest, size(ctx.U, 1))
check("attach_compressed_factual_workspace: exactly 1 new ECONOMIC_WORKSPACE_ALLOCATIONS for this fresh context", ECONOMIC_WORKSPACE_ALLOCATIONS[] == n_alloc_before_attach + 1)
pe = build_pivot_elimination(ctx_ws)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
w_calib = vcat(ctx_ws.θ0_up[3+D], vec(pivot_reduce(log.(reshape(ctx_ws.θ0_up[3+D+1:end], D, Ddest)), pe)))
x_free_calib = x_free_from_w(w_calib)
rng = MersenneTwister(20260727)
w_pert = w_calib .+ vcat(0.0, 0.03 .* randn(rng, length(w_calib) - 1))
x_free_pert = x_free_from_w(w_pert)

reset_economic_moment_state_counters!()

function compare_point(label, x_free)
    ra, _ = evaluate_fullA_fast(x_free, ctx_ws; cache = nothing, use_cache = false, warm = true, moment_representation = :dense)
    rb, _ = evaluate_fullA_fast(x_free, ctx_ws; cache = nothing, use_cache = false, warm = true, moment_representation = :compressed)
    dK = abs(ra.K_hard - rb.K_hard)
    dDelta = abs(ra.Delta_dual - rb.Delta_dual)
    dlam = isempty(ra.lambda) ? 0.0 : maximum(abs.(ra.lambda .- rb.lambda))
    ok = ra.inner_status == rb.inner_status && dK < 1e-8 && dDelta < 1e-8 && dlam < 1e-8
    lp(rpad(label, 55), "inner_status(dense/compr)=", ra.inner_status, "/", rb.inner_status,
       "  dK=", @sprintf("%.3e", dK), "  dDelta=", @sprintf("%.3e", dDelta), "  dlambda=", @sprintf("%.3e", dlam))
    check(label, ok)
    return rb
end

compare_point("calibration point (1st, workspace attached)", x_free_calib)
n_alloc_after_1 = ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[]
check("calib point: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0 after 1st compressed call", n_alloc_after_1 == 0)
n_inplace_after_1 = INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS[]
check("calib point: INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS > 0 after 1st compressed call", n_inplace_after_1 > 0)

compare_point("perturbed point (2nd, SAME workspace -- stale-value check)", x_free_pert)
check("perturbed point: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS still 0 after 2nd compressed call", ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[] == 0)
n_refills_after_2 = ECONOMIC_WORKSPACE_REFILLS[]
check("2 distinct outer points -> ECONOMIC_WORKSPACE_REFILLS >= 2", n_refills_after_2 >= 2)

n_dup_before = DUPLICATE_ECONOMIC_STATE_BUILDS[]
θ_full_pert = CS.reconstruct_full(x_free_pert, ctx_ws.m)
_ = build_economic_moment_state!(θ_full_pert, ctx_ws; check_ties = true)   # exact same θ_full as the just-built perturbed point -> duplicate
n_dup_after = DUPLICATE_ECONOMIC_STATE_BUILDS[]
check("re-building the SAME θ_full on the SAME workspace increments DUPLICATE_ECONOMIC_STATE_BUILDS", n_dup_after == n_dup_before + 1)

check("ECONOMIC_WORKSPACE_ALLOCATIONS: 0 NEW allocations across the measured window (reused, not reallocated)", ECONOMIC_WORKSPACE_ALLOCATIONS[] == 0)
check("ECONOMIC_WORKSPACE_RESIZES == 0 (no shape change occurred)", ECONOMIC_WORKSPACE_RESIZES[] == 0)

println("="^90)
println("Part 4: no workspace attached -> allocating fallback fires unchanged (no regression)")
println("="^90)
reset_economic_moment_state_counters!()
ctx_nows = d4_exact_setup(find_smallest = true)
check("fresh ctx carries no cf_workspace", !hasproperty(ctx_nows, :cf_workspace))
_ = evaluate_fullA_fast(ctx_nows.θ0_up[ctx_nows.free_idx], ctx_nows; cache = nothing, use_cache = false, warm = true, moment_representation = :compressed)
check("no-workspace ctx: ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS > 0 (fallback path exercised, exactly as before this task)", ALLOCATING_BUILD_COMPRESSED_FACTUAL_CALLS[] > 0)
check("no-workspace ctx: INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS == 0", INPLACE_BUILD_COMPRESSED_FACTUAL_CALLS[] == 0)

println("="^90)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
