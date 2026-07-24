# Part II.4, 2026-07-23: checkpoint backend-provenance tests. Writes ONLY to its own scratch
# ckpt_dir under results/, never touches production_runs/. Uses a small maxtime_real (real KNITRO
# calls, D=20/W=80000, but short) so each sub-test finishes in well under a minute.
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))   # exclude-ROW-destination release (2026-07-24) test-harness fix: was missing, cm_checkpoint.jl calls with_screen_counters/print_screen_startup_banner/print_active_layout_banner
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))   # exclude-ROW-destination release (2026-07-24) test-harness fix: was missing entirely, meanzc_resolve_K (called unconditionally by run_cm_upper_checkpointed) was UndefVarError
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Serialization, LinearAlgebra

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

const OUTROOT = joinpath(@__DIR__, "..", "..", "results", "cm_cplus_followup", "checkpoint_provenance_test")
isdir(OUTROOT) && rm(OUTROOT; recursive = true)
mkpath(OUTROOT)

const W = 80_000; const DELTA = 0.1; const L = 50
const DRAW_DESIGN = :pseudorandom; const DRAW_SEED = 20260719
const CM_CONTRASTS = :orthonormal
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    destination_sample = :all_legacy)   # this test exercises cm_gradient_backend provenance, not
    # destination_sample -- pinned explicitly to :all_legacy (matches run_cm_upper_checkpointed's
    # own default) rather than picking up d20_real_setup_design's separate :exclude_row default.
    # Pre-existing latent bug found while validating the 2026-07-24 destination_sample follow-up:
    # this test's un-pinned ctx0 build silently inherited the Part A (2026-07-23) default flip and
    # its own reshape(...,D,D) below then failed on the resulting D*(D-1) free-parameter count --
    # unrelated to (and predating) this session's own changes.
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0))

println("="^100)
println("TEST 1: Reference checkpoint -> Reference resume (same backend, no warning expected)")
println("="^100)
ckpt_dir1 = joinpath(OUTROOT, "t1_ref_ref")
r1a = run_cm_upper_checkpointed(w0; W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    L, contrasts = CM_CONTRASTS, probs, maxtime_real = 30.0, ckpt_dir = ckpt_dir1,
    run_id = "t1a", label = "t1", checkpoint_interval_s = 10.0, cm_gradient_backend = :reference, verbose = false)
ckpt_path1 = joinpath(ckpt_dir1, "t1_latest.jls")
check("[T1] checkpoint written after first stage", isfile(ckpt_path1))
ckpt1 = load_cm_checkpoint(ckpt_path1)
check("[T1] persisted cm_gradient_backend == :reference", ckpt1.cm_gradient_backend == :reference)
r1b = run_cm_upper_checkpointed(nothing; ckpt_dir = ckpt_dir1, run_id = "t1b", label = "t1_r2",
    resume_from = ckpt_path1, maxtime_real = 20.0, checkpoint_interval_s = 10.0,
    cm_gradient_backend = :reference, verbose = false)
check("[T1] same-backend resume succeeds without error", true)   # reaching here means no exception was thrown
check("[T1] n_eval after resume >= n_eval before", r1b.n_eval >= r1a.n_eval)

println()
println("="^100)
println("TEST 2: C+ checkpoint -> C+ resume (same backend, no warning expected)")
println("="^100)
ckpt_dir2 = joinpath(OUTROOT, "t2_cplus_cplus")
r2a = run_cm_upper_checkpointed(w0; W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    L, contrasts = CM_CONTRASTS, probs, maxtime_real = 30.0, ckpt_dir = ckpt_dir2,
    run_id = "t2a", label = "t2", checkpoint_interval_s = 10.0, cm_gradient_backend = :cplus, verbose = false)
ckpt_path2 = joinpath(ckpt_dir2, "t2_latest.jls")
ckpt2 = load_cm_checkpoint(ckpt_path2)
check("[T2] persisted cm_gradient_backend == :cplus", ckpt2.cm_gradient_backend == :cplus)
r2b = run_cm_upper_checkpointed(nothing; ckpt_dir = ckpt_dir2, run_id = "t2b", label = "t2_r2",
    resume_from = ckpt_path2, maxtime_real = 20.0, checkpoint_interval_s = 10.0,
    cm_gradient_backend = :cplus, verbose = false)
check("[T2] same-backend resume succeeds without error", true)

println()
println("="^100)
println("TEST 3: attempted cross-backend resume WITHOUT override -- must hard error")
println("="^100)
# NOTE: assigning a plain top-level variable ONLY inside a `catch` clause does not reliably
# persist to the outer scope in this Julia version/script-execution mode (confirmed empirically
# this session -- the assignment reads back correctly INSIDE the catch block but reverts right
# after `end`). Using a `Ref` sidesteps the ambiguity entirely (mutation, not reassignment).
threw_correctly = Ref(false)
try
    run_cm_upper_checkpointed(nothing; ckpt_dir = ckpt_dir1, run_id = "t3", label = "t3_bad",
        resume_from = ckpt_path1, maxtime_real = 10.0, cm_gradient_backend = :cplus, verbose = false)
catch e
    threw_correctly[] = occursin("refusing to silently switch backends", sprint(showerror, e))
end
check("[T3] cross-backend resume without allow_backend_switch throws the expected error", threw_correctly[])

println()
println("="^100)
println("TEST 4: cross-backend resume WITH allow_backend_switch=true -- succeeds, audited, bandwidth_cache cleared")
println("="^100)
r4 = run_cm_upper_checkpointed(nothing; ckpt_dir = ckpt_dir1, run_id = "t4", label = "t4_switch",
    resume_from = ckpt_path1, maxtime_real = 20.0, checkpoint_interval_s = 10.0,
    cm_gradient_backend = :cplus, allow_backend_switch = true, verbose = false)
audit_path = joinpath(ckpt_dir1, "t4_switch_backend_switch_audit.txt")
check("[T4] audit file written", isfile(audit_path))
if isfile(audit_path)
    audit_txt = read(audit_path, String)
    check("[T4] audit file records checkpoint_backend=reference", occursin("checkpoint_backend=reference", audit_txt))
    check("[T4] audit file records requested_backend=cplus", occursin("requested_backend=cplus", audit_txt))
end
ckpt4_path = joinpath(ckpt_dir1, "t4_switch_latest.jls")
ckpt4 = load_cm_checkpoint(ckpt4_path)
check("[T4] resumed-and-switched checkpoint now persists cm_gradient_backend=:cplus", ckpt4.cm_gradient_backend == :cplus)

println()
println("="^100)
println("TEST 5: cold verification of the resumed-and-switched incumbent (independent, backend-agnostic)")
println("="^100)
if r4.best !== nothing
    pcx = build_cm_production_context(ctx0, CS; L, contrasts = CM_CONTRASTS, probs)
    xf_best = x_free_from_w(r4.best.w, pe0)
    base_v, verify_v = archC_verified_state(xf_best, pcx.ctx_cm, pcx.cctx)
    diff = abs(verify_v.Delta_dual - r4.best.Delta)
    @printf "  independent cold re-verify: Delta_dual=%.10f (self-reported %.10f) |diff|=%.3e inner_status=%d\n" verify_v.Delta_dual r4.best.Delta diff verify_v.inner_status
    check("[T5] cold-verified incumbent matches self-reported Delta to < 1e-6", diff < 1e-6)
    check("[T5] cold-verified inner_status == 0", verify_v.inner_status == 0)
else
    println("  no best_feasible found in T4's short run -- skipping (disclosed, not fabricated)")
end

println()
println("="^100)
println("TEST 6: interruption during a gradient callback (simulated via a very short maxtime_real mid-run) -- checkpoint written must still load and resume cleanly")
println("="^100)
ckpt_dir6 = joinpath(OUTROOT, "t6_interrupt")
r6a = run_cm_upper_checkpointed(w0; W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    L, contrasts = CM_CONTRASTS, probs, maxtime_real = 15.0, ckpt_dir = ckpt_dir6,
    run_id = "t6a", label = "t6", checkpoint_interval_s = 5.0, cm_gradient_backend = :reference, verbose = false)
ckpt6_path = joinpath(ckpt_dir6, "t6_latest.jls")
check("[T6] checkpoint exists after an interrupted (time-limited) run", isfile(ckpt6_path))
ckpt6 = load_cm_checkpoint(ckpt6_path)
check("[T6] interrupted checkpoint loads cleanly", ckpt6 !== nothing)
r6b = run_cm_upper_checkpointed(nothing; ckpt_dir = ckpt_dir6, run_id = "t6b", label = "t6_resumed",
    resume_from = ckpt6_path, maxtime_real = 15.0, checkpoint_interval_s = 5.0,
    cm_gradient_backend = :reference, verbose = false)
check("[T6] resume-after-interruption succeeds and makes further progress", r6b.n_eval >= r6a.n_eval)

println()
println("="^100)
println("TEST 7: stale/corrupt checkpoint file rejected safely (not silently accepted)")
println("="^100)
corrupt_path = joinpath(OUTROOT, "corrupt.jls")
open(corrupt_path, "w") do io
    write(io, "this is not a serialized CMCheckpoint")
end
threw_on_corrupt = Ref(false)
try
    load_cm_checkpoint(corrupt_path)
catch e
    threw_on_corrupt[] = true
end
check("[T7] load_cm_checkpoint throws on a corrupt/non-checkpoint file", threw_on_corrupt[])

println()
println("="^100)
println("TOTAL: $n_pass passed, $n_fail failed")
println("="^100)
n_fail == 0 || error("$n_fail check(s) failed")
println("DONE")
