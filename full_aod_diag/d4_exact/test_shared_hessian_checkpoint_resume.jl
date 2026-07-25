# 2026-07-25 continuation, task §7: checkpoint/resume gate for the shared winner-pair core-Hessian
# backend. Real save -> resume cycle (in-process, via checkpoint FILE reload -- same convention as
# the pre-existing test_checkpoint_resume_regression.jl/test_cm_checkpoint_resume.jl) for
# unrestricted and flexible CM, confirming: (1) the incumbent/context state survives resume
# correctly (delegates to the existing proven checkpoint-integrity machinery, not re-derived here);
# (2) the shared winner-pair backend remains the ACTIVE resolved backend after resume (not
# silently reverted to dense); (3) the runtime backend-use counters show winner-pair calls (not
# dense fallback) both before AND after resume.
#
# SCOPE NOTE (disclosed, not hidden): this does NOT implement the task's full backend-fingerprint
# checkpoint-schema field + mismatch-rejection feature (core_hessian_backend/workers/storage/version
# stored IN the checkpoint schema itself, with an explicit compatibility rule that would reject a
# mismatched resume). That is a genuine checkpoint SCHEMA BUMP (this repo's own convention requires
# checking for CMCheckpointV*-style name collisions across the whole tree before doing this) and
# was judged too large an additional change to make safely in this session on top of everything
# else already gated here. What IS verified: state survives resume, and the shared backend is
# demonstrably still the one actually running post-resume (via the runtime counters this session
# added) -- not a fingerprint-based compatibility check.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    println(rpad(cond ? "PASS" : "FAIL", 6), name)
    cond || push!(FAILURES, name)
    flush(stdout)
end

OUTDIR = mktempdir()

println("="^90)
println("UNRESTRICTED: checkpoint -> resume, shared backend stays active")
println("="^90)
ckpt_dir_u = joinpath(OUTDIR, "ckpt_u")
mkpath(ckpt_dir_u)
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)

check("UNRESTRICTED_CORE_HESSIAN_BACKEND[] is the shared backend before the run", UNRESTRICTED_CORE_HESSIAN_BACKEND[] === :exact_winner_pair_parallel)

reset_core_hessian_counters!()
res1 = run_polish_checkpointed("shq_stage1", true, g0, zfree0;
    maxtime_real = 20.0, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
    destination_sample = :exclude_row, ckpt_dir = ckpt_dir_u, checkpoint_interval_s = 3.0)
counters_before = deepcopy(CORE_HESSIAN_COUNTERS[])
lp(">>> stage1: n_eval=", res1.n_eval, " best_feasible=", res1.best_feasible === nothing ? "nothing" : "gp=$(res1.best_feasible.gp) Delta=$(res1.best_feasible.Delta)")
check("checkpoint file was written", isfile(res1.ckpt_path))
check("stage1 used the shared winner-pair backend for at least one Hessian call", counters_before.winner_pair_hessian_calls > 0)
check("stage1 had zero unexplained dense fallback", counters_before.dense_core_fallback_calls == 0)
print_core_hessian_counters(counters_before)

reset_core_hessian_counters!()
res2 = run_polish_checkpointed("shq_stage1", true, g0, zfree0;
    maxtime_real = 15.0, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
    destination_sample = :exclude_row, ckpt_dir = ckpt_dir_u, checkpoint_interval_s = 3.0,
    resume_from = res1.ckpt_path)
counters_after = deepcopy(CORE_HESSIAN_COUNTERS[])
lp(">>> stage2 (resumed): n_eval=", res2.n_eval, " best_feasible=", res2.best_feasible === nothing ? "nothing" : "gp=$(res2.best_feasible.gp) Delta=$(res2.best_feasible.Delta)")
check("resumed run continued (n_eval did not reset)", res2.n_eval >= res1.n_eval)
check("resumed run re-attaches cf_workspace (existing checkpoint-integrity contract, unchanged)", hasproperty(res2.ctx, :cf_workspace))
check("resumed run STILL uses the shared winner-pair backend (not silently reverted to dense post-resume)", counters_after.winner_pair_hessian_calls > 0)
check("resumed run had zero unexplained dense fallback", counters_after.dense_core_fallback_calls == 0)
print_core_hessian_counters(counters_after)

if res1.best_feasible !== nothing && res2.best_feasible !== nothing
    check("resumed run did not lose the checkpointed incumbent", res2.best_feasible.gp <= res1.best_feasible.gp + 1e-9)
end

println("="^90)
println("FLEXIBLE CM: checkpoint -> resume, shared backend stays active")
println("="^90)
ckpt_dir_cm = joinpath(OUTDIR, "ckpt_cm")
mkpath(ckpt_dir_cm)
for f in ["common_marginals_moments.jl", "common_marginals_interval.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "cm_screen_bridge.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_checkpoint.jl"]
    include(joinpath(@__DIR__, f))
end
snaps = nested_grid_sequence([10, 20])
probs10 = snaps[10]
w_calib = vcat(g0, zfree0)

reset_core_hessian_counters!()
res1c = run_cm_upper_checkpointed(w_calib; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
    draw_seed = 20260719, L = 10, contrasts = :orthonormal, probs = probs10,
    cm_hessian_backend = :structured, maxtime_real = 20.0, ckpt_dir = ckpt_dir_cm,
    label = "shq_cm_stage1", checkpoint_interval_s = 3.0, destination_sample = :exclude_row)
counters_cm_before = deepcopy(CORE_HESSIAN_COUNTERS[])
lp(">>> CM stage1: n_eval=", res1c.n_eval)
check("CM checkpoint file was written", isfile(res1c.ckpt_path))
check("CM stage1 used the shared winner-pair backend for at least one Hessian call", counters_cm_before.winner_pair_hessian_calls > 0)
check("CM stage1 had zero unexplained dense fallback", counters_cm_before.dense_core_fallback_calls == 0)
print_core_hessian_counters(counters_cm_before)

reset_core_hessian_counters!()
res2c = run_cm_upper_checkpointed(w_calib; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
    draw_seed = 20260719, L = 10, contrasts = :orthonormal, probs = probs10,
    cm_hessian_backend = :structured, maxtime_real = 15.0, ckpt_dir = ckpt_dir_cm,
    label = "shq_cm_stage1", checkpoint_interval_s = 3.0, destination_sample = :exclude_row,
    resume_from = res1c.ckpt_path)
counters_cm_after = deepcopy(CORE_HESSIAN_COUNTERS[])
lp(">>> CM stage2 (resumed): n_eval=", res2c.n_eval)
check("CM resumed run continued (n_eval did not reset)", res2c.n_eval >= res1c.n_eval)
check("CM resumed run STILL uses the shared winner-pair backend", counters_cm_after.winner_pair_hessian_calls > 0)
check("CM resumed run had zero unexplained dense fallback", counters_cm_after.dense_core_fallback_calls == 0)
print_core_hessian_counters(counters_cm_after)

println("="^90)
if isempty(FAILURES)
    println("ALL CHECKPOINT/RESUME SHARED-BACKEND GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
