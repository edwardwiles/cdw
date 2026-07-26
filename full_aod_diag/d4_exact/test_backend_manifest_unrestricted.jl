# ============================================================================
# Public entry-point backend manifest assertion, unrestricted family (allocation/Hessian port
# task §3). Companion: test_backend_manifest_cm_originzc.jl (CM/CM+meanZC/origin-ZC).
#
# "Prove that the public driver actually reaches the intended implementation" -- calls the REAL
# public checkpointed driver (run_polish_checkpointed) with a short maxtime_real budget and
# asserts on the actual "[backend-manifest] ..." lines its own stdout produces, not on
# resolve_unrestricted_manifest called in isolation.
#
# Real D=20/W=80,000 context (same convention as test_checkpoint_resume_regression.jl), short
# maxtime_real to keep this bounded.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/test_backend_manifest_unrestricted.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization, LinearAlgebra

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

"Runs `f()`, capturing everything it prints to stdout, and returns (result, captured_text)."
function capture_stdout(f)
    old = stdout
    rd, wr = redirect_stdout()
    result = try
        f()
    finally
        redirect_stdout(old)
        close(wr)
    end
    text = read(rd, String)
    close(rd)
    print(text)   # still show it in this test's own log
    return result, text
end

ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)

OUTDIR = mktempdir()

println("="^78); println("run_polish_checkpointed (unrestricted upper)"); println("="^78)
_, txt = capture_stdout() do
    run_polish_checkpointed("assert_u", true, g0, zfree0; maxtime_real = 5.0, W_in = 80_000,
        delta_in = 1.0, draw_seed_in = 20260719, destination_sample = :exclude_row,
        ckpt_dir = joinpath(OUTDIR, "u"), checkpoint_interval_s = 9999.0)
end
check("prints [backend-manifest] family=unrestricted", occursin("[backend-manifest] family=unrestricted", txt))
# 2026-07-25 continuation: was asserting the PRE-PORT literal (`dense_exact`) -- the manifest print
# call sites in c10_d20_production_driver.jl were hardcoding `hessian_backend = :dense_exact`
# regardless of what actually ran, a real bug this session found and fixed (now reads the live
# UNRESTRICTED_CORE_HESSIAN_BACKEND[] default). Updated to assert the shared winner-pair backend
# this family now actually resolves to by default.
check("unrestricted.hessian_backend=exact_winner_pair_parallel", occursin("hessian_backend=exact_winner_pair_parallel", txt))
check("unrestricted.core_hessian_backend=exact_winner_pair_parallel", occursin("core_hessian_backend=exact_winner_pair_parallel", txt))
# final-gate continuation 2026-07-25 (task §3): workers now resolves dynamically
# (resolve_core_hessian_workers_default()) instead of a hard-coded 10 -- assert against the same
# function the manifest itself calls, so this test tracks whatever thread count it's actually run
# under (e.g. -t 4 here resolves to 4, -t 20 resolves to 20) rather than a stale literal.
check("unrestricted.core_hessian_workers=$(resolve_core_hessian_workers_default())",
      occursin("core_hessian_workers=$(resolve_core_hessian_workers_default())", txt))
check("unrestricted.core_hessian_storage=full_stride", occursin("core_hessian_storage=full_stride", txt))
check("unrestricted.checkpoint_schema=4", occursin("checkpoint_schema=4", txt))
check("unrestricted.core_moment_representation=compressed", occursin("core_moment_representation=compressed", txt))
check("unrestricted screen_stack has 5 screens", occursin("screen_stack=pairwise_certificate,screen_hard_winners,envelope", txt))

println("="^78)
println("run_profile_checkpointed (unrestricted profile/lower-direction entry point)")
println("="^78)
_, txt2 = capture_stdout() do
    run_profile_checkpointed("assert_u_profile", g0, true, zfree0; maxtime_real = 5.0, W_in = 80_000,
        delta_in = 1.0, draw_seed_in = 20260719, destination_sample = :exclude_row,
        ckpt_dir = joinpath(OUTDIR, "u_profile"), checkpoint_interval_s = 9999.0)
end
check("run_profile_checkpointed also prints [backend-manifest] family=unrestricted", occursin("[backend-manifest] family=unrestricted", txt2))

println("="^78)
println(isempty(FAILURES) ? "ALL UNRESTRICTED BACKEND ASSERTIONS PASSED" :
        "FAILURES ($(length(FAILURES))): " * join(FAILURES, "; "))
println("="^78)
isempty(FAILURES) || error("test_backend_manifest_unrestricted.jl: $(length(FAILURES)) assertion(s) failed")
