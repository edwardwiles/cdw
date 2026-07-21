# Continuation 10, Section 6 validation: "resumed" run -- a FRESH Julia
# process (this script is invoked as a SEPARATE `julia` run, not a
# continuation of c10_prod_driver_smoke_original.jl's process) loads the
# latest checkpoint from that prior run and validates reproducibility.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))

CKPT_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", "c10_ckpt_smoke_test")
latest_path = joinpath(CKPT_DIR, "smoke_lower_latest.jls")   # addendum: c10_prod_driver_smoke_original.jl's own label was corrected from "smoke_upper" to "smoke_lower" (it passes find_smallest=false, the real lower direction) -- path updated to match
@assert isfile(latest_path) "no checkpoint found at $latest_path -- run c10_prod_driver_smoke_original.jl first"

latest = load_checkpoint(latest_path)
println("Loaded checkpoint: reason=", latest.checkpoint_reason, " n_eval=", latest.n_eval,
        " knitro_iter=", latest.knitro_iter, " draw_seed=", latest.draw_seed, " W=", latest.W)
println("  original verify_Delta_dual=", latest.verify_Delta_dual, " verify_gravity_value=", latest.verify_gravity_value,
        " verify_max_abs_moment_kkt_resid=", latest.verify_max_abs_moment_kkt_resid,
        " verify_moment_resid_norm=", latest.verify_moment_resid_norm)

println("\n=== RESUMED RUN: run_profile_checkpointed(resume_from=checkpoint), tiny budget ===")
CKPT_DIR2 = joinpath(D4X_ROOT, "results", "fullA_d4", "c10_ckpt_smoke_test_resumed")
rm(CKPT_DIR2; recursive = true, force = true)
mkpath(CKPT_DIR2)
res2 = run_profile_checkpointed("smoke_upper", NaN, true, Float64[];   # g/find_smallest/zfree overridden by resume_from
    maxtime_real = 5.0, ckpt_dir = CKPT_DIR2, checkpoint_interval_s = 200.0,
    resume_from = latest_path)

println("\nDONE_RESUME (see 'RESUME VALIDATION' lines above for the acceptance-test diffs)")
