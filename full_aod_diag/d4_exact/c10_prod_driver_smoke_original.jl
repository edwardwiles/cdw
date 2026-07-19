# Continuation 10, Section 6 validation: "original" (unbroken) run -- short
# profile_checkpointed run at real D=20/W=80000, producing checkpoints.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))

ctx_probe = d20_real_setup(W = 80000, find_smallest = true)
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D; D2 = D^2
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe_probe)
gp0 = ctx_probe.θ0_up[3+D]

CKPT_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", "c10_ckpt_smoke_test")
rm(CKPT_DIR; recursive = true, force = true)
mkpath(CKPT_DIR)

println("=== ORIGINAL RUN: run_profile_checkpointed, short budget ===")
res = run_profile_checkpointed("smoke_upper", gp0 * 1.01, false, zfree0;
    maxtime_real = 90.0, W_in = 80000, delta_in = 1.0, draw_seed_in = 20260719,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 15.0)

println("\n=== ORIGINAL RUN SUMMARY ===")
println("n_eval=", res.n_eval, " wall_ext=", res.wall_ext, " screen_counts=", res.screen_counts)
println("best=", res.best)
println("ckpt_path=", res.ckpt_path)
latest = load_checkpoint(res.ckpt_path)
println("latest checkpoint: reason=", latest.checkpoint_reason, " n_eval=", latest.n_eval,
        " knitro_iter=", latest.knitro_iter, " g=", latest.g,
        " verify_Delta_dual=", latest.verify_Delta_dual, " verify_gravity_value=", latest.verify_gravity_value)

# list all checkpoints written
for f in sort(readdir(CKPT_DIR))
    println("  ckpt file: ", f)
end
println("DONE_ORIGINAL")
