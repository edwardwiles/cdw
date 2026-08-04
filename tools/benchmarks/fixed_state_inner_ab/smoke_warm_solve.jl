include(joinpath(@__DIR__, "cold_solve_pair.jl"))
using Printf

family = :unrestricted
sci = mode_a_scientific_manifest()
setup = build_reduced_setup(family, sci)
ctx = setup.ctx

cold_dir_r = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/cold_solve_runs/reduced_unrestricted_mode_a_P0"
cold_dir_f = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/cold_solve_runs/full_unrestricted_mode_a_P0"
warm_scratch = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/warm_solve_smoke"

println("REDUCED warm solve from own cold checkpoint..."); flush(stdout)
r_warm = reduced_warm_solve(setup, cold_dir_r; delta = 1.0, maxtime_real = 30.0, threaded_gradient = true,
    ckpt_dir = joinpath(warm_scratch, "reduced_unrestricted_P0"), run_id = "warm_smoke_reduced")
@printf("REDUCED warm: status=%d n_eval=%d n_grad=%d wall=%.1fs best=%s\n",
    r_warm.knitro_status, r_warm.n_eval, r_warm.n_grad, r_warm.wall,
    r_warm.best === nothing ? "none" : "gp=$(r_warm.best.gp) Delta=$(r_warm.best.Delta)")

println("FULL warm solve from own cold checkpoint..."); flush(stdout)
r_warm_f = full_warm_solve(family, ctx, cold_dir_f; delta = 1.0, maxtime_real = 30.0,
    ckpt_dir = joinpath(warm_scratch, "full_unrestricted_P0"), sci = sci)
@printf("FULL warm: status=%d n_eval=%d n_grad=%d wall=%.1fs best=%s\n",
    r_warm_f.knitro_status, r_warm_f.n_eval, r_warm_f.n_grad, r_warm_f.wall,
    r_warm_f.best === nothing ? "none" : string(r_warm_f.best))
