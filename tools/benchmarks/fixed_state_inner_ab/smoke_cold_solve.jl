include(joinpath(@__DIR__, "cold_solve_pair.jl"))
using Printf

family = Symbol(ARGS[1])
maxtime_real = parse(Float64, ARGS[2])

sci = mode_a_scientific_manifest()
println("Building REDUCED setup for $family ..."); flush(stdout)
setup = build_reduced_setup(family, sci)
ctx = setup.ctx

D = ctx.D
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gp0 = θ0[3+D]
w0 = reduce_to_w_profiled(gp0, z_calib, setup.pe)

scratch = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/cold_solve_smoke"

println("REDUCED cold solve, $family, P0 ..."); flush(stdout)
t1 = @elapsed r_reduced = reduced_cold_solve(setup, w0; delta = 1.0, maxtime_real = maxtime_real,
    threaded_gradient = true, ckpt_dir = joinpath(scratch, "reduced_$(family)_P0"), run_id = "smoke_$(family)")
@printf("REDUCED: status=%d n_eval=%d n_grad=%d wall=%.1fs best=%s\n",
    r_reduced.knitro_status, r_reduced.n_eval, r_reduced.n_grad, r_reduced.wall,
    r_reduced.best === nothing ? "none" : "gp=$(r_reduced.best.gp) Delta=$(r_reduced.best.Delta)")

println("FULL cold solve, $family, P0 ..."); flush(stdout)
t2 = @elapsed r_full = full_cold_solve(family, gp0, z_calib, ctx; delta = 1.0, maxtime_real = maxtime_real,
    ckpt_dir = joinpath(scratch, "full_$(family)_P0"), sci = sci, meanzc_K_mean = sci.K_mean)
@printf("FULL: status=%d n_eval=%d n_grad=%d wall=%.1fs best=%s\n",
    r_full.knitro_status, r_full.n_eval, r_full.n_grad, r_full.wall,
    r_full.best === nothing ? "none" : string(r_full.best))

println("Elapsed: reduced=$(round(t1,digits=1))s full=$(round(t2,digits=1))s")
