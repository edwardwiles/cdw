# Step 1 (hard-max inversion task): reconstruct (log_x, p) for Point 1, sanity-check the
# existing rho=2e-3 invert_destination call reproduces lambdaData, and measure the rho=0
# hard-max gap at that solution (the size of the problem this task is trying to close).
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true julia -t 19 --project=. \
#     sequential_gravity/derivative_diagnostics/hardmax_step1_recon.jl
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, JLD2

println("\n" * "="^78); println(">>> STEP 1: reconstruction + sanity check, Point 1"); println("="^78)

d1 = JLD2.load(joinpath(@__DIR__, "..", "head_to_head", "out_lc", "lc_T1_Astar.jld2"))
@assert d1["done"] == true "Point 1 job not marked done"
θ = d1["best_feasible_theta"]
@printf("Point 1: mu=%.6f sigma=%.4f gammap_focal=%.6f  (theta length=%d, D=%d)\n", θ[1], θ[2], θ[3], length(θ), D)
@assert θ[2] == σ "sigma mismatch"

log_x = build_log_x(Uσ, θ[1])
uf = focal_u(θ)
t0 = time()
p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
@printf("recover_lfd: ok=%s  wall=%.1fs  sum(p)=%.10f  min(p)=%.3e  max(p)=%.3e\n",
    ok, time() - t0, sum(p), minimum(p), maximum(p))
@assert ok "recover_lfd failed for Point 1"

dtest = omitted[1]
@printf("\nTesting destination d=%d (focal=%d, omitted=%s...)\n", dtest, focal, string(omitted[1:min(5,end)]))

inv = invert_destination(log_x, p, λData[:, dtest]; ref = ref, ρ = ρ, tol = 1e-6, maxit = 150, ls_iters = 50)
@printf("EXISTING rho=%.4g solver: converged=%s iters=%d max_abs_share_error=%.3e gradient_norm=%.3e\n",
    ρ, inv.converged, inv.iterations, inv.max_abs_share_error, inv.gradient_norm)
@assert inv.converged && inv.max_abs_share_error < 1e-4 "existing solver sanity check failed"

# Focal check (step 1 of the 3-step validation): focal_u reproduces lambdaData under rho=0
logp = log.(p)
focal_shares, _ = dest_share(log_x, logp, uf; ρ = 0.0)
focal_err = maximum(abs.(focal_shares .- λData[:, focal]))
@printf("\nFOCAL check (rho=0 hard-argmin, focal_u(theta)): max|model-empirical|=%.3e\n", focal_err)

# Hard-max gap at the rho=2e-3-converged u (informational, per the handoff prompt's step 2)
hard_shares, _ = dest_share(log_x, logp, inv.u_full; ρ = 0.0)
hard_err = maximum(abs.(hard_shares .- λData[:, dtest]))
@printf("HARD-MAX gap at rho=%.4g solution (destination %d): max|model-empirical|=%.3e  (expected O(rho)=%.0e)\n",
    ρ, dtest, hard_err, ρ)

# Save reconstruction for reuse by later steps
JLD2.save(joinpath(@__DIR__, "hardmax_point1_recon.jld2"),
    "theta", θ, "p", p, "log_x", log_x, "uf", uf, "dtest", dtest,
    "u_rho_solution", inv.u_full, "lambda_dtest", λData[:, dtest])

println("\nSTEP 1 DONE")
