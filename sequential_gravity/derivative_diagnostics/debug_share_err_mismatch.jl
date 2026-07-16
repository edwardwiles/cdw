ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra, JLD2

const OUT_DIR = joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull_scaled05")
d = JLD2.load(joinpath(OUT_DIR, "seq_upper_delta5.0.jld2"))   # worst case: 3.29e-3
θsol = d["best_feasible_theta"]

col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
@printf("fresh re-solve: ok=%s R_mean=%.4e\n", ok, R)

log_x = build_log_x(Uσ, θsol[1])
logp = log.(p)

# worst destination from the earlier run was #3
dd = 3
model_shares, _ = dest_share(log_x, logp, umat[:, dd]; ρ=0.0)
err_myrecompute = maximum(abs.(model_shares .- λData[:, dd]))
@printf("\n[MY dest_share recompute] destination %d: max share err = %.4e\n", dd, err_myrecompute)
@printf("model_shares  = %s\n", model_shares)
@printf("lambdaData[:,%d] = %s\n", dd, λData[:, dd])
@printf("diff          = %s\n", model_shares .- λData[:, dd])

# Now independently re-run invert_destination FRESH (tight tol) at the SAME (log_x, p, target),
# warm-started from the current umat[:,dd], to see what IT converges to, and what ITS OWN
# max_abs_share_error says, completely independent of my dest_share-based recompute above.
inv_fresh = invert_destination(log_x, p, λData[:, dd]; ref=ref, ρ=ρ, tol=1e-10, maxit=300, ls_iters=100, u_init=umat[:, dd])
@printf("\n[fresh invert_destination, tol=1e-10, warm from current umat] converged=%s iters=%d max_abs_share_error=%.4e\n",
    inv_fresh.converged, inv_fresh.iterations, inv_fresh.max_abs_share_error)
@printf("u_full (fresh) vs umat[:,%d] (from seq_gravcol) max|diff| = %.4e\n", dd, maximum(abs.(inv_fresh.u_full .- umat[:, dd])))

# And what does invert_destination's OWN model_shares say (its internal accounting) vs mine?
@printf("inv_fresh.model_shares = %s\n", inv_fresh.model_shares)
@printf("my dest_share(inv_fresh.u_full) = %s\n", dest_share(log_x, logp, inv_fresh.u_full; ρ=0.0)[1])

# Also check: is DEST_INV_TOL actually being honored inside seq_gravcol's OWN invert_all calls?
@printf("\nDEST_INV_TOL (production's own tolerance) = %.2e\n", DEST_INV_TOL)
