ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = "3"
ENV["DVAL"] = "20"
ENV["WVAL"] = "80000"
ENV["PARALLEL_INVERSION"] = "true"
ENV["REAL_DATA_DIR"] = joinpath(@__DIR__, "..", "..", "real_data", "noah_D20")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, JLD2

d_new = JLD2.load(joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull", "seq_upper_delta1.0.jld2"))
θconv = d_new["theta_star"]
@printf("Saved R_mean_at_solution (from the actual run) = %.6e\n", d_new["R_mean_at_solution"])
@printf("Saved gravity_feasible = %s\n", d_new["gravity_feasible"])

@printf("\nDirect fresh seq_gravcol(theta_converged; delta=1.0), no warm start:\n")
col, R, Rcol, umat, p, ok = seq_gravcol(θconv; δ=1.0, verbose=true)
@printf("R_mean=%.6e  ok=%s\n", R, ok)
