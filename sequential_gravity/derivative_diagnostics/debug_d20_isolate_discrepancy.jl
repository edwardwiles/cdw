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

println("=== Step 1: direct seq_gravcol(theta_converged; delta=1.0) ===")
col1, R1, Rcol1, umat1, p1, ok1 = seq_gravcol(θconv; δ=1.0)
@printf("R1=%.6e ok1=%s\n", R1, ok1)

println("\n=== Step 2: via make_stateful_moments + inner_loop (fresh Refs) ===")
find_smallest = true
d = D + 2; oci = d + 1
m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ_st, lastRcol_st, dRdθ_st, lastok_st =
    make_stateful_moments(; use_exact_grad=true, find_smallest=find_smallest, δ=1.0)
obj = CS.PsiObjectiveBundleImplicitMethodB(δ=1.0, find_smallest=find_smallest, γ=γ,
    (moments!)=m!, moments_jacobian! = error, d=d, outer_constr_index=oci,
    inequality_index=Int64[], complement_index=[0 0], l=length(θconv), U=U, N=JacW,
    lower_limit=-50, use_cached_x=false, outer_loop_opt=OUTER_OPT_FILE, inner_loop_opt=INNER_OPT_FILE)
δstar, x_star, nStatus_inner = inner_loop(obj, θconv)
@printf("nStatus_inner=%d  lastRmean[]=%.6e  lastok_st[]=%s\n", nStatus_inner, lastRmean[], lastok_st[])

println("\n=== Step 3: re-check via direct seq_gravcol call AGAIN (after step 2) ===")
col3, R3, Rcol3, umat3, p3, ok3 = seq_gravcol(θconv; δ=1.0)
@printf("R3=%.6e ok3=%s\n", R3, ok3)

@printf("\nR1 (before)=%.6e  lastRmean-from-m!=%.6e  R3 (after)=%.6e\n", R1, lastRmean[], R3)
