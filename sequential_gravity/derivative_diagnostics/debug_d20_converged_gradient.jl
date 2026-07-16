ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = "3"
ENV["DVAL"] = "20"
ENV["WVAL"] = "80000"
ENV["PARALLEL_INVERSION"] = "true"
ENV["REAL_DATA_DIR"] = joinpath(@__DIR__, "..", "..", "real_data", "noah_D20")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra, JLD2

d_new = JLD2.load(joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull", "seq_upper_delta1.0.jld2"))
θconv = d_new["theta_star"]
@printf("Converged theta[3] (gamma')=%.6f\n", θconv[3])
@printf("rel‖Acol_converged - Acol_init‖ = %.6f\n", norm(θconv[4:end] .- θr0[4:end]) / norm(θr0[4:end]))

find_smallest = true
δ = 1.0
d = D + 2; oci = d + 1
m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ_st, lastRcol_st, dRdθ_st, lastok_st =
    make_stateful_moments(; use_exact_grad=true, find_smallest=find_smallest, δ=δ)
obj = CS.PsiObjectiveBundleImplicitMethodB(δ=δ, find_smallest=find_smallest, γ=γ,
    (moments!)=m!, moments_jacobian! = error, d=d, outer_constr_index=oci,
    inequality_index=Int64[], complement_index=[0 0], l=length(θconv), U=U, N=JacW,
    lower_limit=-50, use_cached_x=false, outer_loop_opt=OUTER_OPT_FILE, inner_loop_opt=INNER_OPT_FILE)

my_free_idx = vcat(3, collect(4:3+D)); my_fixed_idx = [1, 2]; my_fixed_vals = θconv[my_fixed_idx]
fpmap = CS.FreeParamMap(length(θconv), my_free_idx, my_fixed_idx, my_fixed_vals)
x_free_conv = CS.pack_free(θconv, fpmap)

δstar, x_star, nStatus_inner = inner_loop(obj, θconv)
@printf("inner solve at theta_converged: nStatus=%d  lastok_st[]=%s  R_mean=%.4e\n", nStatus_inner, lastok_st[], lastRcol_st[])

g_ad = zeros(length(x_free_conv))
fn_ad = make_seq_div_grad_fn!(obj, fpmap)
fn_ad(g_ad, x_free_conv, θconv, x_star)

g_full = zeros(length(x_free_conv))
fn_full = make_seq_div_grad_fn_full!(obj, fpmap, γ, U, D, gcol, lastθ_st, lastRcol_st, dRdθ_st, lastok_st, :fixed_dual_fd_full)
fn_full(g_full, x_free_conv, θconv, x_star)

@printf("\nAT THE CONVERGED POINT:\n")
@printf("max|g_ad[Acol]|   = %.6e\n", maximum(abs.(g_ad[2:end])))
@printf("max|g_full[Acol]| = %.6e\n", maximum(abs.(g_full[2:end])))
@printf("max|g_ad-g_full| over Acol = %.6e\n", maximum(abs.(g_ad[2:end] .- g_full[2:end])))
@printf("g_ad[2:end]   = %s\n", g_ad[2:end])
@printf("g_full[2:end] = %s\n", g_full[2:end])
