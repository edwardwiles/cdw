ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = "3"
ENV["DVAL"] = "20"
ENV["WVAL"] = "80000"
ENV["PARALLEL_INVERSION"] = "true"
ENV["REAL_DATA_DIR"] = joinpath(@__DIR__, "..", "..", "real_data", "noah_D20")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

find_smallest = true   # upper bound
δ = 1.0
θinit = copy(θr0)

d = D + 2; oci = d + 1
m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ_st, lastRcol_st, dRdθ_st, lastok_st =
    make_stateful_moments(; use_exact_grad=true, find_smallest=find_smallest, δ=δ)
obj = CS.PsiObjectiveBundleImplicitMethodB(δ=δ, find_smallest=find_smallest, γ=γ,
    (moments!)=m!, moments_jacobian! = error, d=d, outer_constr_index=oci,
    inequality_index=Int64[], complement_index=[0 0], l=length(θinit), U=U, N=JacW,
    lower_limit=-50, use_cached_x=false, outer_loop_opt=OUTER_OPT_FILE, inner_loop_opt=INNER_OPT_FILE)

my_free_idx = vcat(3, collect(4:3+D)); my_fixed_idx = [1, 2]; my_fixed_vals = θinit[my_fixed_idx]
fpmap = CS.FreeParamMap(length(θinit), my_free_idx, my_fixed_idx, my_fixed_vals)
x_free0 = CS.pack_free(θinit, fpmap)

δstar, x_star, nStatus_inner = inner_loop(obj, θinit)
@printf("inner solve at theta_init: delta*(H_save, =gamma' for MethodB, ignore)=%.6f nStatus=%d\n", δstar, nStatus_inner)
@printf("lastok_st[]=%s  R_mean=%.4e\n", lastok_st[], lastRcol_st[])

g_ad = zeros(length(x_free0))
fn_ad = make_seq_div_grad_fn!(obj, fpmap)
fn_ad(g_ad, x_free0, θinit, x_star)
@printf("\ng_free (pointwise_ad), gamma'-component and first 5 Acol components:\n%s\n", g_ad[1:min(6,end)])

g_full = zeros(length(x_free0))
fn_full = make_seq_div_grad_fn_full!(obj, fpmap, γ, U, D, gcol, lastθ_st, lastRcol_st, dRdθ_st, lastok_st, :fixed_dual_fd_full)
t0 = time()
fn_full(g_full, x_free0, θinit, x_star)
@printf("FD gradient wall time: %.3fs\n", time()-t0)
@printf("g_free (fixed_dual_fd_full), gamma'-component and first 5 Acol components:\n%s\n", g_full[1:min(6,end)])

@printf("\nmax|g_ad - g_full| over Acol block = %.6e\n", maximum(abs.(g_ad[2:end] .- g_full[2:end])))
@printf("max|g_ad| over Acol block = %.6e,  relative diff = %.4e\n", maximum(abs.(g_ad[2:end])), maximum(abs.(g_ad[2:end] .- g_full[2:end]))/maximum(abs.(g_ad[2:end])))
@printf("g_ad[2:end]   = %s\n", g_ad[2:end])
@printf("g_full[2:end] = %s\n", g_full[2:end])
