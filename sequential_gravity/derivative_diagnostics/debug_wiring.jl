ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

find_smallest = false
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
@printf("x_free0 = %s\n", x_free0)

# Solve the inner problem at theta=θinit directly (this ALSO populates obj.H via obj.moments! = m!,
# and updates lastθ_st/lastok_st/gcol/etc as a side effect, exactly as inner_loop_internal does).
δstar, x_star, nStatus_inner = inner_loop(obj, θinit)
@printf("inner solve at theta_init: delta*=%.8f nStatus=%d\n", δstar, nStatus_inner)
@printf("lastok_st[]=%s  lastθ_st[]==θinit? %s  lastRcol_st[]=%.4e\n", lastok_st[], lastθ_st[] == θinit, lastRcol_st[])
@printf("dRdθ_st[] = %s\n", dRdθ_st[])
@printf("gcol_st[][1:5] = %s\n", gcol[][1:5])

g_ad = zeros(length(x_free0))
fn_ad = make_seq_div_grad_fn!(obj, fpmap)
fn_ad(g_ad, x_free0, θinit, x_star)
@printf("\ng_free (pointwise_ad) = %s\n", g_ad)

g_full = zeros(length(x_free0))
fn_full = make_seq_div_grad_fn_full!(obj, fpmap, γ, U, D, gcol, lastθ_st, lastRcol_st, dRdθ_st, lastok_st, :fixed_dual_fd_full)
fn_full(g_full, x_free0, θinit, x_star)
@printf("g_free (fixed_dual_fd_full) = %s\n", g_full)

println("\nDONE debug_wiring")
