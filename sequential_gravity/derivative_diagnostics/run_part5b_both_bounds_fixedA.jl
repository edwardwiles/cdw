# ============================================================================
# Part 5b: extend Part 5's comparison to BOTH bounds (lower AND upper, delta=1),
# and add the "fixed-A efficiency" comparison the user asked for: for each
# gradient_method's result (gamma'_focal_result), compute the divergence
# delta*_fixedA that would be NEEDED to reach that SAME gamma'_focal (same GT)
# if A were held fixed at A* the whole time (Acol == theta_r0's own Acol,
# i.e. the Frechet benchmark) -- via a direct inner CC solve at A=A*,
# gamma'=gamma'_focal_result (D+1 baseline moments, no outer search at all).
#
# Interpretation: if delta*_fixedA > 1 (the actual budget spent), moving A
# reached a GT that fixed-A COULD NOT reach within the same budget -- a
# genuine efficiency gain, quantified in "extra effective delta bought by
# moving A" = delta*_fixedA - 1. If delta*_fixedA <= 1, moving A bought
# nothing (fixed-A could already reach that GT within budget, or reaches it
# more cheaply).
#
#   DVAL=4 julia --project=. sequential_gravity/derivative_diagnostics/run_part5b_both_bounds_fixedA.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["DVAL"] = get(ENV, "DVAL", "4")
ENV["DELTA_GRID"] = get(ENV, "DELTA_GRID", "1.0")

include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf

function outer_solve_plain(find_smallest, θinit, method::Symbol; δ::Real=δ)
    d = D + 2; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm = make_stateful_moments(; use_exact_grad=true, find_smallest=find_smallest, δ=δ)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ=δ, find_smallest=find_smallest, γ=γ,
        (moments!) = m!, moments_jacobian! = error, d=d, outer_constr_index=oci,
        inequality_index=Int64[], complement_index=[0 0], l=length(θinit), U=U, N=JacW,
        lower_limit=-50, use_cached_x=false,
        outer_loop_opt=OUTER_OPT_FILE, inner_loop_opt=INNER_OPT_FILE)
    l_full = length(θinit)
    free_idx_ = vcat(3, collect(4:3+D)); fixed_idx_ = [1, 2]; fixed_vals_ = θinit[fixed_idx_]
    fpmap = CS.FreeParamMap(l_full, free_idx_, fixed_idx_, fixed_vals_)
    div_grad_fn! = method == :pointwise_ad ? make_seq_div_grad_fn!(obj, fpmap) :
        make_seq_div_grad_fn_corrected!(obj, fpmap, γ, U, D, (1 / θinit[1]) / (σ - 1), method)
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0); g_free[1] = (-1.0)^find_smallest
    end
    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = OUTER_OPT_FILE)
    return r.θ_min_full[3], r.θ_min_full, r.nStatus, r.cache
end

"delta*(A=A*, gamma'_focal=gp) via a direct D+1-moment inner CC solve, NO outer search, A held at theta_r0's own Acol."
function delta_star_fixed_A(gp::Float64)
    θ = copy(θr0); θ[3] = gp
    obj = build_fixed_dual_bundle(γ, U, length(θ), D + 1, EK_moments_focal_norm_directgp!)
    val, x, nStatus = inner_loop(obj, θ)
    return val, nStatus
end

results = NamedTuple[]
for (bound_name, find_smallest) in ((:lower, false), (:upper, true))
    for method in (:pointwise_ad, :fixed_dual_fd, :boundary)
        println("\n" * "="^78); @printf(">>> bound=%s  gradient_method=%s  (D=%d, delta=%g)\n", bound_name, method, D, δ); println("="^78)
        t0 = time()
        gp, θstar, nStatus, cache = outer_solve_plain(find_smallest, copy(θr0), method; δ=δ)
        wall = time() - t0
        κ = gp2kappa(gp)
        _, Rθ, _, _, _, okθ = seq_gravcol(θstar; δ=δ)
        Acol0 = θr0[4:3+D]; Acol_star = θstar[4:3+D]
        dAcol = norm(Acol_star .- Acol0) / norm(Acol0)
        δ_fixedA, status_fixedA = delta_star_fixed_A(gp)
        @printf("gamma'_focal=%.6f -> kappa=%.6f  status=%d  ||dAcol||/||Acol0||=%.4e  R_mean=%.3e  gravity_ok=%s  wall=%.1fs\n",
            gp, κ, nStatus, dAcol, Rθ, okθ, wall)
        @printf("  fixed-A(A*) delta* needed for the SAME gamma'_focal = %.6f (status=%d)  vs actual budget delta=%.4g  ->  extra effective delta bought by moving A = %.6f\n",
            δ_fixedA, status_fixedA, δ, δ_fixedA - δ)
        push!(results, (bound=bound_name, method=method, gp=gp, kappa=κ, nStatus=nStatus, rel_dAcol=dAcol,
            R_mean=Rθ, gravity_ok=okθ, wall=wall, n_inner_solve=cache.n_inner_solve,
            delta_fixedA=δ_fixedA, status_fixedA=status_fixedA, extra_delta=δ_fixedA - δ))
    end
end

println("\n" * "="^78); println(">>> PART 5b SUMMARY: both bounds x 3 gradient methods, vs fixed-A efficiency"); println("="^78)
@printf("%6s %15s %10s %10s %8s %14s %10s %14s %12s\n", "bound", "method", "gamma_p", "kappa", "status", "rel||dAcol||", "R_mean", "delta*_fixedA", "extra_delta")
for r in results
    @printf("%6s %15s %10.6f %10.6f %8d %14.4e %10.3e %14.6f %12.6f\n",
        r.bound, r.method, r.gp, r.kappa, r.nStatus, r.rel_dAcol, r.R_mean, r.delta_fixedA, r.extra_delta)
end

open(joinpath(@__DIR__, "part5b_both_bounds_fixedA_D$(D)_delta$(δ).csv"), "w") do io
    println(io, "bound,method,gamma_p,kappa,nStatus,rel_dAcol,R_mean,gravity_ok,wall,n_inner_solve,delta_fixedA,status_fixedA,extra_delta")
    for r in results
        println(io, "$(r.bound),$(r.method),$(r.gp),$(r.kappa),$(r.nStatus),$(r.rel_dAcol),$(r.R_mean),$(r.gravity_ok),$(r.wall),$(r.n_inner_solve),$(r.delta_fixedA),$(r.status_fixedA),$(r.extra_delta)")
    end
end

println("\nPART 5b DONE")
