# ============================================================================
# Part 5 driver: run the ACTUAL production sequential outer-loop bound search
# (sequential_gravity/run_profiled_production.jl::outer_solve_nested_cached)
# with gradient_method in {:pointwise_ad, :fixed_dual_fd, :boundary}, and
# compare the real outcomes.
#
# Per-call instrumentation caveat (honest scoping note): this codebase's
# outer_loop_cached/OuterEvalCache does not expose KNITRO's own internal
# trust-region accept/reject bookkeeping per iteration (only aggregate call
# counts, cf. cc_algo/outer_eval_cache.jl::summarize). Rather than patch
# KNITRO's callback internals (out of scope), this driver instruments
# div_grad_fn! itself (norm of the gradient and the step between consecutive
# DISTINCT free-x points KNITRO's own search visits) as an honest, if
# approximate, substitute for a true per-iteration predicted/realized log.
#
#   DVAL=4 julia --project=. sequential_gravity/derivative_diagnostics/run_part5_gradient_method_comparison.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["DVAL"] = get(ENV, "DVAL", "4")
ENV["DELTA_GRID"] = get(ENV, "DELTA_GRID", "1.0")

include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using Printf

const CALL_LOG = Dict{Symbol,Vector{NamedTuple}}()

"Wraps a div_grad_fn! to additionally log (call_idx, x_free copy, norm_gradient, norm_step_from_last_distinct_x)."
function instrumented(method::Symbol, inner_fn!)
    log = get!(CALL_LOG, method, NamedTuple[])
    last_x = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    call_idx = Ref(0)
    return function (g_free, x_free, θ_full, inner_x)
        inner_fn!(g_free, x_free, θ_full, inner_x)
        call_idx[] += 1
        step = last_x[] === nothing ? NaN : norm(x_free .- last_x[])
        if last_x[] === nothing || step > 1e-12
            last_x[] = copy(x_free)
        end
        push!(log, (call=call_idx[], norm_gradient=norm(g_free), norm_step=step))
        return g_free
    end
end

"Same as outer_solve_nested_cached but with the div_grad_fn! wrapped for logging."
function outer_solve_logged(find_smallest, θinit, method::Symbol; δ::Real=δ)
    d = D + 2; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm = make_stateful_moments(; use_exact_grad=true, find_smallest=find_smallest, δ=δ)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ=δ, find_smallest=find_smallest, γ=γ,
        (moments!) = m!, moments_jacobian! = error, d=d, outer_constr_index=oci,
        inequality_index=Int64[], complement_index=[0 0], l=length(θinit), U=U, N=JacW,
        lower_limit=-50, use_cached_x=false,
        outer_loop_opt=OUTER_OPT_FILE, inner_loop_opt=INNER_OPT_FILE)

    l_full = length(θinit)
    free_idx_ = vcat(3, collect(4:3+D))
    fixed_idx_ = [1, 2]
    fixed_vals_ = θinit[fixed_idx_]
    fpmap = CS.FreeParamMap(l_full, free_idx_, fixed_idx_, fixed_vals_)

    raw_div_grad_fn! = method == :pointwise_ad ? make_seq_div_grad_fn!(obj, fpmap) :
        make_seq_div_grad_fn_corrected!(obj, fpmap, γ, U, D, (1 / θinit[1]) / (σ - 1), method)
    div_grad_fn! = instrumented(method, raw_div_grad_fn!)
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0)
        g_free[1] = (-1.0)^find_smallest
    end

    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = OUTER_OPT_FILE)
    gp = r.θ_min_full[3]
    return gp, r.θ_min_full, r.nStatus, best_θ[], best_κ[], best_warm[], r.cache
end

# ============================================================================
# Comparison: lower bound (find_smallest=false, i.e. MAXIMIZE gamma'_focal ->
# minimize kappa) at delta=1, all 3 gradient methods, starting from the SAME
# theta_r0.
# ============================================================================
println("\n" * "="^78); println(">>> PART 5: gradient_method comparison, real outer bound search (D=$D, delta=$δ)"); println("="^78)

results = NamedTuple[]
for method in (:pointwise_ad, :fixed_dual_fd, :boundary)
    println("\n--- gradient_method = $method ---")
    t0 = time()
    gp, θstar, nStatus, bθ, bκ, bwarm, cache = outer_solve_logged(false, copy(θr0), method; δ=δ)
    wall = time() - t0
    κ = gp2kappa(gp)
    _, Rθ, _, _, _, okθ = seq_gravcol(θstar; δ=δ)
    Acol0 = θr0[4:3+D]; Acol_star = θstar[4:3+D]
    dAcol = norm(Acol_star .- Acol0) / norm(Acol0)
    @printf("gamma'_focal=%.6f -> kappa=%.6f  status=%d  ||Acol*-Acol0||/||Acol0||=%.4e  R_mean=%.3e  gravity_ok=%s  wall=%.1fs\n",
        gp, κ, nStatus, dAcol, Rθ, okθ, wall)
    CS.summarize(cache; label="$method cache stats")
    log = CALL_LOG[method]
    @printf("div_grad_fn! calls: %d   final ||gradient||=%.4e   max step seen=%.4e\n",
        length(log), log[end].norm_gradient, maximum(r.norm_step for r in log if isfinite(r.norm_step); init=0.0))
    push!(results, (method=method, gp=gp, kappa=κ, nStatus=nStatus, rel_dAcol=dAcol, R_mean=Rθ, gravity_ok=okθ,
        wall=wall, n_inner_solve=cache.n_inner_solve, n_grad_compute=cache.n_grad_compute, n_calls=length(log)))
end

println("\n" * "="^78); println(">>> SUMMARY"); println("="^78)
@printf("%15s %10s %10s %8s %14s %12s %10s %8s %8s\n", "method", "gamma_p", "kappa", "status", "rel||dAcol||", "R_mean", "grav_ok", "n_inner", "wall(s)")
for r in results
    @printf("%15s %10.6f %10.6f %8d %14.4e %12.3e %10s %8d %8.1f\n",
        r.method, r.gp, r.kappa, r.nStatus, r.rel_dAcol, r.R_mean, r.gravity_ok, r.n_inner_solve, r.wall)
end

open(joinpath(@__DIR__, "part5_comparison_D$(D)_delta$(δ).csv"), "w") do io
    println(io, "method,gamma_p,kappa,nStatus,rel_dAcol,R_mean,gravity_ok,wall,n_inner_solve,n_grad_compute,n_calls")
    for r in results
        println(io, "$(r.method),$(r.gp),$(r.kappa),$(r.nStatus),$(r.rel_dAcol),$(r.R_mean),$(r.gravity_ok),$(r.wall),$(r.n_inner_solve),$(r.n_grad_compute),$(r.n_calls)")
    end
end
for (method, log) in CALL_LOG
    open(joinpath(@__DIR__, "part5_calllog_$(method)_D$(D)_delta$(δ).csv"), "w") do io
        println(io, "call,norm_gradient,norm_step")
        for r in log
            println(io, "$(r.call),$(r.norm_gradient),$(r.norm_step)")
        end
    end
end

println("\nPART 5 DONE")
