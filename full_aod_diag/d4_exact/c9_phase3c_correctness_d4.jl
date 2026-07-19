# ============================================================================
# Continuation 9, Phase 3C: correctness check for all three dense-Hessian-free
# inner-solve variants (compressed_inner_alt_solvers.jl) at D=4, BEFORE any
# D=20 timing. Compares each variant's converged (ζ*, λ*) and Delta_dual
# against the trusted dense solve (solve_base_state / evaluate_fullA), cold
# start, at the calibration point.
#
# All three variants solve the SAME convex problem (the CC inner dual is
# convex in (ζ,λ) -- Psi is convex, G is fixed at this theta), so a DIFFERENT
# solver path converging to the SAME optimum (agreeing objective/duals to
# KNITRO's own optimality tolerance) is the correctness bar here -- not
# bit-identical iterate paths.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))   # -> d4_exact_setup AND d_exact_setup_scaled; includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "compressed_inner_alt_solvers.jl"))
using Printf, LinearAlgebra

ctx = d4_exact_setup(find_smallest = true)
xf_nat = ctx.θ0_up[ctx.free_idx]
obj = ctx.obj
θ_full = CS.reconstruct_full(xf_nat, ctx.m)

# ---- trusted reference: dense solve (oracle.jl::evaluate_fullA) ----
r_dense = evaluate_fullA(xf_nat, ctx; warm = false)
println("dense reference: inner_status=", r_dense.inner_status, "  Delta_dual=", r_dense.Delta_dual,
        "  zeta=", r_dense.zeta)
ζ_ref = r_dense.zeta
λ_ref = r_dense.lambda[1:(obj.outer_constr_index - 1)]

const DEFAULT_OPT = obj.inner_loop_opt   # ek_inner.opt (hessopt=exact) -- restore after each variant

function run_variant(label, opt_path, variant_sym)
    obj.inner_loop_opt = opt_path
    obj.x .= NaN   # force cold start (defensive; use_cached_x defaults false anyway)
    K_hard, x, nStatus, n_fg, n_hess, st = inner_loop_internal_compressed_variant(obj, θ_full, ctx; variant = variant_sym)
    obj.inner_loop_opt = DEFAULT_OPT
    ζ = x[1]; λ = x[2:end]
    dζ = abs(ζ - ζ_ref)
    dλ = maximum(abs.(λ .- λ_ref))
    @printf("%-28s status=%-4d n_fg=%-4d n_hess=%-4d  |dζ|=%.3e  max|dλ|=%.3e  %s\n",
            label, nStatus, n_fg, n_hess, dζ, dλ, (nStatus in (0,-100,-101,-103) && dζ < 1e-5 && dλ < 1e-5) ? "PASS" : "CHECK")
    return (label = label, nStatus = nStatus, n_fg = n_fg, n_hess = n_hess, dzeta = dζ, dlambda = dλ)
end

results = NamedTuple[]
push!(results, run_variant("qn_bfgs (hessopt=2)", joinpath(D4X_ROOT, "full_aod_diag", "d4_exact", "ek_inner_bfgs.opt"), :qn))
push!(results, run_variant("qn_sr1 (hessopt=3)", joinpath(D4X_ROOT, "full_aod_diag", "d4_exact", "ek_inner_sr1.opt"), :qn))
push!(results, run_variant("qn_lbfgs (hessopt=6)", joinpath(D4X_ROOT, "full_aod_diag", "d4_exact", "ek_inner_lbfgs.opt"), :qn))
push!(results, run_variant("denseaccum (hessopt=1, HVP-built)", DEFAULT_OPT, :denseaccum))
push!(results, run_variant("hvp (hessopt=5, matrix-free)", joinpath(D4X_ROOT, "full_aod_diag", "d4_exact", "ek_inner_hvp.opt"), :hvp))

n_pass = count(r -> r.nStatus in (0,-100,-101,-103) && r.dzeta < 1e-5 && r.dlambda < 1e-5, results)
println("\n", n_pass, "/", length(results), " variants PASS (agree with dense reference to <1e-5)")
