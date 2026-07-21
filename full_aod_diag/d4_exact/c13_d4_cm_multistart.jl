# Continuation 13, Section 7: D4 outer rerun with the production CM-aware gradient, multistart
# across the genuinely nested Q10/Q20/Q50 grids, with cross-grid incumbent injection as the
# nested-monotonicity diagnostic the brief requires.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization

# w_up40: the existing unrestricted D4 upper headline candidate (kappa=0.17245688540655113),
# hardcoded from candidate_registry.jl's own literal (NOT re-included here -- that file rebuilds
# its own separate `ctx` via a second d4_exact_setup() call, which collides with this script's own
# oracle_fast.jl/cm_hessian_architectures.jl include chain via an ambiguous PsiObjectiveBundleDelta
# binding; the underlying candidate vector itself is unaffected by that, only the loader script).
const w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
                0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
                1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
                0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D

w_from_xfree(xf) = vcat(xf[1], pivot_reduce(log.(reshape(xf[2:end], D, D)), pe))
w_calib = w_from_xfree(ctx.θ0_up[ctx.free_idx])
w_unrestricted = w_up40   # from candidate_registry.jl, kappa=0.17246 (badly CM-infeasible per C12)

snaps = nested_grid_sequence([10, 20, 50])
MAXTIME = 120.0
DELTA = 1.0

results = Dict{Int,Any}()
prev_best_w = nothing

for L in (10, 20, 50)
    println("="^100)
    @printf "GRID L=%d (%d cutpoints)\n" L length(snaps[L])
    println("="^100)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])

    starts = Dict{String,Vector{Float64}}("calibration" => w_calib, "unrestricted_candidate" => w_unrestricted)
    if prev_best_w !== nothing
        starts["prev_grid_best"] = prev_best_w
    end

    runs = NamedTuple[]
    for (sname, w0) in starts
        println("-- start: ", sname)
        try
            res = run_cm_upper(pcx, ctx, pe, copy(w0); delta = DELTA, maxtime_real = MAXTIME, verbose = true)
            @printf "   status=%d wall=%.1fs n_eval=%d kappa=%s\n" res.knitro_status res.wall res.n_eval string(res.kappa)
            push!(runs, (start = sname, res = res))
        catch e
            println("   FAILED: ", sprint(showerror, e)[1:min(200, end)])
        end
    end

    valid = [r for r in runs if r.res.best !== nothing]
    if isempty(valid)
        println("  ** NO FEASIBLE RESULT AT L=", L, " -- all starts failed to find a feasible point **")
        continue
    end
    best_run = valid[argmax([r.res.kappa for r in valid])]
    @printf "  BEST at L=%d: start=%s kappa=%.8f Delta=%.6f\n" L best_run.start best_run.res.kappa best_run.res.best.Delta
    results[L] = (pcx = pcx, best_run = best_run, all_runs = runs, probs = snaps[L])
    global prev_best_w = best_run.res.best.w
end

println()
println("="^100)
println("CROSS-GRID MONOTONICITY CHECK (nested-grid incumbent injection)")
println("="^100)
for Lden in (20, 50)
    haskey(results, Lden) || continue
    for Lcoarse in (10, 20)
        Lcoarse >= Lden && continue
        haskey(results, Lcoarse) || continue
        wden = results[Lden].best_run.res.best.w
        pcx_coarse = results[Lcoarse].pcx
        try
            _, base = cm_production_value(x_free_from_w(wden, pe), pcx_coarse)
            Delta_at_coarse = -base.ζstar
            feasible_at_coarse = Delta_at_coarse <= DELTA + 1e-6
            kappa_den = results[Lden].best_run.res.kappa
            kappa_coarse = results[Lcoarse].best_run.res.kappa
            monotone_ok = kappa_coarse >= kappa_den - 1e-6
            @printf "  L=%d best (kappa=%.6f) under L=%d grid: Delta=%.6f feasible=%s | L=%d's own best kappa=%.6f | monotone (coarse>=dense)? %s\n" Lden kappa_den Lcoarse Delta_at_coarse feasible_at_coarse Lcoarse kappa_coarse monotone_ok
        catch e
            println("  L=$Lden best under L=$Lcoarse grid: FAILED (infeasible or error) -- ", sprint(showerror, e)[1:min(150,end)])
        end
    end
end

println()
println("="^100)
println("COLD DENSE-REFERENCE VERIFICATION of the final (L=50 if available, else best) candidate")
println("="^100)
final_L = haskey(results, 50) ? 50 : (haskey(results, 20) ? 20 : 10)
if haskey(results, final_L)
    wfinal = results[final_L].best_run.res.best.w
    xf_final = x_free_from_w(wfinal, pe)
    aug_ref = build_cm_augmented_obj(ctx, CS; L = final_L, contrasts = :anchored, probs = snaps[final_L])
    ctx_ref = merge(ctx, (obj = aug_ref.obj_cm,))
    r_ref = evaluate_fullA(xf_final, ctx_ref; use_cache = false, warm = false)
    kappa_final = results[final_L].best_run.res.kappa
    @printf "  final candidate (L=%d, kappa=%.8f): fresh dense-reference nStatus=%d Delta_dual=%.8f (<=%.2f? %s) gravity=%.3e\n" final_L kappa_final r_ref.inner_status (-r_ref.zeta) DELTA ((-r_ref.zeta)<=DELTA+1e-6) r_ref.gravity_value

    serialize(joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_cm_multistart_best.jls"),
        (L = final_L, kappa = kappa_final, w = wfinal, Delta_dual = -r_ref.zeta, gravity = r_ref.gravity_value,
         inner_status = r_ref.inner_status, probs = snaps[final_L]))
    println("  saved to results/fullA_d4/c13_cm_multistart_best.jls")
end
println("DONE")
