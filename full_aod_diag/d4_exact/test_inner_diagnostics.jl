# Task §8: inner CC dual validation. Run: julia --project=. full_aod_diag/d4_exact/test_inner_diagnostics.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "inner_diagnostics.jl"))

const COMMIT = "b048943"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)

println("="^78); println("TEST A: direct dual-objective cross-check vs production callback"); println("="^78)
r0 = evaluate_fullA(x0, ctx; cache = nothing)
θ_full0 = r0.θ_full
inner_x0 = vcat(r0.zeta, r0.lambda)
Δ_direct = direct_dual_objective(ctx.obj, inner_x0, θ_full0)
println("production Delta_dual = ", r0.Delta_dual)
println("direct re-evaluation  = ", -Δ_direct, "  (direct_dual_objective returns f; Delta=-f per verified sign convention)")
println("abs diff = ", abs(r0.Delta_dual - (-Δ_direct)))
@assert abs(r0.Delta_dual - (-Δ_direct)) < 1e-10
println("PASS: independent direct dual-objective evaluation matches production callback")

println("\n" * "="^78); println("TEST B: inner dual Hessian eigenvalues / conditioning"); println("="^78)
# repopulate obj.arg0/H at inner_x0 (evaluate_fullA already left obj in this state, but be explicit)
cbuf = zeros(ctx.obj.d - ctx.obj.outer_constr_index + 2)
ctx.obj(inner_x0, constr = @view(cbuf[1:length(cbuf)]))
diag = inner_dual_conditioning(ctx.obj, inner_x0)
println("dim = ", diag.dim, "  lambda_min = ", diag.λmin, "  lambda_max = ", diag.λmax)
println("condition number = ", diag.cond_number, "  numerical_rank = ", diag.numerical_rank, " / ", diag.dim)
@assert diag.λmin > 0 "inner dual Hessian is not positive definite at a converged solve -- unexpected"
println("PASS: inner dual Hessian positive definite (SPD required for a genuine strict-convex CC dual minimum)")

open(joinpath(OUTDIR, "inner_checks.csv"), "w") do io
    println(io, "point,dim,lambda_min,lambda_max,cond_number,numerical_rank")
    println(io, "theta0_up,", diag.dim, ",", diag.λmin, ",", diag.λmax, ",", diag.cond_number, ",", diag.numerical_rank)
end

println("\n" * "="^78); println("TEST C: cold vs warm, two inner tolerances"); println("="^78)
rows = inner_solve_reliability(x0, ctx; n_cold = 3,
    tol_opt_files = [joinpath(D4X_ROOT, "full_aod_diag", "d4_exact", "ek_inner_loose.opt"),
                     joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt")])
ctx.obj.inner_loop_opt = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt")  # restore default
for r in rows
    println("  ", r.kind, " trial=", r.trial, " opt=", basename(r.opt_file), "  Delta_dual=", r.Delta_dual, "  nStatus=", r.nStatus)
end
Δs_tight = [r.Delta_dual for r in rows if occursin("ek_inner.opt", r.opt_file) && !occursin("loose", r.opt_file)]
spread_tight = maximum(Δs_tight) - minimum(Δs_tight)
println("spread across cold/warm repeats at the TIGHT tolerance: ", spread_tight)
@assert spread_tight < 1e-9 "inner solve is not repeatable at tight tolerance across cold/warm restarts"
println("PASS: inner solve reproducible across cold/warm restarts at tight tolerance (spread=", spread_tight, ")")

open(joinpath(OUTDIR, "inner_tolerance_reliability.csv"), "w") do io
    println(io, "kind,trial,opt_file,Delta_dual,nStatus")
    for r in rows
        println(io, r.kind, ",", r.trial, ",", basename(r.opt_file), ",", r.Delta_dual, ",", r.nStatus)
    end
end
println("\nWrote ", joinpath(OUTDIR, "inner_checks.csv"), " and inner_tolerance_reliability.csv")
println("\nALL INNER-DIAGNOSTIC TESTS PASSED")
