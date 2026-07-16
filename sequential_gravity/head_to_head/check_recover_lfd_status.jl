# ============================================================================
# Direct test (2026-07-16): recover_lfd never checks KNITRO's own nStatus from
# inner_loop, only that the returned dual variables/LFD are finite. Does the
# inner solve at GC's T2/warm point actually terminate with a clean/optimal
# KNITRO status, or does it silently accept a non-converged solve?
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf

d = JLD2.load(joinpath(@__DIR__, "out_gc", "gc_T2_warm.jld2"))
θsol = Float64.(d["theta_star"])

# Replicate recover_lfd's own inner_loop call directly, but keep nStatus instead of discarding it.
oci = D + 1 + 1
obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = EK_moments_focal_norm_directgp!, moments_jacobian! = error,
    d = D + 1, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
    l = length(θsol), U = U, N = W, lower_limit = -5000,
    outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
val, x, nStatus = inner_loop(obj, θsol)

@printf("recover_lfd's own inner_loop call at GC T2/warm theta:\n")
@printf("  nStatus = %d\n", nStatus)
@printf("  all(isfinite, x) = %s  (this is the ONLY check recover_lfd actually performs)\n", all(isfinite, x))
@printf("  val (objective) = %.6e\n", val)

# KNITRO status code meaning (from libknitro.jl): 0 = optimal, -100s = feasible-but-iter-limit,
# -200s = infeasible, -300s = unbounded, -400s = other failure (eval error etc).
if nStatus == 0
    println("\n  => KNITRO reports CLEAN OPTIMAL convergence. If so, the moment violation found")
    println("     must have a different cause than 'inner solve didn't finish' -- needs more digging.")
elseif -199 <= nStatus <= -100
    println("\n  => KNITRO reports FEASIBLE BUT NOT FULLY CONVERGED (iteration/time limit hit).")
    println("     This directly confirms the hypothesis: recover_lfd silently accepted a")
    println("     non-converged dual solve because it only checks finiteness, not nStatus.")
else
    println("\n  => KNITRO reports some OTHER non-optimal status ($nStatus) -- also confirms")
    println("     recover_lfd accepted a non-converged/problematic solve without checking.")
end
println("\nCHECK_RECOVER_LFD_STATUS DONE")
