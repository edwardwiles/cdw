# ============================================================================
# Ad-hoc test (2026-07-16): does exact_inner_divergence_at's audit succeed on
# a previously-FAILING point (LC T2/rand1, audited_delta_star=1e10) if the
# inner dual solve is warm-started from a nearby easy point's converged x,
# instead of the current hardcoded zeros(outer_constr_index)?
#
# build_fixed_dual_bundle (derivative_diagnostics/fixed_dual_criterion.jl)
# never sets use_cached_x/x on the PsiObjectiveBundleDelta it constructs, so
# inner_loop_initial_values(obj::PsiObjectiveBundleDelta) always falls to the
# zeros(...) branch (cc_algo/inner_loop_functions.jl:140) -- every audit call
# is a cold start from literal zero duals, regardless of how far theta is
# from the calibrated baseline.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra

function exact_inner_divergence_at_warmstart(θ::Vector{Float64}, x_init::Vector{Float64})
    frozen = freeze_gravity_linearization(θ, seq_gravcol, grad_R_theta)
    frozen.ok || return (δ_star=Inf, R_mean=Inf, nStatus=-999, gravity_ok=false)
    moments_fn = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ)
    obj = build_fixed_dual_bundle(γ, U, length(θ), D + 2, moments_fn; find_smallest=true)
    obj.use_cached_x = true
    obj.x = copy(x_init)
    δ_star, x_star, nStatus = inner_loop(obj, θ)
    return (δ_star=δ_star, R_mean=frozen.R, nStatus=nStatus, gravity_ok=(abs(frozen.R) <= 5e-4), x_star=x_star)
end

println("="^78)
println(">>> WARM-START AUDIT TEST: LC T2/rand1 (cold audit failed, audited_delta_star=1e10)")
println("="^78)

d = JLD2.load(joinpath(@__DIR__, "out_lc", "lc_T2_rand1.jld2"))
θ_fail = Float64.(d["best_feasible_theta"])
@printf("Loaded θ: gp=%.6f, saved audited_delta_star=%s, saved nStatus of audit unknown (only scalar saved)\n",
        θ_fail[3], string(d["audited_delta_star"]))

# --- 1. Re-confirm the cold failure reproduces (sanity check before declaring anything fixed) ---
t0 = time()
cold = exact_inner_divergence_at(θ_fail)
@printf("\n[COLD, x0=zeros] delta_star=%.6g  nStatus=%d  gravity_ok=%s  wall=%.1fs\n",
        cold.δ_star, cold.nStatus, cold.gravity_ok, time() - t0)

# --- 2. Get a warm x from an EASY, nearby point: same target gp, A_od=A* (the calibrated
#     baseline -- this is exactly the "delta*(A_od=A*)" baseline check already computed
#     successfully all over this project, e.g. in the LU/GU drivers' own baseline evals). ---
θ_easy = copy(θr0)
θ_easy[3] = θ_fail[3]   # same gamma'_focal target, A_od = A* (unmoved)
t0 = time()
easy = exact_inner_divergence_at(θ_easy)
@printf("\n[DONOR: A_od=A*, same gp target] delta_star=%.6g  nStatus=%d  gravity_ok=%s  wall=%.1fs  ‖x_star‖=%.3e\n",
        easy.δ_star, easy.nStatus, easy.gravity_ok, time() - t0, norm(easy.x_star))

if !easy.gravity_ok || !hasproperty(easy, :x_star) || isempty(easy.x_star)
    println("\nDonor point itself failed -- cannot proceed with warm-start test as designed.")
else
    # --- 3. Use the donor's converged x as the warm start for the FAILING point ---
    t0 = time()
    warm = exact_inner_divergence_at_warmstart(θ_fail, collect(easy.x_star))
    @printf("\n[WARM, x0=donor's converged x] delta_star=%.6g  nStatus=%d  gravity_ok=%s  wall=%.1fs\n",
            warm.δ_star, warm.nStatus, warm.gravity_ok, time() - t0)

    println("\n" * "="^78)
    if isfinite(warm.δ_star) && warm.δ_star < 50.0 && warm.gravity_ok
        println(">>> RESULT: WARM START SUCCEEDED where cold start failed.")
        @printf(">>> This confirms the failure is an initialization/conditioning issue in the audit's\n")
        @printf(">>> cold (zeros) dual start, NOT evidence the underlying point is economically invalid.\n")
        @printf(">>> True delta* for LC T2/rand1 (via warm-started audit) = %.6f\n", warm.δ_star)
    else
        println(">>> RESULT: warm start did NOT resolve it either -- the failure is not simply about")
        println(">>> initialization; something else is going on and this needs deeper investigation")
        println(">>> before trusting either explanation.")
    end
    println("="^78)
end
println("\nTEST_WARMSTART_AUDIT DONE")
