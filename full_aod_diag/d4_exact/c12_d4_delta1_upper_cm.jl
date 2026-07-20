# Continuation 12, section 13: D=4 full CONSTRAINED upper-bound solve at delta=1 with the
# common-marginals restriction, using the fully GENERIC outer-loop path (CS.outer_loop, dense
# ForwardDiff jac_h -- safe at D=4's scale, unlike D=20 -- see docs/fullA_jach_audit.md for why
# this is disabled by default at D=20). Deliberately does NOT use the specialized Method-B
# envelope-gradient path (run_fullA_D4_production.jl's div_grad_fn!/moment_map!): that path
# hardcodes a direct call to EK_moments_gammanorm_directgp! (bypassing obj.moments!, hence
# bypassing the CM wrapper entirely) rather than dispatching through obj.moments! generically --
# confirmed by reading full_aod_diag/ad_benchmark/derivative_core.jl::moment_map!. The plain
# CS.outer_loop path (cc_algo/outer_loop_functions.jl) DOES dispatch generically through
# obj.moments! (calculate_grad_k_autodiff!, calculate_jac_θ_autodiff! -- both call obj.moments!
# directly), so it is the only currently-verified-CM-correct outer-loop entry point. Slower per
# iterate than the specialized paths but correct; a fast CM-aware envelope-gradient path is a
# follow-up (see handoff doc "what reduced wall time").
#
# SIGN CONVENTION: verified EMPIRICALLY (c12_sign_convention_smoke_test.jl), not just derived from
# reading code -- the docstring's "swap find_smallest" language is easy to misapply and an earlier
# version of this file got it backwards. With the plain generic CS.outer_loop path (NOT the
# specialized run_fullA_D4_production.jl cached path, which applies its own compensating
# (-1)^find_smallest flip inside a custom obj_grad_fn! that this script does not use),
# find_smallest=true empirically drives kappa from 0.0642 (calibration) toward the KNOWN upper
# anchor (~0.1725, reaching 0.1636 in a 25-iter smoke test); find_smallest=false drives it toward
# the known lower anchor (~0.0044, reaching 0.0039). So find_smallest=TRUE is the upper-bound
# direction here -- the opposite of a naive reading of moments_gammanorm.jl's docstring.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
using Printf

L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
δ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0

println(">>> Continuation 12 section 13: D=4 CM-constrained upper bound, L=$L, delta=$δ"); flush(stdout)

ctx = d4_exact_setup(δ = δ, find_smallest = true, needs_outer_moment_jacobian = true,
                      outer_loop_opt = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_100.opt"))
aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
obj_cm = aug.obj_cm
println(">>> obj_cm: d=$(obj_cm.d)  outer_constr_index=$(obj_cm.outer_constr_index)  ncm=$(aug.ncm)"); flush(stdout)

# sanity: calibration point still solves under obj_cm before committing to a full outer search
r_calib = evaluate_fullA(ctx.θ0_up[ctx.free_idx], merge(ctx, (obj = obj_cm,)); use_cache = false, warm = false)
println(">>> calibration sanity: nStatus=$(r_calib.inner_status)  Delta_dual=$(r_calib.Delta_dual)"); flush(stdout)
@assert r_calib.inner_status in (0, -100, -101, -103) "calibration point must be inner-feasible before starting the outer search"

t0 = time()
γp_min, θ_min, nStatus, lambda_ = CS.outer_loop(obj_cm, ctx.θ_lo, ctx.θ_hi, ctx.θ0_up)
t_elapsed = time() - t0

κ = 1 - γp_min^(ctx.σ / (ctx.σ - 1))
println("\n================ RESULT (L=$L, delta=$δ) ================")
@printf("status=%d  gamma'_focal=%.10f  kappa=%.10f  wall=%.2fs\n", nStatus, γp_min, κ, t_elapsed)

# independent re-verification at the reported solution: fresh cold inner solve via evaluate_fullA,
# checking Delta_dual<=delta, gravity residual, and the CM-block residual specifically
θfull_free = θ_min[ctx.free_idx]
r_final = evaluate_fullA(θfull_free, merge(ctx, (obj = obj_cm,)); use_cache = false, warm = false)
@printf("INDEPENDENT RECHECK: nStatus=%d  Delta_dual=%.8f (target<=%.4f)  gravity_value=%.3e  gamma'=%.10f  kappa=%.10f\n",
        r_final.inner_status, r_final.Delta_dual, δ, r_final.gravity_value, r_final.gamma_focal_prime,
        1 - r_final.gamma_focal_prime^(ctx.σ / (ctx.σ - 1)))

W = size(ctx.U, 1)
K = zeros(W); G = zeros(W, obj_cm.d)
obj_cm.moments!(K, G, r_final.θ_full, ctx.U, obj_cm)
m_full = copy(obj_cm.arg1)
ncore = aug.ncore
cm_cols = ncore:(ncore + aug.ncm - 1)
cm_kkt = maximum(abs(sum(m_full .* G[:, j]) / W) for j in cm_cols)
@printf("CM-block max KKT residual at solution: %.3e\n", cm_kkt)

using Serialization
outdir = joinpath(D4X_ROOT, "results", "fullA_d4", "c12_common_marginals")
mkpath(outdir)
outfile = joinpath(outdir, "delta$(δ)_L$(L)_anchored.jls")
serialize(outfile, (L = L, δ = δ, γp = γp_min, κ = κ, θ_min = θ_min, nStatus = nStatus,
                     Delta_dual = r_final.Delta_dual, gravity_value = r_final.gravity_value,
                     cm_kkt = cm_kkt, wall = t_elapsed))
println(">>> saved: ", outfile)
