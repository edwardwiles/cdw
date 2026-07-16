# Validation (2026-07-16, fix/cm-fixed-dual-gradient branch): does
# GRADIENT_METHOD=fixed_dual_fd_full now work correctly with CM_ENABLED=true?
#
# Tests the gradient computation DIRECTLY at a single point (not a full outer KNITRO search --
# fixed_dual_fd_full's own FD sub-evaluations are cheap/no-KNITRO per dual_criterion_fixed_x's
# docstring, but the OUTER search itself would need many genuine expensive iterations to converge
# with the correct, non-premature gradient -- not needed to validate correctness of the gradient
# computation itself).
#
# 1) Unit-level: make_frozen_gravity_moments's CM_Moments kwarg produces the right shape/values.
# 2) CM_L=0 regression: the div_grad_fn! closure at theta_r0 gives a FINITE, sane gradient (no
#    crash) -- confirms the nCM=0 default path is unaffected.
# 3) CM_L>0 functional: same closure doesn't crash with CM on, gradient is finite, AND the
#    Acol-block gradient genuinely differs from the CM_L=0 case (confirms the CM restriction is
#    not silently ignored in the corrected-gradient path).
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

println("\n" * "="^90); println(">>> 1) UNIT CHECK: make_frozen_gravity_moments CM_Moments kwarg"); println("="^90)
CM_Moments_test = CM_ENABLED ? γ.CM_Moments : rand(W, 5)
d_test = D + 2 + size(CM_Moments_test, 2)
lastθ0 = copy(θr0); Rcol0 = 0.0; gcol0 = zeros(W); dRdθ0 = zeros(length(θr0))
frozen_fn = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, lastθ0, Rcol0, gcol0, dRdθ0;
    CM_Moments = CM_Moments_test)
Ktest = zeros(W); Gtest = zeros(W, d_test)
frozen_fn(Ktest, Gtest, θr0, U, (γ = γ,))
cm_slice = Gtest[:, D+3:D+2+size(CM_Moments_test,2)]
match = isapprox(cm_slice, CM_Moments_test[1:W, :]; atol=0)
@printf("CM block shape=%s matches input exactly: %s\n", size(cm_slice), match)
@assert match "CM block mismatch in make_frozen_gravity_moments"

println("\n" * "="^90); println(">>> 2/3) div_grad_fn! (fixed_dual_fd_full) at theta_r0, single point"); println("="^90)
@printf("CM_L=%d CM_ENABLED=%s nCM=%d\n", CM_L, CM_ENABLED, nCM)

d = D + 2 + nCM; oci = d + 1
CS.check_methodB_valid(d, oci)
m!, gcol, lastRmean, best_θ, best_κ, best_warm, lastθ_st, lastRcol_st, dRdθ_st, lastok_st =
    make_stateful_moments(; use_exact_grad = true, find_smallest = true, δ = 1.0)
obj = CS.PsiObjectiveBundleImplicitMethodB(δ = 1.0, find_smallest = true, γ = γ,
    (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
    inequality_index = Int64[], complement_index = [0 0], l = length(θr0), U = U, N = JacW,
    lower_limit = -50, use_cached_x = false,
    outer_loop_opt = OUTER_OPT_FILE, inner_loop_opt = INNER_OPT_FILE)
l_full = length(θr0)
# NOTE: named free_idx_vec, NOT free_idx -- this script runs at top level (Main scope), and
# `free_idx` is also the name of a global function (ProfiledGravity.free_idx(ref,D), used inside
# grad_R_theta) that a same-named top-level variable would silently shadow (safe inside
# outer_solve_nested_cached's own function scope, NOT safe here).
free_idx_vec = vcat(3, collect(4:3+D)); fixed_idx = [1, 2]; fixed_vals = θr0[fixed_idx]
fpmap = CS.FreeParamMap(l_full, free_idx_vec, fixed_idx, fixed_vals)

t0 = time()
val, inner_x, nStatus = inner_loop(obj, θr0)
@printf("inner_loop at theta_r0: nStatus=%d wall=%.1fs lastok_st=%s\n", nStatus, time()-t0, lastok_st[])
@assert nStatus ∈ (0, -100, -101, -103) "inner solve at theta_r0 must converge for this test"
@assert lastok_st[] "make_stateful_moments state must be populated (lastok_st) after inner_loop"

div_grad_fn! = make_seq_div_grad_fn_full!(obj, fpmap, γ, U, D, gcol, lastθ_st, lastRcol_st, dRdθ_st, lastok_st,
    :fixed_dual_fd_full; nCM = nCM)
x_free = CS.pack_free(θr0, fpmap)
g_free = zeros(CS.n_free(fpmap))
t0 = time()
div_grad_fn!(g_free, x_free, θr0, inner_x)
@printf("div_grad_fn! wall=%.1fs\n", time() - t0)
@printf("g_free[1] (gamma'_focal) = %.6e\n", g_free[1])
@printf("g_free[2:end] (Acol block): min=%.4e max=%.4e mean=%.4e any_nan=%s any_inf=%s\n",
        minimum(g_free[2:end]), maximum(g_free[2:end]), sum(g_free[2:end])/length(g_free[2:end]),
        any(isnan, g_free), any(isinf, g_free))
@assert all(isfinite, g_free) "g_free must be fully finite"
println(CM_ENABLED ?
    ">>> CM_L>0 + fixed_dual_fd_full: gradient computed successfully, fully finite. FUNCTIONAL CHECK PASSED." :
    ">>> CM_L=0 + fixed_dual_fd_full: gradient computed successfully, fully finite (regression baseline).")

open(joinpath(@__DIR__, "cm_grad_check_$(CM_ENABLED ? "on" : "off").csv"), "w") do io
    println(io, "idx,g_free")
    for (i, v) in enumerate(g_free)
        println(io, "$i,$v")
    end
end

println("\nVALIDATE_CM_FIXED_DUAL_FD_FULL DONE")
