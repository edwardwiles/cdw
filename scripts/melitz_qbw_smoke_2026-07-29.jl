# Quick standalone smoke test for the new Phase 1/2 files (q_bandwidth_policy.jl,
# exact_q_smooth_gradient.jl) before running the full test suite. Not part of the
# predeclared campaign -- throwaway, D4 only.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf

data = generate_fake_melitz_data(; D=4, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx = obj.γ

obj.use_cached_x = false; obj.x .= NaN
lfd0 = melitz_recover_lfd(obj, theta0)
@assert lfd0.lfd_ok
x0 = copy(lfd0.dual_x)
@printf("Delta0=%.6e nStatus=%d\n", lfd0.Delta, lfd0.nStatus)

D = ctx.D
nA = D^2 - 1
nq = length(theta0) - 1 - nA
println("nq (free q coords) = ", nq)

# --- Phase 2: exact smooth q gradient vs zero-switch fixed-dual secant ---
state0 = MelitzExpandedState(D)
melitz_expand_theta!(state0, melitz_unpower_theta_free(theta0, ctx), ctx, MelitzThetaExpansionWorkspace(D))
grad_free_smooth, dDelta_dq_full = melitz_exact_q_smooth_gradient(obj, x0, state0, ctx)
println("smooth q gradient (free, first 5): ", grad_free_smooth[1:min(5, end)])
println("nonzero full-cell entries: ", count(!iszero, dDelta_dq_full), " / ", length(dDelta_dq_full))

sorted_ctx = ctx.sorted_tail_ctx
max_rel_err = 0.0
for m in 1:min(nq, 5)
    # find a tiny h with zero two-sided crossings
    h = 1e-9
    total_plus, total_minus, _ = melitz_q_two_sided_crossings(theta0, m, h, ctx, sorted_ctx)
    tries = 0
    while (total_plus > 0 || total_minus > 0) && tries < 10
        h /= 10
        total_plus, total_minus, _ = melitz_q_two_sided_crossings(theta0, m, h, ctx, sorted_ctx)
        tries += 1
    end
    theta_p = copy(theta0); theta_p[1+nA+m] += h
    theta_m = copy(theta0); theta_m[1+nA+m] -= h
    melitz_update_operator_at_theta!(obj.op, theta_p, ctx)
    Dp = -obj(x0)
    melitz_update_operator_at_theta!(obj.op, theta_m, ctx)
    Dm = -obj(x0)
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    secant = (Dp - Dm) / (2h)
    exact = grad_free_smooth[m]
    relerr = abs(secant - exact) / (abs(exact) + abs(secant) + 1e-300)
    global max_rel_err = max(max_rel_err, relerr)
    @printf("m=%d h=%.2e crossings=(%d,%d) exact=%.8e secant=%.8e relerr=%.3e\n",
            m, h, total_plus, total_minus, exact, secant, relerr)
end
println("max symmetric relative error (zero-switch): ", max_rel_err)
@assert max_rel_err < 1e-3 "smooth q gradient does not match zero-switch secant"

# --- Phase 1: bandwidth policies smoke test ---
println("\n--- bandwidth policy smoke test ---")
policies = [
    FixedRawQBandwidth(1e-4),
    PowerScaledQBandwidth(1e-4, 20_000, 0.5),
    FixedCrossingQBandwidth(25),
    GrowingCrossingQBandwidth(25, 20_000),
]
m_test = 1
for pol in policies
    r = melitz_q_coordinate_probe(theta0, m_test, pol, obj, ctx; x0=x0, mode=:fixed_dual)
    @printf("%-28s h=%.4e crossings=(+%d,-%d,min=%d) secant=%.6e cells=%d elapsed=%.4fs bytes=%d\n",
            string(typeof(pol)), r.h, r.crossings_plus_total, r.crossings_minus_total,
            r.crossings_min_side, r.secant, length(r.cells), r.elapsed_s, r.bytes_allocated)
end

println("\nSMOKE TEST PASSED")
