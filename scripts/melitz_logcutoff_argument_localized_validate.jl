# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 11: tests whether the EXISTING dependency map (`melitz_localized_dependency_map`,
# built purely from `ctx.A_pivot`/`ctx.f_free_lin`/avoid-sets -- all structural quantities
# SHARED between :logf and :logcutoff, log_cutoff_param.jl's own header: "1. A-gravity:
# UNCHANGED from :logf -- log A is linear in A_free via the SAME ctx.A_pivot") already
# produces a valid SUPERSET claim under :logcutoff too, with NO code changes -- since
# `melitz_expand_theta`'s dispatcher already reconstructs the correct (A,f) pair for
# WHICHEVER parameterization `ctx.outer_parameterization` names, the argument-localized
# backend's economics (which call `melitz_expand_theta` exactly like the dense reference)
# should already be correct at any TOUCHED column; what needs checking is only whether the
# CLAIMED touched-column SET is still a valid superset -- i.e. whether :logcutoff introduces
# a genuinely NEW dependency (e.g. "an A coordinate at fixed q changes f in the SAME cell",
# the governing prompt's own flagged risk) that lands OUTSIDE the existing claimed cells.
#
# Method: build a :logcutoff ctx (same recipe as the existing "Session prompt Section 5"
# test), then compare :B (full reference, parameterization-agnostic by construction) against
# the UNMODIFIED :B_argument_localized_serial/_parallel backends at that ctx. Bit-exact
# agreement would mean the existing dependency map is ALREADY a correct superset for
# :logcutoff -- a genuine, cheap confirmation rather than an assumption.
#
# Usage: JULIA_NUM_THREADS=<n> julia --project=. scripts/melitz_logcutoff_argument_localized_validate.jl

using Printf, Random
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function run_logcutoff_validation(; D=4, W=5_000, seed=29, n_random_points=4)
    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    obj_f, theta0_f = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx_f = obj_f.γ

    ctx_q = merge(ctx_f, (outer_parameterization=:logcutoff,))
    obj_q, _ = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    obj_q = merge_obj_ctx(obj_q, ctx_q)  # see helper below: swap obj.γ to the :logcutoff ctx
    theta0_q = melitz_reduce_theta(data.primitives, ctx_q)

    n = length(theta0_q)
    d = ctx_q.moment_layout.num_moments
    W_ = size(obj_q.U, 1)
    println("D=$D  n=$n  K=$d  W=$W_  outer_parameterization=:logcutoff")

    mj_full = make_melitz_moments_jacobian_b(1e-4)
    mj_argser = make_melitz_moments_jacobian_b_argument_localized_serial(1e-4)
    mj_argpar = make_melitz_moments_jacobian_b_argument_localized_parallel(1e-4)

    rng = MersenneTwister(97)
    all_ok = true
    for trial in 0:n_random_points
        theta_probe = trial == 0 ? copy(theta0_q) : theta0_q .+ 0.01 .* randn(rng, n)
        K1, G1 = zeros(W_, n), zeros(W_, d, n)
        K2, G2 = zeros(W_, n), zeros(W_, d, n)
        K3, G3 = zeros(W_, n), zeros(W_, d, n)
        mj_full(K1, G1, theta_probe, obj_q.U, obj_q)
        mj_argser(K2, G2, theta_probe, obj_q.U, obj_q)
        mj_argpar(K3, G3, theta_probe, obj_q.U, obj_q)

        ok_serial = isapprox(G1, G2; atol=1e-6, rtol=1e-6)
        ok_parallel = isapprox(G1, G3; atol=1e-6, rtol=1e-6)
        max_diff_serial = maximum(abs.(G1 .- G2))
        max_diff_parallel = maximum(abs.(G1 .- G3))
        all_ok &= ok_serial && ok_parallel
        @printf("trial=%d  serial vs full: ok=%s max|diff|=%.3e   parallel vs full: ok=%s max|diff|=%.3e\n",
            trial, ok_serial, max_diff_serial, ok_parallel, max_diff_parallel)
    end
    println(all_ok ? "ALL CHECKS PASSED (existing dependency map is ALREADY valid under :logcutoff)" :
                      "*** MISMATCH: :logcutoff needs its own dependency-map derivation (governing prompt's anticipated case) ***")
    return all_ok
end

# obj.γ (ctx) is not directly mutable on PsiObjectiveBundleDelta in general -- reconstruct
# a fresh bundle sharing the same U/draws but with the :logcutoff ctx installed, via the
# SAME build_melitz_psi_bundle entry point, then overwrite its γ field if settable, else
# fall back to a manual field copy. Kept as a tiny local helper rather than changing any
# production constructor for a one-off diagnostic script.
function merge_obj_ctx(obj, ctx_new)
    try
        obj.γ = ctx_new
        return obj
    catch
        return Base.setproperty!(obj, :γ, ctx_new)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_logcutoff_validation()
    exit(ok ? 0 : 1)
end
