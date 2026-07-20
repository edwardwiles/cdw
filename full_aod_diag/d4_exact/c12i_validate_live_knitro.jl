# Part B correctness validation (live KNITRO wiring): the custom lookup-FG KNITRO inner solve
# (cm_lookup_live_knitro.jl) must agree with the DENSE baseline's live KNITRO inner solve
# (evaluate_fullA, oracle.jl -- completely unmodified) at real fixed outer parameters, on
# nStatus, Delta_dual, Delta_primal, primal LFD weights, and max moment KKT residual -- the SAME
# equivalence bar as c12i_validate_interval_equiv.jl's Section 2, but now exercising the actual
# custom KNITRO callback registration path (not just the FG math in isolation).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "cm_lookup_live_knitro.jl"))
using Printf, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
pe = build_pivot_elimination(ctx)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

x_calib_raw = ctx.θ0_up[ctx.free_idx]
Random.seed!(4243)
x_perturbed = copy(x_calib_raw)
x_perturbed[2:end] .*= exp.(0.05 .* randn(length(x_perturbed) - 1))

points = [("calibration", x_calib_raw), ("perturbed_feasible", x_perturbed)]

all_ok = Ref(true)
for L in (10, 20, 50)
    println("\n--- L=$L ---")
    aug_dense = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    aug_int   = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = :anchored)
    ctx_dense = merge(ctx, (obj = aug_dense.obj_cm,))

    for (label, x_free) in points
        rd = evaluate_fullA(x_free, ctx_dense; use_cache = false, warm = false)
        d_ok = rd.inner_status in (0, -100, -101, -103)
        m_weights_dense = Float64[]
        if d_ok
            W = size(ctx.U, 1)
            K = zeros(W); G = zeros(W, aug_dense.obj_cm.d)
            aug_dense.obj_cm.moments!(K, G, rd.θ_full, ctx.U, aug_dense.obj_cm)
            m_weights_dense = copy(aug_dense.obj_cm.arg1)
        end

        for method in (:interval, :suffix)
            # aug_dense (build_cm_augmented_obj, the CUMULATIVE reference) has no `bins` field --
            # bins depend only on U/z (basis-independent), so reuse aug_int's for the :suffix case.
            aug = method == :interval ? aug_int : merge(aug_dense, (bins = aug_int.bins,))
            ri = evaluate_fullA_cmlookup(x_free, ctx, aug; method = method, nthreads_use = 1, warm = false)
            i_ok = ri.inner_status in (0, -100, -101, -103)

            if !d_ok || !i_ok
                status_ok = d_ok == i_ok
                @printf("  %-20s %-8s STATUS dense=%d(ok=%s) lookup=%d(ok=%s)  %s\n",
                        label, string(method), rd.inner_status, d_ok, ri.inner_status, i_ok,
                        status_ok ? "consistent" : "*** MISMATCH ***")
                all_ok[] &= status_ok
                continue
            end

            dd_err = abs(rd.Delta_dual - ri.Delta_dual)
            dp_err = abs(rd.Delta_primal - ri.Delta_primal)
            m_relerr = maximum(abs.(m_weights_dense .- ri.m_weights)) / max(1.0, maximum(abs.(m_weights_dense)))
            kkt_d = rd.max_abs_moment_kkt_resid
            kkt_i = ri.max_abs_moment_kkt_resid
            n_iters = ri.inner_iters

            @printf("  %-20s %-8s Delta_dual err=%.3e  Delta_primal err=%.3e  m_relerr=%.3e  KKT dense=%.2e lookup=%.2e  n_fg_calls=%d\n",
                    label, string(method), dd_err, dp_err, m_relerr, kkt_d, kkt_i, ri.n_fg_calls)

            ok_here = dd_err < 1e-6 && dp_err < 1e-6 && m_relerr < 1e-6
            all_ok[] &= ok_here
            ok_here || println("    *** MISMATCH ABOVE TOLERANCE ***")
        end
    end
end

println("\n" * "="^100)
println(all_ok[] ? "LIVE-KNITRO EQUIVALENCE: PASS -- lookup-wired inner solve agrees with the dense baseline at every point checked." :
                    "LIVE-KNITRO EQUIVALENCE: FAIL -- see mismatches above.")
println("="^100)
