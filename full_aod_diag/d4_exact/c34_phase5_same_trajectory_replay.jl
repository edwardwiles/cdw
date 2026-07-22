# ============================================================================
# Closure task Phase 5 (SCOPED): end-to-end performance measurement for the unrestricted
# full-A path, real D=20/W=80,000, delta=0.1 and delta=1.0.
#
# HONEST SCOPE NOTE: the task's full Phase 5 spec (every native-eval category incl. screens/
# warm/cold/infeasible/exact-cache/checkpoint-serialization/GC broken out separately, at both
# deltas) is, by the remediation report's own prior assessment, "a genuinely large, multi-hour
# instrumentation-plus-multiple-real-KNITRO-runs project on its own." This script executes the
# highest-value, most-directly-decisive subset within this task's time budget:
#   1. Explicit backend assertion (C+ arm -> :cplus, Reference arm -> :buffered).
#   2. A REAL short C+ trajectory captured at each delta, via a genuinely additive, opt-in
#      instrumentation kwarg (grad_trace_ref=, added this session to
#      c10_d20_production_driver.jl's run_polish_checkpointed, same pattern as the existing
#      full_trace_ref= kwarg -- zero behavior/allocation change unless a caller passes a Ref).
#      (An earlier version of this script tried a runtime monkeypatch of
#      composite_gradient_at_Cplus; caught live: giving the override the EXACT same method
#      signature made the "keep a reference to call the original" alias self-referential
#      (Julia methods are stored in one global per-signature table, not per-binding), causing
#      infinite recursion / a real StackOverflowError. The additive kwarg avoids this
#      entirely and is the same pattern this codebase already uses for full_trace_ref.)
#   3. SAME-TRAJECTORY REPLAY: every captured base state timed under BOTH backends
#      (composite_gradient_at_Cplus vs composite_gradient_at_fast_buffered), confirming
#      numerical equivalence and computing the counterfactual whole-trajectory gradient-only
#      wall time under each backend, holding the function-evaluation sequence fixed.
#   4. Reconciliation against the real run's own n_grad_calls counter and native KNITRO
#      ga_evals count.
# NOT done here (flagged, not silently skipped): per-category screen/warm/cold/infeasible/
# serialization/GC breakdown: that remains the deferred multi-hour item.
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Printf, Statistics

lp(xs...) = (println(xs...); flush(stdout))

lp(">>> Backend assertion:")
lp("    C+ arm        -> ", resolve_price_cache_backend("phase5", nothing, :cplus))
lp("    Reference arm -> ", resolve_price_cache_backend("phase5", nothing, :buffered))
@assert resolve_price_cache_backend("phase5", nothing, :cplus) == :cplus
@assert resolve_price_cache_backend("phase5", nothing, :buffered) == :buffered

results = NamedTuple[]
replay_rows = NamedTuple[]

for delta in (0.1, 1.0)
    lp("\n================ delta=", delta, " ================")
    grad_trace = Ref(Vector{Float64}[])
    fctx = build_fullA_context(W = 80000, δ = delta, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
    ctx = fctx.ctx; pe = fctx.pe; D = ctx.D
    Aod_real = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
    zfree0 = pivot_reduce(log.(Aod_real), pe)
    gF = ctx.θ0_up[3+D]
    ckpt = mktempdir()
    t0 = time()
    res = run_polish_checkpointed("phase5_d$(delta)", true, gF, zfree0; maxtime_real = 90.0,
        W_in = 80000, delta_in = delta, ckpt_dir = ckpt, checkpoint_interval_s = 200.0,
        reuse = fctx, price_cache_backend = :cplus, grad_trace_ref = grad_trace)
    wall_total = time() - t0
    n_captured = length(grad_trace[])
    lp("  real C+ run: wall=", round(wall_total,digits=1), "s n_eval=", res.n_eval,
       " n_grad_calls=", res.n_grad_calls, " native_ga_evals=", res.native_outer_diag.n_ga_evals,
       " captured gradient-callback xf states=", n_captured, " knitro_status=", res.knitro_status)

    # ---- SAME-TRAJECTORY REPLAY: time both backends on every real captured state ----
    grad_pool = build_grad_workspace_pool(80000)
    lfix_c_ws = build_lfix_factorized_workspace(D, 80000)
    t_cplus_total = 0.0; t_buffered_total = 0.0; max_absdiff = 0.0
    for (i, xf) in enumerate(grad_trace[])
        base = solve_base_state(xf, ctx)
        t1 = @elapsed (g_c, _) = composite_gradient_at_Cplus(xf, ctx, pe, grad_pool, lfix_c_ws;
            base = base, threaded = true, h_mode = :fixed, h0 = 0.01)
        t2 = @elapsed (g_b, _) = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base,
            threaded = true, h_mode = :fixed, h0 = 0.01)
        d = maximum(abs.(g_c .- g_b))
        max_absdiff = max(max_absdiff, d)
        t_cplus_total += t1; t_buffered_total += t2
        push!(replay_rows, (delta = delta, idx = i, t_cplus = t1, t_buffered = t2, max_absdiff = d))
    end
    lp("  same-trajectory replay (n=", n_captured, "): counterfactual whole-trajectory GRADIENT-ONLY wall:")
    lp("    C+       total=", round(t_cplus_total, digits = 3), "s  mean/call=", round(n_captured > 0 ? t_cplus_total/n_captured : NaN, digits = 4), "s")
    lp("    buffered total=", round(t_buffered_total, digits = 3), "s  mean/call=", round(n_captured > 0 ? t_buffered_total/n_captured : NaN, digits = 4), "s")
    lp("    ratio (buffered/cplus) = ", n_captured > 0 && t_cplus_total > 0 ? round(t_buffered_total/t_cplus_total, digits = 2) : NaN)
    lp("    max |g_cplus - g_buffered| across all captured states = ", max_absdiff, " (numerical equivalence check)")
    push!(results, (delta = delta, wall_total = wall_total, n_eval = res.n_eval, n_grad_calls = res.n_grad_calls,
                     native_ga_evals = res.native_outer_diag.n_ga_evals, n_captured = n_captured,
                     t_cplus_total = t_cplus_total, t_buffered_total = t_buffered_total, max_absdiff = max_absdiff))
end

open(joinpath(@__DIR__, "..", "..", "docs", "closure_2026-07-22", "FULLA_SAME_TRAJECTORY_BACKEND_REPLAY_2026-07-22.csv"), "w") do io
    println(io, "delta,idx,t_cplus,t_buffered,max_absdiff")
    for r in replay_rows
        println(io, join([r.delta, r.idx, r.t_cplus, r.t_buffered, r.max_absdiff], ","))
    end
end
open(joinpath(@__DIR__, "..", "..", "docs", "closure_2026-07-22", "FULLA_CALLBACK_WALL_DECOMPOSITION_2026-07-22.csv"), "w") do io
    println(io, "delta,wall_total_real_run,n_eval,n_grad_calls,native_ga_evals,n_captured_grad_states,t_cplus_total_replay,t_buffered_total_replay,max_absdiff_replay")
    for r in results
        println(io, join([r.delta, r.wall_total, r.n_eval, r.n_grad_calls, r.native_ga_evals, r.n_captured,
                           r.t_cplus_total, r.t_buffered_total, r.max_absdiff], ","))
    end
end
lp("\n>>> Phase 5 same-trajectory replay complete. CSVs written to docs/closure_2026-07-22/.")
