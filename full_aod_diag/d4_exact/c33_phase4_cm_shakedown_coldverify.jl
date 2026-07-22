# Closure task Phase 4 cold-verify -- fresh process, cache disabled by construction
# (cm_production_value_verified runs a brand-new Architecture-C inner solve directly; no
# evaluate_fullA/exact-cache path is used at all here), fresh ctx/pcx built from scratch in
# THIS process. Reports every field the task's Phase 4 requires.
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, Serialization

lp(xs...) = (println(xs...); flush(stdout))
const CKPT_ROOT = ARGS[1]

const DRAW_SEED = 20260719
Random.seed!(DRAW_SEED)
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
snaps = nested_grid_sequence([10, 20, 50])
L = 50
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
fp = context_fingerprint(ctx)
lp(">>> context_fingerprint = ", fp)

function coldverify(tag, ckpt_path)
    isfile(ckpt_path) || (lp("  [$tag] no checkpoint at $ckpt_path, skip"); return nothing)
    ck = load_cm_checkpoint(ckpt_path)   # current schema=2 only -- hard-refuses schema=1
    lp("  [$tag] schema=", ck.schema, " checkpoint_reason=", ck.checkpoint_reason,
       " n_eval=", ck.n_eval, " n_grad=", ck.n_grad, " wall_elapsed=", round(ck.wall_elapsed, digits=1))
    b = ck.best_feasible
    if b === nothing
        lp("  [$tag] checkpoint has no best_feasible incumbent, skip")
        return nothing
    end
    lp("  [$tag] BEST VERIFIED INCUMBENT: gp=", b.gp, " Delta(reported, canonical Delta_dual)=", b.Delta,
       "  |  TERMINAL ITERATE: g=", ck.g, " (differs from best iff a later :wall_interval checkpoint",
       " fired after the last :new_best)")
    xf = vcat(b.w[1], vec(exp.(pivot_expand(b.w[2:end], pe))))
    t0 = time()
    _, base, verify = cm_production_value_verified(xf, pcx)
    t_cold = time() - t0
    gravity_resid = gravity_from_logz(pivot_expand(b.w[2:end], pe), ctx)
    kappa = 1 - b.gp^(ctx.σ / (ctx.σ - 1))
    cls = classify_inner_result(verify)
    lp("  [$tag] cold-reverify (wall=", round(t_cold,digits=2), "s): Delta_dual=", verify.Delta_dual,
       " |diff vs reported|=", abs(verify.Delta_dual - b.Delta),
       " kappa=", kappa, " gravity_residual=", gravity_resid,
       " class=", cls, " inner_status=", verify.inner_status,
       " primal_dual_gap=", verify.primal_dual_gap, " max_abs_moment_kkt_resid=", verify.max_abs_moment_kkt_resid,
       " mean_m_resid=", verify.mean_m_resid, " m_max=", verify.m_max)
    return (tag = tag, reported_Delta = b.Delta, cold_Delta_dual = verify.Delta_dual,
            kappa = kappa, gravity_residual = gravity_resid, class = cls, inner_status = verify.inner_status,
            primal_dual_gap = verify.primal_dual_gap, max_abs_moment_kkt_resid = verify.max_abs_moment_kkt_resid,
            mean_m_resid = verify.mean_m_resid, m_max = verify.m_max, wall_cold_verify = t_cold,
            context_fingerprint = fp, n_eval_at_checkpoint = ck.n_eval, checkpoint_reason = ck.checkpoint_reason)
end

r1 = coldverify("control", joinpath(CKPT_ROOT, "control", "control_latest.jls"))
r2 = coldverify("interrupt_then_resumed", joinpath(CKPT_ROOT, "interrupt", "interrupt_latest.jls"))

open(joinpath(CKPT_ROOT, "phase4_shakedown_coldverify.csv"), "w") do io
    println(io, "tag,reported_Delta,cold_Delta_dual,kappa,gravity_residual,class,inner_status,primal_dual_gap,max_abs_moment_kkt_resid,mean_m_resid,m_max,wall_cold_verify,context_fingerprint,n_eval_at_checkpoint,checkpoint_reason")
    for r in (r1, r2)
        r === nothing && continue
        println(io, join([r.tag, r.reported_Delta, r.cold_Delta_dual, r.kappa, r.gravity_residual, r.class,
            r.inner_status, r.primal_dual_gap, r.max_abs_moment_kkt_resid, r.mean_m_resid, r.m_max,
            r.wall_cold_verify, r.context_fingerprint, r.n_eval_at_checkpoint, r.checkpoint_reason], ","))
    end
end
lp("Phase 4 shakedown cold-verify complete, CSV written to ", joinpath(CKPT_ROOT, "phase4_shakedown_coldverify.csv"))
