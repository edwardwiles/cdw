# ============================================================================
# Closure task Phase 1: resolve the remaining CM checkpoint/cold-solve discrepancy.
#
# Uses the EXACT two checkpoint files the final-gates diagnostics task produced and cold-
# verified (docs/fullA_FINAL_RESIDUAL_GATES_AND_ENDTOEND_BENCHMARK_2026-07-22.md Phase 4):
#   control_latest.jls      (reported Delta=0.8025018742522267, cold Delta_dual=0.6266708456751989)
#   interrupt_latest.jls    (post-resume; reported Delta=0.21311465244991357, cold Delta_dual=0.183151971696322)
# recovered from that session's own scratchpad (both files still present on disk, schema=1,
# predating the remediation task's F1 fix). This script does NOT re-run the 600s+300s KNITRO
# campaign -- it works from the exact stored artifacts, per the task's own instruction to use
# the exact old vectors rather than infer from nearby points.
#
# Produces: full stdout log + CM_CHECKPOINT_COLD_RECONCILIATION_2026-07-22.csv
# ============================================================================
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
using Printf, Random, Serialization, SHA, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const e_THRESH = exp(1)

const OLD_CKPT_DIR = "/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/cf6a4db4-feb9-40ac-892b-c76994a8c189/scratchpad/phase4/ckpt"
const OUT_CSV = ARGS[1]

# ---- fresh, compatible context: EXACT same construction run_cm_upper_checkpointed used to
# produce these two checkpoints (control/interrupt scripts: W=80000, delta=1.0,
# draw_design=:pseudorandom, draw_seed=20260719, L=50, contrasts=:anchored, nested_family probs) ----
lp(">>> Building fresh context (W=80000, delta=1.0, draw_seed=20260719, pseudorandom)...")
ctx = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
snaps = nested_grid_sequence([10, 20, 50])
L = 50
probs = snaps[L]
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs)
lp(">>> Context ready. draw_meta.checksum_uniform=", ctx.draw_meta.checksum_uniform)
lp(">>> draw_meta.checksum_transformed=", ctx.draw_meta.checksum_transformed)
lp(">>> refIndex1 (CM reference origin, ctx.γ.refIndex1-derived, deterministic given ctx)=", pcx.aug.refIndex1)
lp(">>> KNITRO release: ", (try KNITRO.KN_get_release() catch; "unknown" end))
lp(">>> CMCheckpoint field names: ", fieldnames(CMCheckpoint))

function sha256_vec(v::AbstractVector{Float64})
    io = IOBuffer()
    write(io, v)
    return bytes2hex(sha256(take!(io)))
end

function independent_mean_psi_and_tail(m_star::Vector{Float64})
    # Independent recovery of Psi(q*) from m* = dPsi(q*) alone (cc_algo/Psi.jl, read verbatim --
    # Psi!(arg1,arg0): q<=1 => arg1=exp(q); q>1 => arg1=(q^2+1)*0.5*e; THEN arg1 .-= 1.0 for ALL
    # entries (the "-1" is outside the branch). dPsi!(arg1,arg0): q<=1 => arg1=exp(q)=m;
    # q>1 => arg1=e*q=m.
    #   q<=1 branch: m=exp(q)  => Psi(q)=exp(q)-1=m-1                         (m<=e)
    #   q>1  branch: m=e*q, q=m/e => Psi(q)=0.5*e*(q^2+1)-1=0.5*m^2/e+0.5*e-1  (m>e)
    # (Caught live: an earlier version of this function dropped the "+0.5*e" term, which
    # produced a spurious ~0.066 "discrepancy" that was actually a bug in this check, not in
    # the underlying F1 identity -- see docs/REMEDIATION_CLOSURE... for the full trace.)
    # This does NOT use the (-zeta*)-Delta_dual identity itself -- it is an independent
    # closed-form check computed only from the recovered weights.
    psis = similar(m_star)
    @inbounds for i in eachindex(m_star)
        psis[i] = m_star[i] <= e_THRESH ? (m_star[i] - 1.0) : (0.5 * m_star[i]^2 / e_THRESH + 0.5 * e_THRESH - 1.0)
    end
    return mean(psis), maximum(m_star), count(>(e_THRESH), m_star) / length(m_star)
end
using Statistics: mean

function context_compat_report(tag, raw)
    lp("  [$tag] --- context compatibility ---")
    checks = [
        ("W", raw.W, ctx.draw_meta.W),
        ("draw_seed", raw.draw_seed, ctx.draw_meta.draw_seed),
        ("draw_design", raw.draw_design, ctx.draw_meta.draw_design),
        ("delta", raw.delta, 1.0),
        ("cm_L", raw.cm_L, L),
        ("cm_contrasts", raw.cm_contrasts, :anchored),
        ("cm_grid_rule", raw.cm_grid_rule, :nested_family),
        ("cm_hessian_backend", raw.cm_hessian_backend, :structured),
        ("cm_basis", raw.cm_basis, :cumulative),
        ("draw_checksum_uniform", raw.draw_checksum_uniform, ctx.draw_meta.checksum_uniform),
        ("draw_checksum_transformed", raw.draw_checksum_transformed, ctx.draw_meta.checksum_transformed),
        ("cm_probs_match", raw.cm_probs == collect(probs), true),
    ]
    all_ok = true
    for (name, a, b) in checks
        ok = a == b
        all_ok &= ok
        lp("    ", name, ": stored=", a, "  fresh=", b, "  ", ok ? "MATCH" : "*** MISMATCH ***")
    end
    lp("    knitro_version: stored=", raw.knitro_version, "  fresh=", (try KNITRO.KN_get_release() catch; "unknown" end))
    lp("  [$tag] context compatibility: ", all_ok ? "ALL FIELDS MATCH" : "SOME FIELDS DIFFER (see above)")
    return all_ok
end

results = NamedTuple[]

function analyze(tag, path)
    lp("\n================ ", tag, " (", path, ") ================")
    isfile(path) || (lp("  MISSING FILE"); return)
    raw = deserialize(path)::CMCheckpoint
    lp("  schema=", raw.schema, " run_id=", raw.run_id, " label=", raw.label, " branch=", raw.branch,
       " checkpoint_reason=", raw.checkpoint_reason, " n_eval=", raw.n_eval, " n_grad=", raw.n_grad,
       " wall_elapsed=", round(raw.wall_elapsed, digits = 1))

    ctx_ok = context_compat_report(tag, raw)

    w_terminal = vcat(raw.g, raw.zfree)
    b = raw.best_feasible
    b === nothing && (lp("  NO best_feasible incumbent recorded -- skip"); return)
    w_best = b.w
    lp("  best_feasible (BEST VERIFIED INCUMBENT): gp=", b.gp, " Delta(stored, schema-1 == -zeta_star)=", b.Delta,
       " n_eval=", b.n_eval, " t=", round(b.t, digits = 1))
    lp("  terminal iterate (g/zfree at checkpoint WRITE time, may != best_feasible if a later")
    lp("    :wall_interval checkpoint fired after the last :new_best): w_terminal == w_best? ",
       w_terminal == w_best)
    lp("  same-record check: b.gp == b.w[1] ? ", b.gp == b.w[1], "   (Delta and w come from the ",
       "same NamedTuple literal in cb_F! -- same-record by construction, confirmed structurally)")

    hash_terminal = sha256_vec(w_terminal)
    hash_best = sha256_vec(w_best)
    lp("  SHA256(w_terminal) = ", hash_terminal)
    lp("  SHA256(w_best)     = ", hash_best)

    xf = vcat(w_best[1], vec(exp.(pivot_expand(w_best[2:end], pe))))

    # ---- cold solve #1: default cold start (obj.x forced to NaN => zeros(outer_constr_index)) ----
    pcx.ctx_cm.obj.use_cached_x = false
    pcx.ctx_cm.obj.x .= NaN
    _, base1, verify1 = cm_production_value_verified(xf, pcx)
    neg_zeta1 = -base1.ζstar
    mean_psi1, m_max1, tailfrac1 = independent_mean_psi_and_tail(base1.m_star)

    # ---- cold solve #2: materially different start (large random perturbation, NOT the
    # NaN-cold-start default and NOT related to base1's own converged point) ----
    Random.seed!(999_000_111)
    n_inner = length(pcx.ctx_cm.obj.x)
    pcx.ctx_cm.obj.use_cached_x = true
    pcx.ctx_cm.obj.x = 3.0 .* randn(n_inner)
    _, base2, verify2 = cm_production_value_verified(xf, pcx)
    neg_zeta2 = -base2.ζstar
    mean_psi2, m_max2, tailfrac2 = independent_mean_psi_and_tail(base2.m_star)
    pcx.ctx_cm.obj.use_cached_x = false   # restore default for next tag

    lp("  cold solve #1 (default cold-start, x=NaN->zeros): Delta_dual=", verify1.Delta_dual,
       " -zeta*=", neg_zeta1, " status=", verify1.inner_status)
    lp("  cold solve #2 (perturbed start, ||x0||~", round(norm(3.0 .* randn(n_inner)), digits=1),
       "):        Delta_dual=", verify2.Delta_dual, " -zeta*=", neg_zeta2, " status=", verify2.inner_status)
    lp("  |Delta_dual(#1) - Delta_dual(#2)| = ", abs(verify1.Delta_dual - verify2.Delta_dual),
       "  (convex inner problem -- should reproduce to the existing cold-verification tolerance)")

    # ---- required numerical identity: (-zeta*) - Delta_dual ~= mean(Psi(q*)) [independent] ----
    lhs1 = neg_zeta1 - verify1.Delta_dual
    lp("  IDENTITY check (solve #1): (-zeta*) - Delta_dual = ", lhs1, "   independent mean(Psi(q*)) = ", mean_psi1,
       "   |diff| = ", abs(lhs1 - mean_psi1))
    lp("  m_max = ", m_max1, "  tail_frac(m>e) = ", tailfrac1, "  m_mean(verify) = ", verify1.m_mean)

    feasible_at_caller_delta = verify1.Delta_dual <= raw.delta + 1e-6
    lp("  cold_Delta_dual <= caller's delta(", raw.delta, ")+1e-6 ? ", feasible_at_caller_delta)

    # ---- classify against the OLD (buggy, reported) Delta ----
    diff_reported_vs_cold = abs(b.Delta - verify1.Delta_dual)
    diff_negzeta_vs_reported = abs(neg_zeta1 - b.Delta)
    lp("  reported (schema-1) Delta = ", b.Delta, "   fresh -zeta* = ", neg_zeta1,
       "   |reported - fresh(-zeta*)| = ", diff_negzeta_vs_reported)
    lp("  |reported_Delta - cold_Delta_dual| = ", diff_reported_vs_cold,
       "   vs mean(Psi(q*)) = ", mean_psi1, "   |that diff - mean(Psi(q*))| = ",
       abs(diff_reported_vs_cold - mean_psi1))

    classification = if diff_negzeta_vs_reported < 1e-6 && abs(diff_reported_vs_cold - mean_psi1) < 1e-3 * max(1.0, diff_reported_vs_cold)
        "old checkpoint stored -zeta_star, and the F1 identity (mean(Psi(q*))) fully explains the discrepancy"
    else
        "IDENTITY DOES NOT FULLY EXPLAIN -- further diagnosis needed (see printed fields above)"
    end
    lp("  >>> CLASSIFICATION: ", classification)

    push!(results, (tag = tag, hash_w_terminal = hash_terminal, hash_w_best = hash_best,
        reported_Delta_schema1 = b.Delta, fresh_neg_zeta_star = neg_zeta1, cold_Delta_dual = verify1.Delta_dual,
        cold_Delta_dual_start2 = verify2.Delta_dual, mean_Psi_qstar = mean_psi1, m_max = m_max1,
        tail_frac_m_gt_e = tailfrac1, inner_status = verify1.inner_status,
        primal_dual_gap = verify1.primal_dual_gap, max_abs_moment_kkt_resid = verify1.max_abs_moment_kkt_resid,
        mean_m_resid = verify1.mean_m_resid, feasible_at_caller_delta = feasible_at_caller_delta,
        context_compat_all_match = ctx_ok, classification = classification))
end

analyze("control", joinpath(OLD_CKPT_DIR, "control", "control_latest.jls"))
analyze("interrupt_then_resumed", joinpath(OLD_CKPT_DIR, "interrupt", "interrupt_latest.jls"))

open(OUT_CSV, "w") do io
    println(io, "tag,hash_w_terminal,hash_w_best,reported_Delta_schema1,fresh_neg_zeta_star,cold_Delta_dual,cold_Delta_dual_start2,mean_Psi_qstar,m_max,tail_frac_m_gt_e,inner_status,primal_dual_gap,max_abs_moment_kkt_resid,mean_m_resid,feasible_at_caller_delta,context_compat_all_match,classification")
    for r in results
        println(io, join([r.tag, r.hash_w_terminal, r.hash_w_best, r.reported_Delta_schema1, r.fresh_neg_zeta_star,
            r.cold_Delta_dual, r.cold_Delta_dual_start2, r.mean_Psi_qstar, r.m_max, r.tail_frac_m_gt_e,
            r.inner_status, r.primal_dual_gap, r.max_abs_moment_kkt_resid, r.mean_m_resid,
            r.feasible_at_caller_delta, r.context_compat_all_match, "\"$(r.classification)\""], ","))
    end
end
lp("\n>>> Phase 1 reconciliation CSV written to ", OUT_CSV)
