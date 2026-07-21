# ============================================================================
# Phase D: granular inner-solve profiling at representative delta points, and
# organic infeasibility characterization. Uses REAL saved checkpoints from
# diag/fullA-d20-canonical-rerun's own delta-frontier run (copied into
# organic_pathology/) rather than freshly-generated points, per the plan.
#
# Points profiled:
#   1. accepted delta=1 boundary point (d1_startA_canon_latest.jls)
#   2. accepted delta=2 boundary point (d2_startA_canon_stage_complete_neval158.jls)
#   3. accepted delta=5 interior point (d5_startA_canon_stage_complete_neval16.jls)
#   4. organic delta=5 point that passes every screen and ends in inner -300
#      (nonzero_winner_infeasible_delta5_candidate1.json + its Aod_theta_full CSV --
#      independently reproduced by diag/fullA-d20-canonical-rerun, 2026-07-20)
#
# For each of points 1-3, profiles {neutral cold, exact same-point warm} (the
# "current last-successful warm" and "nearest-cached warm" conditions collapse
# to the same thing for a freshly-rebuilt ctx with no other history -- noted,
# not fabricated).
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, DelimitedFiles, Serialization

"Reuses the regex-based JSON field extractor pattern from
diag/fullA-d20-warmstart-replay's verify_safe_cache_concurrent_real.jl, avoiding
a JSON.jl dependency this worktree's Project.toml doesn't declare directly."
function json_field_num(text, key)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), text)
    m !== nothing || error("json_field_num: key not found: $key")
    return parse(Float64, m.captures[1])
end
function json_field_int(text, key)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9]+)"), text)
    m !== nothing || error("json_field_int: key not found: $key")
    return parse(Int, m.captures[1])
end

lp(xs...) = (println(xs...); flush(stdout))

function profile_point(label::String, ctx, rsc, pe, xf::Vector{Float64}, dual_warm_start::Union{Nothing,Vector{Float64}})
    lp("\n--- ", label, " ---")
    # neutral cold
    ctx.obj.x .= NaN
    t0 = time()
    r_cold, m_cold = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = nothing, use_cache = false, warm = false, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_cold = time() - t0
    lp("  [neutral cold] wall=", round(t_cold, digits=3), "s  status=", r_cold.inner_status,
       "  screen_elapsed=", get(m_cold, :screen_elapsed, get(m_cold, :elapsed, NaN)),
       "  n_inner_solves=", get(m_cold, :n_inner_solves, "n/a"), "  n_inner_iters=", get(m_cold, :n_inner_iters, "n/a"),
       "  n_fg_calls=", get(m_cold, :n_fg_calls, "n/a"), "  n_hess_calls=", get(m_cold, :n_hess_calls, "n/a"),
       "  Delta_dual=", r_cold.Delta_dual, "  kkt=", r_cold.max_abs_moment_kkt_resid)

    # exact same-point warm (re-solve immediately after cold, so obj.x holds the just-converged dual)
    t0 = time()
    r_warm, m_warm = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = nothing, use_cache = false, warm = true, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_warm = time() - t0
    lp("  [exact same-point warm] wall=", round(t_warm, digits=3), "s  status=", r_warm.inner_status,
       "  n_inner_iters=", get(m_warm, :n_inner_iters, "n/a"), "  n_fg_calls=", get(m_warm, :n_fg_calls, "n/a"),
       "  n_hess_calls=", get(m_warm, :n_hess_calls, "n/a"), "  Delta_dual=", r_warm.Delta_dual)

    # checkpoint's own saved dual_warm_start (the "current last-successful warm" this driver would have used)
    if dual_warm_start !== nothing
        ctx.obj.x .= dual_warm_start
        t0 = time()
        r_ckwarm, m_ckwarm = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
            cache = nothing, use_cache = false, warm = true, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
        t_ckwarm = time() - t0
        lp("  [checkpoint's saved dual_warm_start] wall=", round(t_ckwarm, digits=3), "s  status=", r_ckwarm.inner_status,
           "  n_inner_iters=", get(m_ckwarm, :n_inner_iters, "n/a"), "  Delta_dual=", r_ckwarm.Delta_dual)
    end
    return (t_cold = t_cold, t_warm = t_warm, r_cold = r_cold, m_cold = m_cold)
end

x_free_from_w2(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

results = Dict{String,Any}()

for (label, ckpt_path, delta) in [
        ("delta=1 accepted boundary", joinpath(@__DIR__, "organic_pathology", "d1_startA_canon_latest.jls"), 1.0),
        ("delta=2 accepted boundary", joinpath(@__DIR__, "organic_pathology", "d2_startA_canon_stage_complete_neval158.jls"), 2.0),
        ("delta=5 accepted interior", joinpath(@__DIR__, "organic_pathology", "d5_startA_canon_stage_complete_neval16.jls"), 5.0),
    ]
    ckpt = deserialize(ckpt_path)
    lp("\n=== Loading ", label, " (n_eval=", ckpt.n_eval, " knitro_iter=", ckpt.knitro_iter, ") ===")
    Random.seed!(ckpt.draw_seed)
    ctx = d20_real_setup(W = ckpt.W, find_smallest = ckpt.find_smallest, δ = ckpt.delta)
    pe = build_pivot_elimination(ctx)
    rsc = build_ranged_screen_context(ctx)
    xf = x_free_from_w2(vcat(ckpt.g, ckpt.zfree), pe)
    results[label] = profile_point(label, ctx, rsc, pe, xf, ckpt.dual_warm_start)
end

# ---- organic delta=5 point: passes every screen, ends in inner -300 ----
lp("\n=== Loading organic delta=5 -300 pathology point (nonzero_winner_infeasible_delta5_candidate1) ===")
meta_text = read(joinpath(@__DIR__, "organic_pathology", "nonzero_winner_infeasible_delta5_candidate1.json"), String)
gp_focal = json_field_num(meta_text, "gp_focal")
expected_inner_status = json_field_int(meta_text, "inner_status")
Aod_full = readdlm(joinpath(@__DIR__, "organic_pathology", "c11_Aod_theta_full_delta5_upper_FIXED.csv"), ',', Float64)
lp("gp_focal=", gp_focal, "  Aod_full size=", size(Aod_full), "  expected inner_status=", expected_inner_status)

Random.seed!(20260719)
ctx5 = d20_real_setup(W = 80000, find_smallest = true, δ = 5.0)
pe5 = build_pivot_elimination(ctx5)
rsc5 = build_ranged_screen_context(ctx5)
D = ctx5.D
zfree_organic = pivot_reduce(log.(Aod_full), pe5)
xf_organic = x_free_from_w2(vcat(gp_focal, zfree_organic), pe5)

ctx5.obj.x .= NaN
t0 = time()
r_org, m_org = evaluate_fullA_screened_ranged(xf_organic, ctx5, rsc5; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = false, pairwise = ctx5.pairwise, witness = ctx5.witness, use_witness = true)
t_org = time() - t0
lp("[organic -300 point, cold] wall=", round(t_org, digits=3), "s  status=", r_org.inner_status,
   "  screen_status=", get(m_org, :screen_status, "n/a"), "  screen_elapsed=", get(m_org, :screen_elapsed, get(m_org, :elapsed, NaN)),
   "  n_inner_solves=", get(m_org, :n_inner_solves, "n/a"), "  n_inner_iters=", get(m_org, :n_inner_iters, "n/a"))
lp("  MATCHES expected -300 pathology: ", r_org.inner_status == -300, " (independently reproduced in THIS worktree's current code)")

lp("\nDONE_PHASE_D_PROFILE")
