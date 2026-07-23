# ============================================================================
# Standalone cold-verification script for origin-specific-ZC production checkpoints.
# Structural analog of cm_cold_verify.jl (same contract: rebuild ctx/pcx from the
# checkpoint's OWN recorded provenance, cold-re-evaluate best_feasible.w in a fresh
# process with no exact-point cache, assert draw checksums match, refuse silently
# otherwise) but for CMCheckpointV5 origin-family checkpoints
# (distribution_restriction != :unrestricted), loaded via load_cm_checkpoint_v5 and
# verified via cm_originzc_production_value_verified -- never routed through the
# CM-family build_cm_production_context/build_cm_meanzc_production_context path.
#
# Usage: julia --project=. originzc_cold_verify.jl <checkpoint_path> <output_path>
# Exits nonzero if verification cannot be performed -- never a silent best-effort answer.
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
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))
using Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))

const ckpt_path = ARGS[1]
const out_path = ARGS[2]

isfile(ckpt_path) || error("originzc_cold_verify: no checkpoint at $ckpt_path")
ckpt = load_cm_checkpoint_v5(ckpt_path)
ckpt.distribution_restriction !== :unrestricted ||
    error("originzc_cold_verify($ckpt_path): checkpoint has distribution_restriction=:unrestricted " *
          "-- this is not an origin-ZC checkpoint, use cm_cold_verify.jl instead.")
ckpt.best_feasible === nothing && error("originzc_cold_verify($ckpt_path): checkpoint has no best_feasible incumbent to verify")

lp(">>> cold-verifying ", ckpt_path, " (run_id=", ckpt.run_id, " label=", ckpt.label,
   " delta=", ckpt.delta, " W=", ckpt.W, " n_eval=", ckpt.n_eval,
   " distribution_restriction=", ckpt.distribution_restriction, " K_mean=", ckpt.origin_K_mean,
   " K_pair=", ckpt.origin_K_pair, " power_target_layout=", ckpt.power_target_layout, ")")

ctx = d20_real_setup_design(W = ckpt.W, δ = ckpt.delta, find_smallest = ckpt.find_smallest,
                             draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed)
pe = build_pivot_elimination(ctx)

if ctx.draw_meta.checksum_uniform != ckpt.draw_checksum_uniform ||
   ctx.draw_meta.checksum_transformed != ckpt.draw_checksum_transformed
    error("originzc_cold_verify($ckpt_path): draw checksum MISMATCH -- regenerated draws (design=" *
          ":$(ckpt.draw_design), seed=$(ckpt.draw_seed)) do not match the checkpoint's own " *
          "recorded checksums. Refusing to cold-verify against a different problem instance.")
end
ctx.D == ckpt.origin_D ||
    error("originzc_cold_verify($ckpt_path): D MISMATCH -- checkpoint origin_D=$(ckpt.origin_D), regenerated ctx.D=$(ctx.D)")

layout = OriginByPowerLayout(ctx.D, ckpt.origin_K_mean, ckpt.origin_K_pair)
pcx = build_originzc_production_context(ctx, CS, layout)

w = ckpt.best_feasible.w
D2_econ = length(w) - n_eta(layout)
xf = x_free_from_w(w[1:D2_econ], pe)
νfull = exp.(w[D2_econ+1:end])
K, base, verify = cm_originzc_production_value_verified(xf, νfull, pcx)

# C+ versus Reference agreement, at the SAME cold-verified incumbent (release Gate B/C
# requirement: "complete C+ versus Reference gradients").
cplus_pool = build_grad_workspace_pool(size(ctx.obj.U, 1))
cplus_ws = build_lfix_factorized_workspace(ctx.D, size(ctx.obj.U, 1))
g_cplus, _ = cm_originzc_production_gradient_cplus(xf, νfull, pcx, ctx, pe, cplus_pool, cplus_ws;
    base = base, verify = verify, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
g_ref, _ = cm_originzc_production_gradient(xf, νfull, pcx, ctx, pe;
    base = base, verify = verify, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
grad_diff = norm(g_cplus .- g_ref)
grad_cosine = dot(g_cplus, g_ref) / (norm(g_cplus) * norm(g_ref))

reported_Delta = ckpt.best_feasible.Delta
cold_Delta = verify.Delta_dual
diff = abs(cold_Delta - reported_Delta)
verified_ok = is_verified_success(verify)
feasible = isfinite(cold_Delta) && cold_Delta <= ckpt.delta + 1e-6
eta_gradient_finite = all(isfinite, g_cplus[end-n_eta(layout)+1:end])

lp(">>> reported Delta=", reported_Delta, " cold Delta_dual=", cold_Delta, " |diff|=", diff,
   " verified_success=", verified_ok, " feasible=", feasible,
   " eta_nu=", w[D2_econ+1:end])
lp(">>> C+ vs Reference gradient: |diff|=", grad_diff, " cosine=", grad_cosine,
   " eta_gradient_finite=", eta_gradient_finite)

verified_ok || error("originzc_cold_verify($ckpt_path): cold re-evaluation of the stored incumbent " *
                      "did NOT pass is_verified_success -- refusing to hand this vector forward. " *
                      "class=$(classify_inner_result(verify))")
feasible || error("originzc_cold_verify($ckpt_path): cold-verified Delta_dual=$(cold_Delta) exceeds " *
                   "delta=$(ckpt.delta) + tol -- refusing to hand this vector forward.")
eta_gradient_finite || error("originzc_cold_verify($ckpt_path): non-finite eta gradient component(s) in the cold-verified C+ gradient")

result = (w = copy(w), Delta_dual = cold_Delta, reported_Delta = reported_Delta, diff = diff,
          delta = ckpt.delta, W = ckpt.W, draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed,
          schema = ckpt.schema, bi = ctx.bi,
          distribution_restriction = ckpt.distribution_restriction, origin_K_mean = ckpt.origin_K_mean,
          origin_K_pair = ckpt.origin_K_pair, power_target_layout = ckpt.power_target_layout,
          grad_diff = grad_diff, grad_cosine = grad_cosine,
          source_ckpt = ckpt_path, verified_at = string(now()))
serialize(out_path, result)
lp(">>> wrote cold-verified seed vector to ", out_path)
