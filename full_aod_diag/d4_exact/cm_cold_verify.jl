# ============================================================================
# Standalone, single-purpose cold-verification script for the 2026-07-22 CM production
# campaign. Given a CMCheckpoint (.jls) path, this:
#   1. loads the checkpoint (schema=2 only -- load_cm_checkpoint hard-refuses schema-1),
#   2. rebuilds ctx/pcx FROM SCRATCH from the checkpoint's own recorded (W, delta, draw_design,
#      draw_seed, L, probs, contrasts, cm_hessian_backend) -- never trusts the checkpoint's
#      own stored Delta/kappa,
#   3. cold-re-evaluates best_feasible.w (NOT the terminal g/zfree) via
#      cm_production_value_verified in a genuinely fresh process (no exact-point cache exists
#      on the CM path at all -- see docs/CM_PRODUCTION_STATE_2026-07-22.md -- so "fresh process"
#      and "cache disabled" are the same thing here, unlike the unrestricted path),
#   4. asserts the regenerated draw checksums match the checkpoint's own (refuses silently
#      otherwise -- same guarantee run_cm_upper_checkpointed's own resume path gives),
#   5. writes a small serialized NamedTuple result (for the supervisor to read back the
#      verified w-vector + Delta_dual to seed the NEXT delta stage) and prints a one-line
#      summary to stdout.
#
# Usage: julia --project=. cm_cold_verify.jl <checkpoint_path> <output_path>
# Exits nonzero (propagating the underlying error) if verification cannot be performed --
# never silently produces a "best effort" answer.
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
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))   # CM-C+ production integration 2026-07-23: cm_gradient_backend now defaults to :cplus in run_cm_upper_checkpointed, so this must always be on the include path
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))   # CM+moments(+ZC) production integration 2026-07-23: cm_extension may be non-:cm_only, so this must always be on the include path
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))

const ckpt_path = ARGS[1]
const out_path = ARGS[2]

isfile(ckpt_path) || error("cm_cold_verify: no checkpoint at $ckpt_path")
ckpt = load_cm_checkpoint(ckpt_path)   # hard-refuses schema-1 with an actionable message
ckpt.best_feasible === nothing && error("cm_cold_verify($ckpt_path): checkpoint has no best_feasible incumbent to verify (no verified-feasible point was ever found in this stage)")

lp(">>> cold-verifying ", ckpt_path, " (run_id=", ckpt.run_id, " label=", ckpt.label,
   " delta=", ckpt.delta, " W=", ckpt.W, " n_eval=", ckpt.n_eval, ")")

# Rebuild the context strictly from the checkpoint's own recorded provenance -- never from
# whatever ambient defaults this process happens to have.
ctx = d20_real_setup_design(W = ckpt.W, δ = ckpt.delta, find_smallest = ckpt.find_smallest,
                             draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed)
pe = build_pivot_elimination(ctx)

if ctx.draw_meta.checksum_uniform != ckpt.draw_checksum_uniform ||
   ctx.draw_meta.checksum_transformed != ckpt.draw_checksum_transformed
    error("cm_cold_verify($ckpt_path): draw checksum MISMATCH -- regenerated draws (design=" *
          ":$(ckpt.draw_design), seed=$(ckpt.draw_seed)) do not match the checkpoint's own " *
          "recorded checksums. Refusing to cold-verify against a different problem instance.")
end

is_meanzc = ckpt.cm_extension !== :cm_only
pcx = is_meanzc ?
    build_cm_meanzc_production_context(ctx, CS; L = ckpt.cm_L, K_mean = ckpt.meanzc_K_mean,
        K_pair = ckpt.meanzc_K_pair, contrasts = ckpt.cm_contrasts, meanzc_basis = ckpt.meanzc_basis,
        probs = ckpt.cm_probs) :
    build_cm_production_context(ctx, CS; L = ckpt.cm_L, contrasts = ckpt.cm_contrasts, probs = ckpt.cm_probs)

w = ckpt.best_feasible.w
D2_econ = length(w) - (is_meanzc ? ckpt.meanzc_K_mean : 0)
xf = x_free_from_w(w[1:D2_econ], pe)
K, base, verify = if is_meanzc
    νvec = exp.(w[D2_econ+1:end])
    cm_meanzc_production_value_verified(xf, νvec, pcx)
else
    cm_production_value_verified(xf, pcx)
end

reported_Delta = ckpt.best_feasible.Delta
cold_Delta = verify.Delta_dual
diff = abs(cold_Delta - reported_Delta)
verified_ok = is_verified_success(verify)
feasible = isfinite(cold_Delta) && cold_Delta <= ckpt.delta + 1e-6

lp(">>> reported Delta=", reported_Delta, " cold Delta_dual=", cold_Delta, " |diff|=", diff,
   " verified_success=", verified_ok, " feasible=", feasible,
   is_meanzc ? " cm_extension=$(ckpt.cm_extension) K_mean=$(ckpt.meanzc_K_mean) K_pair=$(ckpt.meanzc_K_pair) eta_nu=$(w[D2_econ+1:end])" : "")

verified_ok || error("cm_cold_verify($ckpt_path): cold re-evaluation of the stored incumbent " *
                      "did NOT pass is_verified_success -- refusing to hand this vector forward " *
                      "as a seed for the next delta stage. class=$(classify_inner_result(verify))")
feasible || error("cm_cold_verify($ckpt_path): cold-verified Delta_dual=$(cold_Delta) exceeds " *
                   "delta=$(ckpt.delta) + tol -- refusing to hand this vector forward.")

result = (w = copy(w), Delta_dual = cold_Delta, reported_Delta = reported_Delta, diff = diff,
          delta = ckpt.delta, W = ckpt.W, draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed,
          cm_L = ckpt.cm_L, contrasts = ckpt.cm_contrasts, schema = ckpt.schema, bi = ctx.bi,
          cm_extension = ckpt.cm_extension, meanzc_K_mean = ckpt.meanzc_K_mean,
          meanzc_K_pair = ckpt.meanzc_K_pair, meanzc_basis = ckpt.meanzc_basis,
          source_ckpt = ckpt_path, verified_at = string(now()))
serialize(out_path, result)
lp(">>> wrote cold-verified seed vector to ", out_path)
