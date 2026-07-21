# Continuation 14, Task 6: short real D=20 delta=1 L=50 CM continuation smoke test.
#
# Reuses run_cm_upper (cm_outer_driver.jl) -- the ONLY validated CM continuation driver that
# exists on this branch (produced Continuation 13's real D20/L=50 result,
# results/fullA_d4/c13_d20_cm_continuation/stage_L50_latest.jls, kappa=0.0591). DISCLOSED GAP,
# not silently worked around: unlike the unrestricted path's run_profile_checkpointed
# (c10_d20_production_driver.jl), run_cm_upper does NOT go through the schema-3 D20Checkpoint
# machinery (knitro_version field, resume validation, draw-design checksum gate) -- it uses its
# own simpler save_stage-style NamedTuple checkpoint (see c13_d20_cm_upper_continuation.jl). This
# smoke test still explicitly verifies the KNITRO-version invariant itself (verify_knitro_version,
# same check c10_d20_production_driver.jl runs at include time) since that is the one part of the
# schema-3 contract that materially matters for trusting these numbers. Unifying the CM
# continuation driver onto the schema-3 checkpoint machinery is flagged as a real follow-up item
# in the final handoff doc, not attempted here given the smoke-test's own time budget.
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
include(joinpath(@__DIR__, "knitro_version_check.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
lp("=== c14_smoke_cm_l50 === ", Dates.now())
lp(">>> KNITRO version check: ", verify_knitro_version())

const CKPT_DIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod", "smoke_cm_l50")
rm(CKPT_DIR; recursive = true, force = true)
mkpath(CKPT_DIR)

W = 80000; DELTA = 1.0
MAXTIME = get(ENV, "C14_SMOKE_MAXTIME", "180") |> x -> parse(Float64, x)

t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d free_dims=%d", time()-t0, D, W, D^2))

x_free_calib = ctx.θ0_up[ctx.free_idx]
w_from_xfree(xf) = vcat(xf[1], pivot_reduce(log.(reshape(xf[2:end], D, D)), pe))
w_calib = w_from_xfree(x_free_calib)

snaps = nested_grid_sequence([10, 20, 50])
L = 50
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
lp(@sprintf(">>> CM production context (L=%d) built. ncm=%d d_total=%d", L, pcx.aug.ncm, pcx.ctx_cm.obj.d))

lp(">>> starting short CM upper-bound continuation, maxtime_real=", MAXTIME, "s ...")
t0 = time()
result = run_cm_upper(pcx, ctx, pe, copy(w_calib); delta = DELTA, maxtime_real = MAXTIME, verbose = true)
t_wall = time() - t0

lp(""); lp("="^100); lp("SMOKE TEST SUMMARY"); lp("="^100)
lp(@sprintf("wall=%.1fs  n_eval=%d  n_grad=%d  kappa=%s  knitro_status=%d",
    t_wall, result.n_eval, result.n_grad, string(result.kappa), result.knitro_status))
lp("best !== nothing: ", result.best !== nothing)
if result.best !== nothing
    lp("  best.Delta=", result.best.Delta, "  (must be <= delta=", DELTA, " + small tol for a genuine accepted incumbent)")
end

# ---- checkpoint round-trip: serialize + deserialize, confirm content survives ----
ckpt_path = joinpath(CKPT_DIR, "smoke_result.jls")
serialize(ckpt_path, (L = L, kappa = result.kappa, knitro_status = result.knitro_status,
                       n_eval = result.n_eval, n_grad = result.n_grad,
                       best_w = result.best === nothing ? nothing : result.best.w,
                       best_Delta = result.best === nothing ? nothing : result.best.Delta,
                       xsol = result.xsol, timestamp = string(now())))
reloaded = deserialize(ckpt_path)
roundtrip_ok = reloaded.kappa == result.kappa && reloaded.n_eval == result.n_eval
lp(">>> checkpoint round-trip (serialize -> deserialize): ", roundtrip_ok ? "OK" : "MISMATCH")

lp(""); lp("Sanity checks:")
lp("  n_eval > 0 (at least one candidate evaluated): ", result.n_eval > 0)
lp("  n_grad >= 0: ", result.n_grad >= 0)
lp("  kappa finite (if any incumbent found): ", result.best === nothing ? "n/a (no incumbent yet)" : string(isfinite(result.kappa)))
lp("  knitro_status is a recognized code: ", result.knitro_status)

lp("DONE_C14_SMOKE_CM_L50")
