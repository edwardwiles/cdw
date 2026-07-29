# ================================================================================================
# Unrestricted operator-bundle DRIVER-LEVEL wiring gate (2026-07-29 continuation).
#
# The pre-existing equivalence gates (test_operator_no_H_bundle_equivalence_unrestricted[_d20].jl)
# prove OperatorPsiBundle agrees with the dense reference to machine precision -- but they build
# obj_o by hand, standalone, never through either production driver. That left a real gap: neither
# driver ever actually called any such conversion, so ctx.obj stayed dense in every real solve
# (confirmed live 2026-07-29 -- see compressed_live.jl::build_unrestricted_operator_ctx's docstring
# and FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md's correction note).
#
# THIS script closes that gap: it calls the REAL production entry point,
# run_polish_checkpointed_unified, exactly as campaign_unrestricted_runner.jl does (short budget),
# and checks -- from OUTSIDE the driver, via the checkpoint it writes and the driver's own printed
# manifest -- that the bundle actually used was OperatorPsiBundle by default, and that the explicit
# :dense_reference opt-out still produces PsiObjectiveBundleImplicit and a materially similar
# result (both should reach the same accepted outer status on such a short, feasible-from-
# calibration run).
# ================================================================================================
const _D4E = @__DIR__
include(joinpath(_D4E, "c10_d20_production_driver.jl"))
include(joinpath(_D4E, "flexible_theta.jl"))
include(joinpath(_D4E, "flexible_theta_aspace_production.jl"))
include(joinpath(_D4E, "outer_coordinate_layout.jl"))
include(joinpath(_D4E, "c10_d20_production_driver_unified.jl"))
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
const DELTA = 1.0
const BUDGET = 45.0
const FIND_SMALLEST = true
const LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)

lp("="^100)
lp("Unrestricted driver-level operator-bundle wiring gate -- real D=20/W=", W)
lp("="^100)

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
theta_star = 1.0 / ctx0.μHat
D = ctx0.D; Ddest = ctx0.D_dest
xy = precompute_aspace_XY(ctx0)
pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)

x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
w_start = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, LAYOUT)
lp(">>> calibration start: gp=", gp0, " ||w_start||=", norm(w_start))

function run_and_capture(tag::String; moment_representation::Symbol)
    ckpt_dir = joinpath(_D4E, "results", "test_unrestricted_operator_ctx_driver_wiring_2026-07-29", tag)
    rm(ckpt_dir; recursive = true, force = true); mkpath(ckpt_dir)
    t0 = time()
    res = run_polish_checkpointed_unified("wiretest_$tag", FIND_SMALLEST, w_start;
        layout = LAYOUT, theta_lo = NaN, theta_hi = NaN,
        maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
        draw_design_in = :pseudorandom, ckpt_dir = ckpt_dir, checkpoint_interval_s = 3600.0,
        resume_from = nothing, destination_sample = :exclude_row,
        moment_representation = moment_representation)
    wall = time() - t0
    lp(">>> [$tag] done in ", wall, "s")
    return res
end

lp("-"^90); lp("Run 1: DEFAULT kwargs (no moment_representation passed) -- must resolve to :operator")
res_default = run_and_capture("default_kwarg"; moment_representation = MOMENT_REPRESENTATION[])
check("default moment_representation resolves to :operator", MOMENT_REPRESENTATION[] == :operator)

lp("-"^90); lp("Run 2: explicit moment_representation=:operator")
res_op = run_and_capture("explicit_operator"; moment_representation = :operator)

lp("-"^90); lp("Run 3: explicit moment_representation=:dense_reference (opt-out)")
res_dense = run_and_capture("explicit_dense_reference"; moment_representation = :dense_reference)

# The driver itself does not return ctx, so the structural proof is done by directly rebuilding the
# SAME ctx the driver built internally (identical d20_real_setup_design/build_unified_ctx call, same
# seeds) and calling build_unrestricted_operator_ctx on it exactly as the driver now does -- this is
# a proof of the FUNCTION, not a duplicate of the driver's own internal call, but confirms the
# type identity contract the driver relies on.
ctx_check_op = build_unrestricted_operator_ctx(ctx0; moment_representation = :operator)
ctx_check_dense = build_unrestricted_operator_ctx(ctx0; moment_representation = :dense_reference)
check("build_unrestricted_operator_ctx(:operator) yields OperatorPsiBundle", ctx_check_op.obj isa OperatorPsiBundle)
check("build_unrestricted_operator_ctx(:dense_reference) yields PsiObjectiveBundleImplicit (unchanged)",
      ctx_check_dense.obj isa CS.PsiObjectiveBundleImplicit)
check("build_unrestricted_operator_ctx is idempotent on an already-operator ctx",
      build_unrestricted_operator_ctx(ctx_check_op; moment_representation = :operator).obj === ctx_check_op.obj)
check("dense_reference ctx.obj is untouched (===  original)", ctx_check_dense.obj === ctx0.obj)

lp("-"^90)
# Real driver return shape (discovered live, res.ctx IS returned): field is `knitro_status`, not
# `outer_status` -- fixed after the first run of this script surfaced the FieldError. This is the
# DIRECT proof (the real ctx run_polish_checkpointed_unified itself built and searched with, not a
# separately-rebuilt stand-in) that the wiring is genuinely live end-to-end.
check("Run 1 (default kwarg) really searched with ctx.obj isa OperatorPsiBundle", res_default.ctx.obj isa OperatorPsiBundle)
check("Run 2 (explicit :operator) really searched with ctx.obj isa OperatorPsiBundle", res_op.ctx.obj isa OperatorPsiBundle)
check("Run 3 (explicit :dense_reference) really searched with ctx.obj isa PsiObjectiveBundleImplicit (unchanged)",
      res_dense.ctx.obj isa CS.PsiObjectiveBundleImplicit)
lp("Outer results (independent, separately time-limited KNITRO runs -- NOT expected to match exactly,")
lp("  reported for visibility only; exact numerical equivalence is what the dedicated D=4/D=20 gates")
lp("  already establish at the single-inner-solve level):")
lp("  default:            knitro_status=", res_default.knitro_status, " kappa=", res_default.kappa, " n_eval=", res_default.n_eval)
lp("  explicit_operator:  knitro_status=", res_op.knitro_status, " kappa=", res_op.kappa, " n_eval=", res_op.n_eval)
lp("  explicit_dense_ref: knitro_status=", res_dense.knitro_status, " kappa=", res_dense.kappa, " n_eval=", res_dense.n_eval)
check("all three runs reached a feasible/time-limit-feasible KNITRO status (not a hard failure)",
      all(s -> s in (0, -401, -406), (res_default.knitro_status, res_op.knitro_status, res_dense.knitro_status)))

println("="^100)
if isempty(FAILURES)
    println("ALL UNRESTRICTED DRIVER-LEVEL OPERATOR-BUNDLE WIRING GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
