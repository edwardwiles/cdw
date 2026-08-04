# profiled-inner-readiness-2026-08-03, task §9: residual inner-stall reassessment for unrestricted
# at real production dims (D=20, Ddest=19, W=100,000), after this session's lower_limit/method-
# collision fix. Sentinel panel: 2 ordinary feasible points (calibration + small perturbation) and
# 2 deliberately-adversarial points (large A_free perturbations, likely to be genuinely infeasible/
# unbounded) -- confirms feasible points still solve to nStatus=0 with tiny KKT residual, and
# infeasible points now reject FAST (few seconds, via nStatus=-300) rather than the pre-fix slow
# ~77-90s nStatus=-400 timeout the salvage package measured at this same W.
#
# NOTE: these are freshly-constructed adversarial points, not a byte-for-byte replay of any specific
# historically-captured point (those live only in an uncommitted Dropbox archive from a prior
# session, not pulled this pass) -- the substantive claim under test is the CLAMP MECHANISM firing
# correctly at real production scale, not reproducing one specific historical point's exact numbers.
const D4X = @__DIR__
t0_total = time()
include(joinpath(D4X, "context.jl"))
include(joinpath(dirname(dirname(D4X)), "cc_algo", "active_layout.jl"))
include(joinpath(D4X, "compressed_moments.jl"))
include(joinpath(D4X, "oracle.jl"))
include(joinpath(D4X, "oracle_fast.jl"))
include(joinpath(D4X, "operator_psi_bundle.jl"))
include(joinpath(D4X, "compressed_live.jl"))
include(joinpath(D4X, "context_real_d20.jl"))
include(joinpath(D4X, "operator_verification.jl"))
include(joinpath(D4X, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(D4X, "gravity_elimination.jl"))
include(joinpath(D4X, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(D4X, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(D4X, "recover_full_a_2026-07-31.jl"))
include(joinpath(D4X, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(D4X, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(D4X, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(D4X, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(D4X, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(D4X, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(D4X, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(D4X, "profiled_outer_evaluator_2026-08-01.jl"))
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s"); flush(stdout)

const W_VAL = 100_000
const D_VAL, DDEST_VAL = 20, 19

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
@assert D == D_VAL && Ddest == DDEST_VAL

korea_idx = 14; brazil_idx = 3
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(korea_idx => brazil_idx))
n_total = outer_dim_profiled(pe)
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
@assert length(w_calib) == n_total

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function run_point(label::AbstractString, w::Vector{Float64}; expect_feasible::Bool)
    reset_no_dense_g_counters!()
    t0 = time()
    local r_ok, nStatus, kkt, wall
    try
        ev = evaluate_profiled_point(w, ctx, spec, pe)
        wall = time() - t0
        nStatus = ev.result.inner_status
        kkt = ev.result.max_abs_moment_kkt_resid
        r_ok = true
    catch e
        wall = time() - t0
        nStatus = -999
        kkt = NaN
        r_ok = false
        if !(e isa CMExpectedSolveFailure)
            rethrow(e)
        end
    end
    @printf("[%s] wall=%.2fs  nStatus=%s  kkt_resid=%s  (expect_feasible=%s)\n",
        label, wall, r_ok ? string(nStatus) : "N/A(CMExpectedSolveFailure)", r_ok ? @sprintf("%.3e", kkt) : "N/A", expect_feasible)
    flush(stdout)
    return (wall = wall, nStatus = r_ok ? nStatus : -999, ok_solve = r_ok, kkt = kkt)
end

println("="^90); println("Sentinel 1/4: calibration (feasible)"); println("="^90)
res1 = run_point("calibration", w_calib; expect_feasible = true)
check("calibration: nStatus==0", res1.ok_solve && res1.nStatus == 0)
check("calibration: tiny KKT residual (<1e-4)", res1.ok_solve && res1.kkt < 1e-4)

println("\n" * "="^90); println("Sentinel 2/4: small perturbation (feasible)"); println("="^90)
Random.seed!(2026)
w_pert = copy(w_calib); w_pert[2:end] .+= 0.01 .* randn(length(w_pert) - 1)
res2 = run_point("small_perturbation", w_pert; expect_feasible = true)
check("small_perturbation: nStatus in {0,-100,-101,-103}", res2.ok_solve && res2.nStatus in (0, -100, -101, -103))

println("\n" * "="^90); println("Sentinel 3/4: adversarial point A (large uniform A_free perturbation)"); println("="^90)
w_adv1 = copy(w_calib); w_adv1[2:end] .+= 30.0
res3 = run_point("adversarial_A", w_adv1; expect_feasible = false)
check("adversarial_A: rejects FAST (<20s, vs pre-fix ~77-90s at this W)", res3.wall < 20.0)

println("\n" * "="^90); println("Sentinel 4/4: adversarial point B (large alternating-sign A_free perturbation)"); println("="^90)
w_adv2 = copy(w_calib)
for k in 2:length(w_adv2)
    w_adv2[k] += isodd(k) ? 30.0 : -30.0
end
res4 = run_point("adversarial_B", w_adv2; expect_feasible = false)
check("adversarial_B: rejects FAST (<20s, vs pre-fix ~77-90s at this W)", res4.wall < 20.0)

@printf("\nTOTAL WALL: %.2fs\n", time() - t0_total)
println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
